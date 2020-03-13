package io.mindfulmachines.sockeye.transformer

import org.apache.mxnet.{DType, Shape, Symbol}

/**
  * ModelConfig defines model parameters defined at training time which are relevant to model inference.
  * Add new model parameters here. If you want backwards compatibility for models trained with code that did not
  * contain these parameters, provide a reasonable default under default_values.
  *
  * @param configData   Used training data.
  * @param vocabSourceSize Source vocabulary size.
  * @param vocabTargetSize Target vocabulary size.
  * @param configEmbedSource Embedding config for source.
  * @param configEmbedTarget Embedding config for target.
  * @param configEncoder Encoder configuration.
  * @param configDecoder Decoder configuration.
  * @param configLoss  Loss configuration.
  * @param configLengthTaskLoss  Enables weight tying if True.
  * @param doWeightTying Enables weight tying if true.
  * @param weightTyingType  Determines which weights get tied. Must be set if doWeightTying is enabled.
  * @param doWeightNormalization Enables weight normalization if true.
  * @param doLHUC LHUC (Vilar 2018) is applied at some part of the model if true.
  * @param numPointers The number of pointers to the source sequence that can be outputted by the decoder.
  */
final case class ModelConfig[C <: Config](
    configData: Option[DataConfig] = None,
    vocabSourceSize: Int,
    vocabTargetSize: Int,
    configEmbedSource: EmbeddingConfig,
    configEmbedTarget: EmbeddingConfig,
    configEncoder: C,
    configDecoder: C,
    configLoss: LossConfig,
    configLengthTaskLoss: Option[LossConfig] = None,
    configLengthTask: Option[LengthRatioConfig] = None,
    doWeightTying: Boolean = false,
    weightTyingType: Option[String] = Some(Constants.weightTyingTrgSoftmax),
    doWeightNormalization: Boolean = false,
    doLHUC: Boolean = false,
    numPointers: Int = 0
)

trait Model {
  protected def getEmbedWeights(prefix: String): (Symbol, Symbol, Symbol)
}

/**
  * SockeyeModel shares components needed for both training and inference.
  * The main components of a Sockeye model are
  * 1) Source embedding
  * 2) Target embedding
  * 3) Encoder
  * 4) Decoder
  * 5) Output Layer
  *
  * ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
  * time.
  *
  * @param config  Model configuration.
  * @param prefix  Name prefix for all parameters of this model.
  */
class SockeyeModel[C <: Config](config: ModelConfig[C], prefix: String) extends Model {

  //import TransformerEncoderEmbedding._
  val encoder = Encoder.getEncoder(config.configEncoder, prefix)
  //import TransformerDecoderEmbedding._
  val decoder = Decoder.getDecoder(config.configDecoder, prefix)

  //source and target embeddings
  val (embedWeightSource, embedWeightTarget, outWeightTarget) = getEmbedWeights(prefix)

  val embeddingSource = config.configEmbedSource match {
    case a: PassThroughEmbeddingConfig => new PassThroughEmbedding(a)
    case _ =>
      new Embedding(
        config.configEmbedSource,
        prefix = prefix + Constants.sourceEmbeddingPrefix,
        embedWeight = Some(embedWeightSource),
        isSource = true
      )
  }

  val embeddingTarget = new Embedding(
    config.configEmbedTarget,
    prefix + Constants.targetEmbeddingPrefix,
    Some(embedWeightTarget)
  )

  // output layer
  val outputLayer = new OutputLayer(
    OutputParams(
      hiddenSize = decoder.getNumHidden,
      vocabSize = config.vocabTargetSize - config.numPointers,
      weight = Some(outWeightTarget),
      weightNormalization = config.doWeightNormalization,
      prefix = prefix + Constants.defaultOutputLayerPrefix
    )
  )

  // create length ratio prediction layer(s)
  val lengthRatio = config.configLengthTask match {
    case Some(lt) if lt.weight > 0.0f =>
      Some(
        new LengthRatio(
          hiddenSize = encoder.getNumHidden,
          numLayers = lt.numLayers,
          prefix = prefix + Constants.lenratiosOutputLayerPrefix
        )
      )
    case Some(lt) if lt.weight <= 0.0f =>
      println(
        "Auxiliary length task requested, but its loss weight is zero -- this will have no effect."
      )
      None
    case None => None
  }

  // type Option[Map[???]] //Todo move to executor or module?
  val params    = None
  val auxParams = None

  /**
    * Returns embedding parameters for source and target.
    * When source and target embeddings are shared, they are created here and passed in to each side,
    * instead of being created in the Embedding constructors.
    *
    * @param prefix  Prefix.
    * @return  Tuple of source and target parameter symbols.
    */
  override def getEmbedWeights(prefix: String): (Symbol, Symbol, Symbol) = {

    val wEmbedSource = Symbol.Variable(
      prefix + Constants.sourceEmbeddingPrefix + "weight",
      shape = Shape(config.configEmbedSource.vocabSize, config.configEmbedSource.numEmbed)
    )

    val wEmbedTarget = Symbol.Variable(
      prefix + Constants.targetEmbeddingPrefix + "weight",
      shape = Shape(config.configEmbedSource.vocabSize, config.configEmbedSource.numEmbed)
    )

    val wOutTarget = Symbol.Variable(
      name = prefix + "target_output_weight",
      shape = Shape(config.vocabTargetSize - config.numPointers, decoder.getNumHidden),
      dType = DType.Float32
    )

    val (wes, wet, wot) = config.doWeightTying match {
      case true =>
        config.weightTyingType match {
          case a if (a.contains(Constants.weightTyingSrc) & a.contains(Constants.weightTyingTrg)) =>
            val wEmbedTargetTied = Symbol.Variable(
              name = prefix + Constants.sharedEmbeddingPrefix + "weight",
              shape = Shape(config.configEmbedSource.vocabSize, config.configEmbedSource.numEmbed)
            )
            val wEmbedSourceTied = wEmbedTargetTied
            (wEmbedSourceTied, wEmbedTargetTied, wOutTarget)

          case b if (b.contains(Constants.weightTyingSoftmax)) =>
            Utils
              .checkCondition(
                config.configEmbedTarget.numEmbed == decoder.getNumHidden,
                "Weight tying requires target embedding size and decoder hidden size " +
                  s"to be equal: ${config.configEmbedTarget.numEmbed} vs. ${decoder.getNumHidden}"
              )
            val wOutTargetSliced =
              config.numPointers match {
                case x if x > 0 =>
                  Symbol.api.slice(
                    data = Some(wEmbedTarget),
                    begin = Shape(0), //todo shape is 0,None in python source?!
                    end = Shape(config.vocabTargetSize - config.numPointers, 0)
                  ) //todo shape is ..., None in python source?!

                case _ => wEmbedTarget
              }
            (wEmbedSource, wEmbedTarget, wOutTargetSliced)

          case _ => (wEmbedSource, wEmbedTarget, wOutTarget)

        }
      case false => (wEmbedSource, wEmbedTarget, wOutTarget)
    }

    (wes, wet, wot)

  }
}
