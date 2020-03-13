package io.mindfulmachines.sockeye.transformer

import org.apache.mxnet.{Context, DataDesc, Shape, Symbol}

/**
  * Parameter object used for TrainingModel
  * @param config           Configuration object holding details about the model.
  * @param context          The context(s) that MXNet will be run in (GPU(s)/CPU).
  * @param outputDir        Directory where this model is stored.
  * @param provideData      List of input data descriptions.
  * @param provideLabel     List of label descriptions.
  * @param defaultBucketKey Default bucket key.
  * @param bucketing        If True bucketing will be used, if False the computation graph will always be
  *                         unrolled to the full length
  * @param gradientCompressionParams Optional dictionary of gradient compression parameters.
  * @param gradientAccumulation  Whether to accumulate gradients over batches. Default: False.
  * @param fixedParamNames Optional list of params to fix during training (i.e. their values will not be trained).
  * @param fixedParamStrategy  Optional string indicating a named strategy for fixing parameters.
  * @tparam C
  */
final case class TrainingModelParams[C <: Config](
    config: ModelConfig[C],
    context: List[Context],
    outputDir: String,
    provideData: List[DataDesc],
    provideLabel: List[DataDesc],
    defaultBucketKey: (Int, Int),
    bucketing: Boolean,
    gradientCompressionParams: Option[Map[String, Any]] = None,
    gradientAccumulation: Boolean = false,
    fixedParamNames: Option[List[String]] = None,
    fixedParamStrategy: Option[String] = None
)

/**
  * TrainingModel is a SockeyeModel that fully unrolls over source and target sequences.
  *
  * @param p      References parameter object containing:
  *               -- config           Configuration object holding details about the model.
  *               -- context          The context(s) that MXNet will be run in (GPU(s)/CPU).
  *               -- outputDir        Directory where this model is stored.
  *               -- provideData      List of input data descriptions.
  *               -- provideLabel     List of label descriptions.
  *               -- defaultBucketKey Default bucket key.
  *               -- bucketing        If True bucketing will be used, if False the computation graph will always be
  *                                   unrolled to the full length
  * @param prefix Name prefix for all parameters of this model.
  * @tparam C
  */
class TrainingModel[C <: Config](p: TrainingModelParams[C], prefix: String)
    extends SockeyeModel(p.config, prefix) {

  /**
    * Initializes model components, creates training symbol and module, and binds it.
    * @param provideData    List of input data descriptions.
    * @param provideLabel   List of label descriptions.
    * @param defaultBucketKey  Default bucket key.
    */
  private def initialize(
      provideData: List[DataDesc],
      provideLabel: List[DataDesc],
      defaultBucketKey: (Int, Int)
  ) = {

    val source = Symbol.Variable(Constants.sourceName)
    val sourceWords = Symbol.api
      .split(
        data = Some(source),
        num_outputs = p.config.configEmbedSource.numFactors,
        axis = Some(2),
        squeeze_axis = Some(true)
      )
      .get(0)

    val sourceLength = Utils.computeLength(sourceWords)
    val target       = Symbol.Variable(Constants.targetName)
    val targetLength = Utils.computeLength(target)
    val labels = Symbol.api.reshape(
      data = Some(Symbol.Variable(Constants.targetLabelName)),
      shape = Some(Shape(-1)) // todo should it it be -1, like in sockeye?
    )

    val modelLoss = Loss.getLoss(config = p.config.configLoss)

    println(s"Using model loss: $modelLoss")
    val lengthTaskLoss = p.config.configLengthTaskLoss match {
      case Some(ltl) =>
        println(s"Using length task loss: $ltl")
        Some(Loss.getLengthTaskLoss(ltl))
      case None => None
    }

    val dataNames  = List(Constants.sourceName, Constants.targetName)
    val labelNames = List(Constants.targetLabelName)

    // lengthRatio: (batchSize). Will be pruned if not used

    val lengthRatioSymbol = Symbol.api.broadcast_div(
      rhs = Some(targetLength),
      lhs = Some(sourceLength),
      name = Constants.lenratioLabelName
    )

    // check provide_{data,label} names
    val provideDataNames = provideData.map(d => d.name)

    Utils.checkCondition(
      provideDataNames == dataNames,
      s"incompatible provide_data: $provideDataNames, names should be $dataNames"
    )
    val provideLabelNames = provideLabel.map(d => d.name)
    Utils.checkCondition(
      provideLabelNames == labelNames,
      s"incompatible provide_data: $provideLabelNames, names should be $labelNames"
    )

    /**
      * Returns a (grouped) loss symbol given source & target input lengths. Also returns data and label names for the
      * BucketingModule.
      */
    def symgen(seqLength: (Int, Int)): (Symbol, List[String], List[String]) = {

      val (sourceSeqLength, targetSeqLength) = seqLength

      val sourcePositions = Symbol.arange(0, Some(sourceSeqLength))
      // source embedding
      val (sourceEmbed, sourceEmbedLength, sourceEmbedPositions) =
        embeddingSource.encode(source, Some(sourceLength), sourcePositions)

      val targetPositions = Symbol.arange(0, Some(targetSeqLength))
      // target embedding
      val (targetEmbed, targetEmbedLength, targetEmbedPositions) =
        embeddingTarget.encode(target, Some(targetLength), targetPositions)

      // encoder
      // sourceEncoded: (batchSize, sourceEncodedLength, encoderDepth)
      val (sourceEncoded, sourceEncodedLength, sourceEncodedSeqLength) = encoder
        .encode(
          sourceEmbed.getOrElse(
            throw new RuntimeException("Encoder data cannot be missing at time of training")
          ),
          sourceEmbedLength,
          sourceEmbedPositions
        )

      // decoder
      // targetDecoded: (batchSize, targetLength, decoderDepth)
      val targetDecoded = decoder.decodeSequence(
        DecodeSequenceParams(
          sourceEncoded.getOrElse(
            throw new RuntimeException("Decoder data cannot be missing at time of training")
          ),
          sourceEncodedLength.getOrElse(
            throw new RuntimeException("Decoder data cannot be missing at time of training")
          ),
          seqLength._1,
          targetEmbed.getOrElse(
            throw new RuntimeException("Decoder data cannot be missing at time of training")
          ),
          targetEmbedLength.getOrElse(
            throw new RuntimeException("Decoder data cannot be missing at time of training")
          ),
          seqLength._2,
          targetPositions = targetEmbedPositions
        )
      )

      // targetDecoded: (batchSize, targetSeqLength, decoderDepth)
      val targetDecodedReshape = Symbol.api.reshape(
        data = Some(targetDecoded),
        shape = Some(Shape(-3, 0))
      )

      // output layer
      // logits: (batchSize * targetSeqLength, targetVocabSize)
      val logits = outputLayer(Left(targetDecodedReshape))

      // 1) standard cross-entropy loss
      val netOutputs = List(
        modelLoss.getLoss(
          logits.left.getOrElse(throw new RuntimeException("Must have logit Symbol")),
          labels
        )
      )
      // 2) length task losses
      val updatedNetOutputs = (lengthTaskLoss, lengthRatio) match {
        case (Some(ltl), Some(lr)) =>
          // predictedLengthRatios : (batchSize, 1)
          val predictedLengthRatio = lr(
            sourceEncoded.getOrElse(throw new RuntimeException("Must have source encoded Symbol")),
            sourceEncodedLength.getOrElse(
              throw new RuntimeException("Must have source encoded length Symbol")
            )
          )

          val lossSymbol = ltl match {
            case a: MSELoss     => ltl.getLoss(predictedLengthRatio, lengthRatioSymbol)
            case b: PoissonLoss =>
              //convert ratios to (expected) length estimations for the Poisson loss
              val predictedReferenceLength = predictedLengthRatio * Symbol.api
                .reshape(sourceEncodedLength, shape = Some(Shape(-1, 1)))
              ltl.getLoss(predictedReferenceLength, targetLength)
          }

          // return both the loss symbol, prediction and the computed lengthRatio to be used in metrics
          netOutputs :::
            List(
              lossSymbol,
              Symbol.api.BlockGrad(Some(predictedLengthRatio), name = Constants.lenratioName),
              Symbol.api.BlockGrad(Some(lengthRatioSymbol), name = Constants.lenratioLabelName)
            )
        case (_, _) => netOutputs
      }

      (Symbol.Group(updatedNetOutputs: _*), dataNames, labelNames)

    }

  }

}
