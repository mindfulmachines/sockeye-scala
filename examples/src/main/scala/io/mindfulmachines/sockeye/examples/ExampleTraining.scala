package io.mindfulmachines.sockeye.examples

import io.mindfulmachines.sockeye.transformer.{AttentionParams, Constants, DecodeSequenceParams, Loss, MSELoss, MinMaxEmbedding, MultiHeadAttention, PoissonLoss, SockeyeModel, TrainingModelParams, TransformerConfig, Utils}
import io.mindfulmachines.sockeye.transformer.{AttentionParams, Config, Constants, Loss, MinMaxEmbedding, MultiHeadAttention, SockeyeModel, TrainingModelParams, TransformerConfig, Utils}
import org.apache.mxnet.{Context, DataDesc, Shape, Symbol}

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
class ExampleTraining[C <: Config](p: TrainingModelParams[C], prefix: String)
    extends SockeyeModel(p.config, prefix) {

  /**
    * Initializes model components, creates training symbol and module, and binds it.
    * @param provideData    List of input data descriptions.
    * @param provideLabel   List of label descriptions.
    * @param defaultBucketKey  Default bucket key.
    */
  def initialize(
      provideData: List[DataDesc],
      provideLabel: List[DataDesc],
      defaultBucketKey: (Int, Int)
  ): Symbol = {

    val source = Symbol.Variable(Constants.sourceName)

    val labels = Symbol.api.reshape(
      data = Some(Symbol.Variable(Constants.targetLabelName)),
      shape = Some(Shape(-1)) // todo should it it be -1, like in sockeye?
    )

    val modelLoss = Loss.getLoss(config = p.config.configLoss)

    val dataNames  = List(Constants.sourceName)
    val labelNames = List(Constants.targetLabelName)

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

    val sourceMinMaxEmbedding = new MinMaxEmbedding(
      config = p.config.configEmbedSource,
      prefix = prefix + Constants.sourceEmbeddingPrefix,
      embedWeight = Some(embedWeightSource)
    )

    /**
      * Returns a (grouped) loss symbol given maxEventsNum & maxCodesNum.
      */
    def symgen(maxEventsNum: Int, maxCodesNum: Int): (Symbol, List[String], List[String]) = {

      // (batchSize, maxEventsNum)
      val sourcePositions: Symbol = Symbol.api.slice(
        Some(source),
        begin = Shape(0, 0),
        end = Shape(10000, maxEventsNum)
      )

      // (batchSize, maxEventsNum * maxCodesNum)
      val diagsSlice: Symbol = Symbol.api.slice(
        Some(source),
        begin = Shape(0, maxEventsNum),
        end = Shape(10000, maxEventsNum + maxEventsNum * maxCodesNum)
      )

      // (batchSize, maxEventsNum, maxCodesNum)
      val diags: Symbol = Symbol.api.reshape(
        Some(diagsSlice),
        shape = Some(Shape(0, maxEventsNum, maxCodesNum))
      )

      // (batchSize, maxEventsNum * maxCodesNum)
      val procsSlice: Symbol = Symbol.api.slice(
        Some(source),
        begin = Shape(0, maxEventsNum + maxEventsNum * maxCodesNum),
        end = Shape(10000, maxEventsNum + 2 * maxEventsNum * maxCodesNum)
      )

      // (batchSize, maxEventsNum, maxCodesNum)
      val procs: Symbol = Symbol.api.reshape(
        Some(procsSlice),
        shape = Some(Shape(0, maxEventsNum, maxCodesNum))
      )

      // (batchSize, maxEventsNum, 2*maxCodesNum)
      val diagsProcs = Symbol.api.concat(
        Array(diags, procs),
        num_args = 2,
        dim = Some(2)
      )

      val sourceLength: Symbol = Utils.computeLength(sourcePositions)

      // (batchSize, maxEventsNum, 2 * p.config.configEmbedSource.numFactors)
      val (sourceEmbed, sourceEmbedLength, sourceEmbedPositions) =
        sourceMinMaxEmbedding.encode(diagsProcs, Some(sourceLength), sourcePositions)

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

      val encoderConfig = p.config.configEncoder match {
        case t: TransformerConfig => t
        case _                    => throw new Exception("TransformerConfig expected")
      }

      val decoderConfig = p.config.configDecoder match {
        case t: TransformerConfig => t
        case _                    => throw new Exception("TransformerConfig expected")
      }

      val decoder = new MultiHeadAttention(
        AttentionParams(
          prefix = Constants.decoderPrefix,
          depthAttention = encoderConfig.modelSize,
          heads = decoderConfig.attentionHeads,
          depthOut = decoderConfig.modelSize,
          dropout = decoderConfig.dropoutAttention
        )
      )

      val sourceDecoded = decoder.apply(
        queries = Symbol.ones(Shape(16, 1, encoderConfig.modelSize)),
        memory = sourceEncoded.getOrElse(throw new Exception("sourceEncoded is required"))
      )

      // (batchSize *targetSeqLength, encoderDepth)
      val sourceEncodedReshape = Symbol.api.reshape(
        data = Some(sourceDecoded),
        shape = Some(Shape(-3, 0))
      )

      // output layer
      // logits: (batchSize * targetSeqLength, targetVocabSize)
      val logits = outputLayer(
        Left(
          sourceEncodedReshape
        )
      )

      //standard cross-entropy loss
      val netOutputs =
        modelLoss.getLoss(
          logits.left.getOrElse(throw new RuntimeException("Must have logit Symbol")),
          labels
        )

      (Symbol.Group(netOutputs), dataNames, labelNames)

    }

    val (lossSymbol, _, _) = symgen(defaultBucketKey._1, defaultBucketKey._2)
    lossSymbol
  }

}
