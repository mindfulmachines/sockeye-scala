package io.mindfulmachines.sockeye.transformer

import org.apache.mxnet.{DataDesc, Shape, Symbol}

sealed trait State

case class DecoderState(
    sourceEncoded: Symbol,
    sourceEncodedLengths: Symbol,
    cache: List[Symbol] = Nil
) extends State

/**
  * @param sourceEncoded              Encoded source: (batchSize, sourceEncodedMaxLength, encoderDepth).
  * @param sourceEncodedLengths       Lengths of encoded source sequences. Shape: (batchSize).
  * @param sourceEncodedMaxLength     Size of encoder time dimension.
  * @param targetEmbed                Embedded target sequence. Shape: (batchSize, targetEmbedMaxLength, targetNumEmbed).
  * @param targetEmbedLengths         Lengths of embedded target sequences. Shape: (batchSize,).
  * @param targetEmbedMaxLength       Dimension of the embedded target sequence.
  */
final case class DecodeSequenceParams(
    sourceEncoded: Symbol,
    sourceEncodedLengths: Symbol,
    sourceEncodedMaxLength: Int,
    targetEmbed: Symbol,
    targetEmbedLengths: Symbol,
    targetEmbedMaxLength: Int,
    targetPositions: Symbol
)

trait Decoder {

  //def getDecoder(config: C, prefix: Option[String] = None): Decoder[C]

  def dtype: String

  /**
    * Decodes a sequence of embedded target words and returns sequence of last decoder representations for each time step.
    *
    * @param p References parameter object containing:
    *                    -- sourceEncoded           Encoded source: (batchSize, sourceEncodedMaxLength, encoderDepth).
    *                    -- sourceEncodedLengths    Lengths of encoded source sequences. Shape: (batchSize).
    *                    -- sourceEncodedMaxLength  Size of encoder time dimension.
    *                    -- targetEmbed             Embedded target sequence. Shape: (batchSize, targetEmbedMaxLength, targetNumEmbed).
    *                    -- targetEmbedLenths       Lengths of embedded target sequences. Shape: (batchSize,).
    *                    -- targetEmbedMaxLength    Dimension of the embedded target sequence.
    * @return  Decoder data. Shape: (batchSize, targetEmbedMaxLength, decoderDepth).
    */
  def decodeSequence(p: DecodeSequenceParams): Symbol

  /**
    * Decodes a single time step given the current step, the previous embedded target word,
    * and previous decoder states.
    * Returns decoder representation for the next prediction, attention probabilities, and next decoder states.
    * Implementations can maintain an arbitrary number of states.
    *
    * @param step  Global step of inference procedure, starts with 1.
    * @param targetEmbedPrev  Previous target word embedding. Shape: (batch_size, target_num_embed).
    * @param sourceEmbedMaxLength  Length of encoded source time dimension.
    * @param states  Arbitrary list of decoder states.
    * @return  logit inputs, attention probabilities, next decoder states.
    */
  def decodeStep(
      step: Int,
      targetEmbedPrev: Symbol,
      sourceEmbedMaxLength: Int,
      states: DecoderState
  ): (Symbol, Symbol, DecoderState)

  /* Reset decoder method. Used for infernce. */
  def reset

  /* The representation size of this decoder. */
  def getNumHidden: Int

  /**
    * Returns a list of symbolic states that represent the initial states of this decoder.
    * Used for inference.
    *
    * @param sourceEncdoed Encoded source. Shape: (batchSize, sourceEncodedMaxLength, encoderDepth).
    * @param sourceEncodedLengths  Lengths of encoded source sequences. Shape: (batchSize).
    * @param sourceEncodedMaxLength Size of encoder time dimension.
    * @return  List of symbolic initial states.
    */
  def initState(
      sourceEncdoed: Symbol,
      sourceEncodedLengths: Symbol,
      sourceEncodedMaxLength: Int
  ): DecoderState

  /**
    * Returns the list of symbolic variables for this decoder to be used during inference.
    *
    * @param targetMaxLength Current target sequence lengths.
    * @return  List of symbolic variables.
    */
  def stateVariables(targetMaxLength: Int): List[Symbol]

  /**
    * Returns a list of shape descriptions given batch size, encoded source max length and encoded source depth.
    * Used for inference.
    *
    * @param batchSize  Batch size during inference.
    * @param targetMaxLength  Current target sequence length.
    * @param sourceEncodedMaxLength  Size of encoder time dimension.
    * @param sourceEncodedDepth  Depth of encoded source.
    * @return  List of shape descriptions
    */
  def stateShapes(
      batchSize: Int,
      targetMaxLength: Int,
      sourceEncodedMaxLength: Int,
      sourceEncodedDepth: Int
  ): List[DataDesc]

  /* The maximum length supported by the decoder if such a restriction exists. */
  def getMaxSeqLength: Option[Int]

}

object TransformerDecoderEmbedding {

  implicit def genSinCosPositionalEmbedding(
      config: TransformerConfig,
      prefix: String
  ): Encoder = {
    new SinCosPositionalEmbedding(
      numEmbed = config.modelSize,
      scaleUpInput = true,
      scaleDownPositions = false,
      prefix = prefix + Constants.fixedPositionalEmbedding
    )
  }

  implicit def genLearnedPositionalEmbedding(
      config: TransformerConfig,
      prefix: String
  ): Encoder = {
    new LearnedPositionalEmbedding(
      numEmbed = config.modelSize,
      maxSeqLength = config.maxSeqLenTarget,
      prefix = prefix + Constants.learnedPositionalEmbedding,
      embedWeight = None,
      dType = config.dtype
    )
  }

  implicit def genNoOpPositionalEmbedding(
      config: TransformerConfig,
      prefix: String
  ): Encoder = {
    new NoOpPositionalEmbedding(
      numEmbed = config.modelSize,
      maxSeqLength = config.maxSeqLenTarget,
      dType = config.dtype,
      prefix = prefix + Constants.noPositionalEmbedding
    )
  }
}

/**
  * Transformer decoder as in Vaswani et al, 2017: Attention is all you need.
  * In training, computation scores for each position of the known target sequence are compouted in parallel,
  * yielding most of the speedup.
  * At inference time, the decoder block is evaluated again and again over a maximum length input sequence that is
  * initially filled with zeros and grows during beam search with predicted tokens. Appropriate masking at every
  * time-step ensures correct self-attention scores and is updated with every step.
  *
  * @param config Transformer configuration.
  * @param prefix Name prefix for symbols of this decoder
  */
class TransformerDecoder(
    config: TransformerConfig,
    prefix: String = Constants.transformerDecoderPrefix
) extends Decoder {

  val layers = (0 until config.numLayers).map { l =>
    new TransformerDecoderBlock(config = config, prefix = s"$prefix${l}_")
  }.toList

  val finalProcess = new TransformerProcessBlock(
    sequence = config.preProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}final_process_"
  )

  val positionalEmbedding = config.embeddingType match {
    case SinCosEmbeddingType() =>
      TransformerDecoderEmbedding.genSinCosPositionalEmbedding(config, prefix)
    case LearnedEmbeddingType() =>
      TransformerDecoderEmbedding.genLearnedPositionalEmbedding(config, prefix)
    case NoOpEmbeddingType() =>
      TransformerDecoderEmbedding.genNoOpPositionalEmbedding(config, prefix)
  }

  override def dtype: String = config.dtype

  /**
    * Decodes a sequence of embedded target words and returns sequence of last decoder representations for each time step.
    *
    * @param p References parameter object containing:
    *                    -- sourceEncoded           Encoded source: (batchSize, sourceEncodedMaxLength, encoderDepth).
    *                    -- sourceEncodedLengths    Lengths of encoded source sequences. Shape: (batchSize).
    *                    -- sourceEncodedMaxLength  Size of encoder time dimension.
    *                    -- targetEmbed             Embedded target sequence. Shape: (batchSize, targetEmbedMaxLength, targetNumEmbed).
    *                    -- targetEmbedLengths       Lengths of embedded target sequences. Shape: (batchSize,).
    *                    -- targetEmbedMaxLength    Dimension of the embedded target sequence.
    * @return  Decoder data. Shape: (batchSize, targetEmbedMaxLength, decoderDepth).
    */
  override def decodeSequence(p: DecodeSequenceParams): Symbol = {

    val sourceBias = TransformerBlock.getValidMaskFor(
      data = p.sourceEncoded,
      lengths = p.sourceEncodedLengths,
      numHeads = Some(config.attentionHeads),
      foldHeads = true,
      name = Some(s"${prefix}source_bias")
    )

    val sourceBiasExpandDims = Symbol.api.expand_dims(
      data = Some(sourceBias),
      axis = 1
    )

    val targetBias = TransformerBlock.getAutoRegressiveBias(p.targetEmbedMaxLength)

    val (target, _, targetMaxLength) =
      positionalEmbedding.encode(p.targetEmbed, None, p.targetPositions)

    val targetDropout = config.dropoutPrePost match {
      case d if d > 0.0f => Symbol.api.Dropout(data = target, p = Some(config.dropoutPrePost))
      case _             => target.getOrElse(throw new Exception("target must have value."))
    }

    val layersTarget = layers.foldLeft(targetDropout) {
      case (t, l) =>
        l(
          target = t,
          targetBias = targetBias,
          source = p.sourceEncoded,
          sourceBias = sourceBiasExpandDims
        )
    }

    val targetFinal = finalProcess(data = layersTarget, prev = None)

    targetFinal

  }

  /**
    * Decodes a single time step given the current step, the previous embedded target word, and previous decoder states.
    * Returns decoder representation for the next prediction, attention probabilities, and next decoder states.
    * Implementations can maintain an arbitrary number of states.
    *
    * @param step  Global step of inference procedure, starts with 1.
    * @param targetEmbedPrev  Previous target word embedding. Shape: (batch_size, target_num_embed).
    * @param sourceEmbedMaxLength  Length of encoded source time dimension.
    * @param states  Arbitrary list of decoder states.
    * @return  logit inputs, attention probabilities, next decoder states.
    */
  override def decodeStep(
      step: Int,
      targetEmbedPrev: Symbol,
      sourceEmbedMaxLength: Int,
      states: DecoderState
  ): (Symbol, Symbol, DecoderState) = {

    // for step > 1, states contains sourceEncodded, sourceEncodedLengths, and cache tensors.
    val (sourceEncoded, sourceEncodedLengths, cache) =
      (states.sourceEncoded, states.sourceEncodedLengths, states.cache)

    // symbolic indices of the previous word
    val indices = Symbol.arange(
      start = step - 1.toFloat,
      stop = Some(step.toFloat),
      step = 1f,
      name = "indices"
    )

    // (batchSize, numEmbed)
    val targetEmbedPrevEmbedPos = positionalEmbedding.encodePositions(indices, targetEmbedPrev)

    // (batchSize, 1, numEmbed)
    val target = Symbol.api.expand_dims(
      data = targetEmbedPrevEmbedPos,
      axis = 1
    )

    // (batchSize * heads, maxLength)
    val sourceBias = TransformerBlock.getValidMaskFor(
      data = sourceEncoded,
      lengths = sourceEncodedLengths,
      numHeads = Some(config.attentionHeads),
      foldHeads = true,
      name = Some(s"${prefix}source_bias")
    )

    // (batchSize * heads, 1, maxLength)
    val sourceBiasExpandDims = Symbol.api.expand_dims(Some(sourceBias), axis = 1)

    // auto-regressive bias for last position in sequence
    // (1, targetMaxLength, targetMaxLength)
    val targetBias = TransformerBlock.getAutoRegressiveBias(step)
    val targetBiasSliceAxis =
      Symbol.api.slice_axis(Some(targetBias), axis = 1, begin = -1, end = step)

    val existingLayerCaches = getCachePerLayer(cache)

    // store updated keys and values in states list.
    // In python, (layer.__call__() has the side-effect of updating contents of layer_cache)

    val updatedLayerCaches = layers.zip(existingLayerCaches).map {

      case (l, lc) =>
        val tdb = l(
          target = target,
          targetBias = targetBiasSliceAxis,
          source = sourceEncoded,
          sourceBias = sourceBiasExpandDims,
          cache = Some(lc)
        )
        tdb
    }

    val finalStates = DecoderState(sourceEncoded, sourceEncodedLengths, Nil ++ updatedLayerCaches)

    // (batchSize, 1, modelSize)
    val targetFinalProcess = finalProcess(data = target, prev = None)
    // (batchSize, modelSize)
    val targetReshape =
      Symbol.api.reshape(data = Some(targetFinalProcess), shape = Some(Shape(-3, -1)))

    // TODO(fhieber): no attention probs for now
    val attentionProbs = Symbol.api.sum(
      data = Some(Symbol.api.zeros_like(data = Some(finalStates.sourceEncoded))),
      axis = Some(Shape(2)),
      keepdims = Some(false)
    )

    (targetReshape, attentionProbs, finalStates)

  }

  /**
    * For decoder time steps > 1 there will be cache tensors available that contain
    * previously computed key & value tensors for each transformer layer.
    *
    *
    * @param cache caches (as a List of Symbol) passed to DecoderState in `decodeStep`.
    * @return List of layer cache dictionaries.
    */
  private def getCachePerLayer(cache: List[Symbol]): List[Map[String, Option[Symbol]]] = {
    cache match {
      case Nil => (0 until layers.length).map(_ => Map("" -> Option.empty[Symbol])).toList
      case _ =>
        assert(cache.length == layers.length * 2)
        (0 until layers.length).map { l =>
          Map(cache(2 * l + 0).toString -> Some(cache(2 * l + 1)))
        }.toList

    }
  }

  override def reset: Unit = ()

  override def getNumHidden: Int = config.modelSize

  /**
    * Returns a list of symbolic states that represent the initial states of this decoder.
    * Used for inference.
    *
    * @param sourceEncoded Encoded source. Shape: (batchSize, sourceEncodedMaxLength, encoderDepth).
    * @param sourceEncodedLengths  Lengths of encoded source sequences. Shape: (batchSize).
    * @param sourceEncodedMaxLength Size of encoder time dimension.
    * @return  List of symbolic initial states.
    */
  override def initState(
      sourceEncoded: Symbol,
      sourceEncodedLengths: Symbol,
      sourceEncodedMaxLength: Int
  ): DecoderState =
    DecoderState(sourceEncoded, sourceEncodedLengths, Nil)

  /**
    * Returns the list of symbolic variables for this decoder to be used during inference.
    * @param targetMaxLength Current target sequence lengths.
    * @return  List of symbolic variables.
    */
  override def stateVariables(targetMaxLength: Int): List[Symbol] = {

    val initVariables = List(
      Symbol.Variable(name = Constants.sourceEncodedName),
      Symbol.Variable(name = Constants.sourceLengthName)
    )

    val updatedVariables = targetMaxLength match {
      // no cache for initial decoder step
      case x if x > 1 =>
        (0 until layers.length).flatMap { l =>
          val k = initVariables :+ Symbol.Variable(s"cache_l${l}_k")
          k :+ Symbol.Variable(s"cache_l${l}_v")

        }
      case _ => initVariables
    }

    updatedVariables.toList
  }

  /**
    * Returns a list of shape descriptions given batch size, encoded source max length and encoded source depth.
    * Used for inference.
    *
    * @param batchSize  Batch size during inference.
    * @param targetMaxLength  Current target sequence length.
    * @param sourceEncodedMaxLength  Size of encoder time dimension.
    * @param sourceEncodedDepth  Depth of encoded source.
    * @return  List of shape descriptions
    */
  override def stateShapes(
      batchSize: Int,
      targetMaxLength: Int,
      sourceEncodedMaxLength: Int,
      sourceEncodedDepth: Int
  ): List[DataDesc] = {

    val initShapes = List(
      DataDesc(
        name = Constants.sourceEncodedName,
        shape = Shape(batchSize, sourceEncodedMaxLength, sourceEncodedDepth),
        layout = Constants.batchMajor
      ),
      DataDesc(name = Constants.sourceLengthName, shape = Shape(batchSize), layout = "N")
    )

    val updatedShapes = targetMaxLength match {
      // no cache for initial decoder step
      case x if x > 1 =>
        (0 until layers.length).flatMap { l =>
          val k = initShapes :+ DataDesc(
            name = s"cache_l${l}_k",
            shape = Shape(batchSize, targetMaxLength - 1, config.modelSize),
            layout = Constants.batchMajor
          )
          k :+ DataDesc(
            name = s"cache_l${l}_v",
            shape = Shape(batchSize, targetMaxLength - 1, config.modelSize),
            layout = Constants.batchMajor
          )

        }
      case _ => initShapes
    }

    updatedShapes.toList

  }

  override def getMaxSeqLength: Option[Int] = positionalEmbedding.getMaxSeqLength

}

object TransformerDecoder {

  /**
    * Light wrapper around TransformerDecoder class for consistent style.
    *
    * @param config Configuration for transformer decoder.
    * @param prefix Prefix for variable names
    * @return Decoder instance.
    */
  def getTransformerDecoder(
      config: TransformerConfig,
      prefix: String
  ): Decoder =
    new TransformerDecoder(config, prefix)

}

object Decoder {

  def getDecoder[C <: Config](config: C, prefix: String): Decoder = {

    import TransformerDecoderEmbedding._

    config match {
      case a: TransformerConfig => TransformerDecoder.getTransformerDecoder(a, prefix)
    }
  }
}
