package io.mindfulmachines.sockeye.transformer

import org.apache.mxnet.{Shape, Symbol}

trait EmbeddingConfig {
  def vocabSize: Int
  def numEmbed: Int
  def dropout: Float
  def factorConfigs: Option[List[FactorConfig]] = None
  def sourceFactorsCombine: String              = Constants.sourceFactorsCombineConcat
  def dType: String                             = Constants.dTypeFloatPrecision32
  def numFactors: Int                           = 1
}

case class FactorConfig(vocabSize: Int, numEmbed: Int)

case class PassThroughEmbeddingConfig() extends EmbeddingConfig {

  override def vocabSize = 0
  override def numEmbed  = 0
  override def dropout   = 0.0f

}

trait Encoder {

  /**
    * Encodes data given sequence lengths of individual examples and maximum sequence length.
    *
    * @param data Input data
    * @param dataLength Vector with sequence lengths
    * @param positions ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol)

  /**
    *
    * @return The representation size of this encoder.
    */
  def getNumHidden: Int

  /**
    *
    * @return The size of the encoded sequence.
    */
  def getEncodedSeqLength(seqLength: Int): Int = seqLength

  /**
    *
    * @return The maximum length supported by the encoder if such a restriction exists.
    */
  def getMaxSeqLength: Option[Int] = None

  def encodePositions(positions: Symbol, data: Symbol): Option[Symbol] = None

}

/**
  * A sequence of encoders is itself an encoder.
  * @param encoders List of encoders.
  * @param dType Data type.
  */
class EncoderSequence(encoders: List[Encoder], dType: String = Constants.dTypeFloatPrecision32)
    extends Encoder {

  /**
    * Encodes data given sequence lengths of individual examples and maximum sequence length.
    * @param data Input data
    * @param dataLength Vector with sequence lengths
    * @param positions ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) =
    encoders
      .map { e =>
        e.encode(data, dataLength, positions)
      }
      .reverse
      .headOption
      .getOrElse(throw new RuntimeException("at least one encoder required"))

  /**
    * @return The size of the encoded sequence.
    */
  override def getEncodedSeqLength(seqLength: Int): Int =
    encoders
      .map { e =>
        e.getEncodedSeqLength(seqLength)
      }
      .reverse
      .headOption
      .getOrElse(throw new RuntimeException("at least one encoder required"))

  /**
    *
    * @return The maximum length supported by the encoder if such a restriction exists.
    */
  override def getMaxSeqLength: Option[Int] = {
    val f = encoders.flatMap(e => e.getMaxSeqLength)

    f match {
      case Nil => None
      case _   => Some(f.min)
    }
  }

  /**
    *
    * @return The representation size of this encoder.
    */
  override def getNumHidden: Int =
    encoders.reverse.headOption
      .getOrElse(throw new RuntimeException("at least one encoder required"))
      .getNumHidden

  /**
    * Extends sequence with new encoder
    */
  def append(newEncoder: Encoder): Encoder =
    new EncoderSequence(encoders :+ newEncoder, dType)

}

/**
  * Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.
  * @param config  Embedding config.
  * @param prefix  Name prefix for symbols of this encoder.
  * @param embedWeight Optionally use an existing embedding matrix instead of creating a new one.
  * @param isSource  Whether this is the source embedding instance. Default: False.
  */
class Embedding(
    config: EmbeddingConfig,
    prefix: String,
    embedWeight: Option[Symbol] = None,
    isSource: Boolean = false
) extends Encoder {

  val updatedEmbedWeight = embedWeight match {
    case None =>
      Symbol.Variable(prefix + "weight", shape = Shape(config.vocabSize, config.numEmbed))
    case Some(e) => e
  }

  val embedFactorWeights = config.factorConfigs match {
    case None => List.empty[Symbol]
    //Factor weights aren't shared so they're not passed in and we create them here
    case Some(lc) =>
      lc.zipWithIndex.map { fc =>
        Symbol.Variable(
          prefix + s"factor${fc._2}_weight",
          shape = Shape(fc._1.vocabSize, fc._1.numEmbed)
        )

      }
  }

  /**
    * Encodes data given sequence lengths of individual examples and maximum sequence length.
    * @param data Input data
    * @param dataLength Vector with sequence lengths
    * @param positions ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) = {

    val factorEmbeddings: List[Symbol] = isSource match {
      case true =>
        val split = Symbol.api.split(
          data = Some(data),
          num_outputs = config.numFactors,
          axis = Some(2),
          squeeze_axis = Some(true),
          name = prefix + "factor_split"
        )
        val dataList = (0 until config.numFactors).map { e =>
          split.get(e)
        }.toList
        val dataHead =
          dataList.headOption.getOrElse(throw new Exception("numFactors must be greater thatn 0."))
        val dataTail = dataList.tail
        config.factorConfigs match {
          case None => List.empty[Symbol]
          case Some(fc) =>
            dataTail.zip(fc).zip(embedFactorWeights).zipWithIndex.map {
              // factor data, factor config, embed factor weight
              case (((fd, fc), fw), i) =>
                (Symbol.api.Embedding(
                  data = Some(fd),
                  input_dim = fc.vocabSize,
                  weight = Some(fw),
                  output_dim = fc.numEmbed,
                  name = prefix + s"factor${i}_embed"
                ))
            }
        }
      case false => Nil
    }

    val embeddings = Symbol.api.Embedding(
      data = Some(data),
      input_dim = config.vocabSize,
      weight = Some(updatedEmbedWeight),
      output_dim = config.numEmbed,
      name = prefix + "embed"
    )

    val updatedEmbeddings = config.factorConfigs match {
      case None => embeddings
      case Some(_) =>
        config.sourceFactorsCombine match {
          case x if x.equals(Constants.sourceFactorsCombineConcat) =>
            Symbol.api.concat(
              (embeddings :: factorEmbeddings).toArray,
              num_args = factorEmbeddings.size + 1,
              dim = Some(2),
              name = prefix + "embed_plus_factors"
            )
          case _ =>
            Symbol.api
              .add_n((embeddings :: factorEmbeddings).toArray, name = prefix + "embed_plus_factors")
        }
    }

    val dropoutEmbedding = config.dropout match {
      case x if x > 0 =>
        Symbol.api.Dropout(
          data = Some(updatedEmbeddings),
          p = Some(config.dropout),
          name = "source_embed_dropout"
        )
      case _ => updatedEmbeddings
    }

    (Some(dropoutEmbedding), dataLength, positions)
  }

  /**
    * @return The representation size of this encoder.
    */
  override def getNumHidden: Int = config.numEmbed
}

/**
  * Takes an encoded sequence and adds fixed positional embeddings as in Vaswani et al, 2017 to it.
  * @param numEmbed Embedding size.
  * @param scaleUpInput If True, scales input data up by num_embed ** 0.5.
  * @param scaleDownPositions If True, scales positional embeddings down by num_embed ** -0.5.
  * @param prefix Name prefix for symbols of this encoder.
  */
class SinCosPositionalEmbedding(
    numEmbed: Int,
    scaleUpInput: Boolean,
    scaleDownPositions: Boolean,
    prefix: String
) extends Encoder {

  /**
    * @param data Input data (batchSize, sourceSeqLen, numEmbed)
    * @param dataLength Vector with sequence lengths (batchSize)
    * @param positions ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) = {

    val embedding = encodePositions(positions, data)

    (
      embedding,
      dataLength,
      positions
    )

  }

  override def getNumHidden: Int                        = numEmbed
  override def getEncodedSeqLength(seqLength: Int): Int = seqLength
  override def getMaxSeqLength: Option[Int]             = None

  /**
    * @param positions (batchSize)
    * @param data (batchSize, numEmbed)
    * @return (batchSize, numEmbed)
    */
  override def encodePositions(positions: Symbol, data: Symbol): Option[Symbol] = {

    // (batchSize, 1)
    val p = Symbol.api.expand_dims(Some(positions), axis = 1)

    // (numEmbed)
    val channels = Symbol.arange(0, Some(numEmbed / 2))

    val scaling = Symbol.api
      .expand_dims(
        Some(Symbol.ones(Shape(1)) / Symbol.pow(10000, (channels * 2) / numEmbed)),
        axis = 0
      )

    // (batchSize, numEmbed/2)
    val scaledPositions = Symbol.api.dot(Some(p), Some(scaling))

    val sin = Symbol.api.sin(Some(scaledPositions))
    val cos = Symbol.api.cos(Some(scaledPositions))

    // (batchSize, numEmbed)
    val positionalEmbedding = Symbol.api.concat(Array(sin, cos), dim = Some(1), num_args = 2)

    val scaledUpData = scaleUpInput match {
      case true  => data * math.pow(numEmbed, 0.5)
      case false => data
    }

    val scaledDownPositionalEmbedding = scaleDownPositions match {
      case true  => positionalEmbedding * math.pow(numEmbed, -0.5)
      case false => positionalEmbedding
    }

    val positionalEmbeddingGrad = Symbol.api.BlockGrad(Some(scaledDownPositionalEmbedding))

    Some(
      Symbol.api.broadcast_add(Some(scaledUpData), Some(positionalEmbeddingGrad), s"${prefix}add")
    )

  }
}

/**
  * Takes an encoded sequence and adds positional embeddings to it, which are learned jointly. Note that this will
  * limited the maximum sentence length during decoding.
  * @param numEmbed Embedding Size.
  * @param maxSeqLength Maximum sequence length.
  * @param prefix Prefix for symbols of this encoder.
  * @param embedWeight Optionally use an existing embedding matrix instead of creating a new one.
  * @param dType Data type.
  */
class LearnedPositionalEmbedding(
    numEmbed: Int,
    maxSeqLength: Int,
    prefix: String,
    embedWeight: Option[Symbol],
    dType: String
) extends Encoder {

  /**
    * @param data Input data
    * @param dataLength Vector with sequence lengths
    * @param positions ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) = {

    embedWeight match {
      case None    => Some(Symbol.Variable(s"${prefix}weight"))
      case Some(e) => e
    }

    // (1, sourceSeqLength)
    val p = Symbol.api
      .expand_dims(
        data = Some(Symbol.arange(start = 0, stop = Some(maxSeqLength), step = 1)),
        axis = 0
      )

    // (1, sourceSeqLength, numEmbed)
    val positionalEmbedding = Symbol.api
      .Embedding(
        data = Some(positions),
        input_dim = maxSeqLength,
        weight = embedWeight,
        output_dim = numEmbed,
        name = s"${prefix}pos_embed"
      )

    (
      Some(Symbol.api.broadcast_add(Some(data), Some(positionalEmbedding), name = s"${prefix}add")),
      dataLength,
      positions
    )

  }

  override def getNumHidden: Int                        = numEmbed
  override def getEncodedSeqLength(seqLength: Int): Int = seqLength
  override def getMaxSeqLength: Option[Int]             = Some(maxSeqLength)

  /**
    * @param positions (batchSize)
    * @param data (batchSize, numEmbed)
    * @return (batchSize, numEmbed)
    */
  override def encodePositions(positions: Symbol, data: Symbol): Option[Symbol] = {

    // (bachSize, sourceSeqLength, numEmbed)
    val positionalEmbedding = Symbol.api
      .Embedding(
        data = Some(positions),
        input_dim = maxSeqLength,
        weight = embedWeight,
        output_dim = numEmbed,
        name = s"${prefix}pos_embed"
      )

    Some(Symbol.api.broadcast_add(Some(data), Some(positionalEmbedding), name = s"${prefix}add"))

  }
}

/**
  * Simple NoOp pos embedding. It does not modify the data, but avoids lots of if statements.
  */
class NoOpPositionalEmbedding(numEmbed: Int, maxSeqLength: Int, dType: String, prefix: String)
    extends Encoder {

  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) = (Some(data), dataLength, positions)

  override def getEncodedSeqLength(seqLength: Int): Int = seqLength

  override def getMaxSeqLength: Option[Int] = Some(maxSeqLength)

  override def getNumHidden: Int = numEmbed

  override def encodePositions(positions: Symbol, data: Symbol): Option[Symbol] = Some(data)
}

/**
  * This is an embedding which passes through an input symbol without doing any operation.
  *
  * @param config PassThroughEmbeddingConfig config.
  */
class PassThroughEmbedding(config: PassThroughEmbeddingConfig) extends Encoder {

  /**
    *
    * @param data       Input data
    * @param dataLength Vector with sequence lengths
    * @param positions  ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) = (Some(data), dataLength, positions)

  /**
    * @return The representation size of this encoder.
    */
  override def getNumHidden: Int = 0
}

/**
  * Representation learning for very short texts using weighted word embedding aggregation by DeBoom, et al.
  * (https://arxiv.org/pdf/1607.00570.pdf)
  *
  * Thin wrapper around MXNet's Embedding symbol. Works with both time- and batch-major data layouts.
  * Very similar to the existing `Embedding` class. Just adds one more dimension then, smashes them all together.
  *
  *
  * @param config  Embedding config.
  * @param prefix  Name prefix for symbols of this encoder.
  * @param embedWeight Optionally use an existing embedding matrix instead of creating a new one.
  */
class MinMaxEmbedding(
    config: EmbeddingConfig,
    prefix: String,
    embedWeight: Option[Symbol] = None
) extends Encoder {

  val updatedEmbedWeight = embedWeight match {
    case None =>
      Symbol.Variable(prefix + "weight", shape = Shape(config.vocabSize, config.numEmbed))
    case Some(e) => e
  }

  val embedFactorWeights = config.factorConfigs match {
    case None => List.empty[Symbol]
    //Factor weights aren't shared so they're not passed in and we create them here
    case Some(lc) =>
      lc.zipWithIndex.map { fc =>
        Symbol.Variable(
          prefix + s"factor${fc._2}_weight",
          shape = Shape(fc._1.vocabSize, fc._1.numEmbed)
        )

      }
  }

  /**
    * Encodes data given sequence lengths of individual examples and maximum sequence length.
    * @param data Input data
    * @param dataLength Vector with sequence lengths
    * @param positions ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) = {

    //(batchSize, maxEventsNum, maxCodesNum, numEmbed)
    val embeddings = Symbol.api.Embedding(
      data = Some(data),
      input_dim = config.vocabSize,
      weight = Some(updatedEmbedWeight),
      output_dim = config.numEmbed,
      name = prefix + "embed"
    )

    val embeddingMin = Symbol.api.min(
      Some(embeddings),
      axis = Some(Shape(2)),
      keepdims = Some(false)
    )

    //(batchSize, maxEventsNum, numEmbed)
    val embeddingMax = Symbol.api.max(
      Some(embeddings),
      axis = Some(Shape(2)),
      keepdims = Some(false)
    )

    // (batchSize, maxEventsNum, 2 x numEmbed)
    val embeddingMinMax = Symbol.api.concat(
      Array(embeddingMin, embeddingMax),
      num_args = 2,
      dim = Some(2)
    )

    val dropoutEmbedding = config.dropout match {
      case x if x > 0 =>
        Symbol.api.Dropout(
          data = Some(embeddingMinMax),
          p = Some(config.dropout),
          name = "source_embed_dropout"
        )
      case _ => embeddingMinMax
    }

    (Some(dropoutEmbedding), dataLength, positions)
  }

  /**
    * @return The representation size of this encoder.
    */
  override def getNumHidden: Int = config.numEmbed
}

/**
  * Non-recurrent encoder based on the transformer architecture in:
  * Attention Is All You Need, Figure 1 (left)
  * Vaswani et al. (https://arxiv.org/pdf/1706.03762.pdf).
  * @param config  Configuration for transformer encoder.
  * @param prefix Name prefix for operations in this encoder.
  */
class TranformerEncoder(
    config: TransformerConfig,
    prefix: String = Constants.transformerEncoderPrefix
) extends Encoder {

  val layers = (1 until config.numLayers).map { l =>
    new TransformerEncoderBlock(config, s"$prefix$l")
  }.toList

  val finalProcess = new TransformerProcessBlock(
    sequence = config.preProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}final_process_"
  )

  /**
    * Encodes data given sequence lengths of individual examples and maximum sequence length.
    * @param data Input data.
    * @param dataLength Vector with sequence lengths.
    * @param positions ???
    * @return Encoded versions of input data (data, dataLength, seqLength).
    */
  override def encode(
      data: Symbol,
      dataLength: Option[Symbol],
      positions: Symbol
  ): (Option[Symbol], Option[Symbol], Symbol) = {

    val castData = Utils.castConditionally(data, config.dtype)

    val dropoutData = config.dropoutPrePost match {
      case d if d > 0.0f =>
        Symbol.api.Dropout(data = Some(castData), p = Some(config.dropoutPrePost))
      case _ => castData
    }

    val expandDimBias = Symbol.api.expand_dims(
      Some(
        TransformerBlock.getValidMaskFor(
          data = dropoutData,
          lengths =
            dataLength.getOrElse(throw new RuntimeException("Invalid length for input data!")),
          numHeads = Some(config.attentionHeads),
          foldHeads = true,
          name = Some(s"${prefix}bias")
        )
      ),
      axis = 1
    )

    val castBias = Utils.castConditionally(expandDimBias, config.dtype)

    // (batchSize, SeqLength, config.modelSize)
    val layerData = layers.foldLeft(dropoutData) {
      case (d, l) => l(d, castBias)
    }

    val processData = finalProcess(layerData, prev = None)

    val uncastData = Utils.uncastConditionally(processData, config.dtype)

    (Some(uncastData), dataLength, positions)

  }

  /**
    * @return The representation size of this encoder.
    */
  override def getNumHidden: Int = config.modelSize

  override def getEncodedSeqLength(seqLength: Int): Int = seqLength

}

object TransformerEncoderEmbedding {

  def genSinCosPositionalEmbedding(
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

  def genLearnedPositionalEmbedding(
      config: TransformerConfig,
      prefix: String
  ): Encoder = {
    new LearnedPositionalEmbedding(
      numEmbed = config.modelSize,
      maxSeqLength = config.maxSeqLenSource,
      prefix = prefix + Constants.learnedPositionalEmbedding,
      embedWeight = None,
      dType = config.dtype
    )
  }

  def genNoOpPositionalEmbedding(
      config: TransformerConfig,
      prefix: String
  ): Encoder = {
    new NoOpPositionalEmbedding(
      numEmbed = config.modelSize,
      maxSeqLength = config.maxSeqLenSource,
      dType = config.dtype,
      prefix = prefix + Constants.noPositionalEmbedding
    )
  }
}

object TransformerEncoder {

  /**
    * Returns a Transformer encoder, consisting of an embedding layer with
    * positional encodings and a TransformerEncoder instance.
    *
    * @param config Configuration for transformer encoder.
    * @param prefix Prefix for variable names
    * @return Encoder instance.
    */
  def getTransformerEncoder(config: TransformerConfig, prefix: String): Encoder = {

    val embedding = config.embeddingType match {
      case SinCosEmbeddingType() =>
        TransformerEncoderEmbedding.genSinCosPositionalEmbedding(config, prefix)
      case LearnedEmbeddingType() =>
        TransformerEncoderEmbedding.genLearnedPositionalEmbedding(config, prefix)
      case NoOpEmbeddingType() =>
        TransformerEncoderEmbedding.genNoOpPositionalEmbedding(config, prefix)
    }

    val encoderSequence = new EncoderSequence(List(embedding), config.dtype)

    encoderSequence.append(new TranformerEncoder(config, prefix))
  }
}

object Encoder {

  def getEncoder[C <: Config](config: C, prefix: String): Encoder = {

    import TransformerEncoderEmbedding._

    config match {
      case a: TransformerConfig => TransformerEncoder.getTransformerEncoder(a, prefix)
    }
  }
}
