package io.mindfulmachines.sockeye.transformer

import org.apache.mxnet.Symbol
import org.apache.mxnet.Shape

object TransformerBlock {

  /**
    * Returns bias/mask for variable sequence lengths.
    * @param data Input data to mask. Shape: (batch, seqLength, _).
    * @param lengths Sequence lengths. Shape: (batch).
    * @param numHeads Number of attention heads.
    * @param foldHeads Whether to fold heads dimension into batch dimension.
    * @param name Name of symbol.
    * @return Bias symbol. Shape: (batch, seqLength)
    */
  def getValidMaskFor(
      data: Symbol,
      lengths: Symbol,
      numHeads: Option[Int] = None,
      foldHeads: Boolean = true,
      name: Option[String] = None
  ): Symbol = {

    val zerosLike = Symbol.api.zeros_like(data = Some(lengths))

    // (batch, 1)
    val zeros = Symbol.api.reshape(data = Some(zerosLike), shape = Some(Shape(-1, 1)))

    // (batch, seqLength)
    val zerosBroadcast = Symbol.api.broadcast_like(
      lhs = Some(zeros),
      rhs = Some(data),
      lhs_axes = Some(Shape(1)),
      rhs_axes = Some(Shape(1))
    )

    // (batchSize, maxLength)
    val x = Symbol.api.SequenceMask(
      data = Some(zerosBroadcast),
      use_sequence_length = Some(true),
      sequence_length = Some(lengths),
      axis = Some(1),
      value = Some(Constants.largeNegativeValue)
    )

    val xBroadcast = numHeads match {
      case Some(h) => Layers.broadcastToHeads(x = x, numHeads = h, nDim = 2, foldHeads = foldHeads)
      case None    => x
    }
    Symbol.api.BlockGrad(data = Some(xBroadcast), name = s"${name}_bias")

  }

  /**
    * Returns bias/mask to ensure position i can only attend to positions <i.
    * @param maxLength Sequence length.
    * @param dType dtype of bias
    * @return Bias symbol of shape (1, max_length, max_length).
    */
  def getAutoRegressiveBias(
      maxLength: Int,
      dType: String = Constants.dTypeFloatPrecision32
  ): Symbol = {

    val lengthArray: Symbol = ???

    //matrix with lower triangle and main diagonal set to 0, upper triangle set to 1

    val bias = Symbol.api.broadcast_greater(
      lhs = Some(Symbol.api.reshape(Some(lengthArray), shape = Some(Shape(1, -1)))),
      rhs = Some(Symbol.api.reshape(Some(lengthArray), shape = Some(Shape(-1, 1))))
    )

    val biasSmallValue = bias * -Constants.largeValues(dType)

    val biasReshape =
      Symbol.api.reshape(data = Some(biasSmallValue), shape = Some(Shape(1, maxLength, maxLength)))

    Symbol.api.BlockGrad(Some(biasReshape))

  }

}

/**
  * Position-wise feed-forward network with activation.
  */
class TransformerFeedForward(
    numHidden: Int,
    numModel: Int,
    actType: String,
    dropout: Float,
    prefix: String
) {

  val w_i2h = Symbol.Variable(s"${prefix}_i2h_weight")
  val b_i2h = Symbol.Variable(s"${prefix}_i2h_bias")
  val w_h2o = Symbol.Variable(s"${prefix}_h2o_weight")
  val b_h2o = Symbol.Variable(s"${prefix}_h2o_bias")

  /**
    * Position-wise feed-forward network with activation.
    * @param x Symbol of shape (batchSize, seqLength, numHidden)
    * @return Symbol of shape (batchSize, seqLength, numHidden)
    */
  def apply(x: Symbol): Symbol = {

    val fcH = Symbol.api.FullyConnected(
      data = Some(x),
      num_hidden = numHidden,
      weight = Some(w_i2h),
      bias = Some(b_i2h),
      flatten = Some(false)
    )

    val activatedH = dropout match {
      case d if d > 0.0f => fcH
      case _             => Layers.activation(fcH, actType)
    }

    val y = Symbol.api.FullyConnected(
      data = Some(activatedH),
      num_hidden = numModel,
      weight = Some(w_h2o),
      bias = Some(b_h2o),
      flatten = Some(false)
    )

    y

  }

}

/**
  * Block to perform pre/post processing on layer inputs.
  * @param sequence List of processes performed on layer inputs, it can have one of three operations at each step:
  *                 N: layer normalization
  *                 R: residual connection
  *                 D: dropout
  */
class TransformerProcessBlock(sequence: List[PrePost], dropout: Float, prefix: String) {

  lazy val layerNorm = new LayerNormalization(prefix = s"${prefix}_norm")

  /**
    * Apply processing sequence to data with optional previous input.

    * @param data Input data. Shape: (batch, length, numHidden).
    * @param prev Previous data. Shape: (batch, length, numHidden).
    * @return Processed data. Shape: (batch, length, numHidden).
    */
  def apply(data: Symbol, prev: Option[Symbol]): Symbol = {

    val processedData = sequence match {
      case Nil => data
      case list: List[PrePost] =>
        prev match {
          case None =>
            assert(!list.contains(R), "Residual connection not allowed if no previous value given.")
          case _ =>
        }

        list.foldLeft(data) { (d, e) =>
          e match {
            case R =>
              prev match {
                case Some(p) => d + p
                case None    => d
              }
            case N => layerNorm(d)
            case D =>
              dropout match {
                case drop if drop > 0.0f =>
                  Symbol.api.Dropout(data = Some(d), p = Some(dropout), name = s"${prefix}_dropout")
                case _ => d
              }
            case _ => throw new NoSuchFieldException(s"Unknown step in sequence: $e")
          }
        }
    }

    processedData
  }

}

/**
  * A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
  * in between.
  */
class TransformerEncoderBlock(config: TransformerConfig, prefix: String) {

  val preSelfAttention = new TransformerProcessBlock(
    sequence = config.preProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_att_self_pre"
  )

  val selfAttention = new MultiHeadSelfAttention(
    a = AttentionParams(
      depthAttention = config.modelSize,
      heads = config.attentionHeads,
      depthOut = config.modelSize,
      dropout = config.dropoutAttention,
      prefix = s"${prefix}_att_self"
    )
  )

  val postSelfAttention = new TransformerProcessBlock(
    sequence = config.postProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_att_self_post"
  )
  val preFF = new TransformerProcessBlock(
    sequence = config.preProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_ff_pre"
  )

  val ff = new TransformerFeedForward(
    numHidden = config.feedForwardNumHidden,
    numModel = config.modelSize,
    actType = config.actType,
    dropout = config.dropoutAct,
    prefix = s"${prefix}_ff"
  )

  val postFF = new TransformerProcessBlock(
    sequence = config.postProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_ff_post"
  )

  lazy val lhuc = new LHUC(config.modelSize, prefix = Some(prefix))

  def apply(data: Symbol, bias: Symbol): Symbol = {
    // self-attention
    val (selfAttData, _) = selfAttention(
      inputs = preSelfAttention(data = data, prev = None),
      bias = Some(bias),
      cache = None
    )

    val postSelfAttData = postSelfAttention(selfAttData, Some(data))

    // feed-forward
    val ffData     = ff(preFF(postSelfAttData, None))
    val postFFData = postFF(ffData, Some(data))

    config.useLHUC match {
      case true  => lhuc(postFFData)
      case false => postFFData
    }

  }
}

/**
  * A transformer encoder block consists self-attention, encoder attention, and a feed-forward layer
  * with pre/post process blocks in between.
  */
class TransformerDecoderBlock(config: TransformerConfig, prefix: String) {
  val preSelfAttention = new TransformerProcessBlock(
    sequence = config.preProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_att_self_pre"
  )

  val selfAttention = new MultiHeadSelfAttention(
    a = AttentionParams(
      depthAttention = config.modelSize,
      heads = config.attentionHeads,
      depthOut = config.modelSize,
      dropout = config.dropoutAttention,
      prefix = s"${prefix}_att_self"
    )
  )

  val postSelfAttention = new TransformerProcessBlock(
    sequence = config.postProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_att_self_post"
  )

  val preEncAttention = new TransformerProcessBlock(
    sequence = config.preProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_att_enc_pre"
  )

  val encAttention = new MultiHeadAttention(
    a = AttentionParams(
      depthAttention = config.modelSize,
      heads = config.attentionHeads,
      depthOut = config.modelSize,
      dropout = config.dropoutAttention,
      prefix = s"${prefix}_att_enc"
    )
  )

  val postEncAttention = new TransformerProcessBlock(
    sequence = config.postProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_att_enc_post"
  )

  val preFF = new TransformerProcessBlock(
    sequence = config.preProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_ff_pre"
  )

  val ff = new TransformerFeedForward(
    numHidden = config.feedForwardNumHidden,
    numModel = config.modelSize,
    actType = config.actType,
    dropout = config.dropoutAct,
    prefix = s"${prefix}_ff"
  )

  val postFF = new TransformerProcessBlock(
    sequence = config.postProcessSequence,
    dropout = config.dropoutPrePost,
    prefix = s"${prefix}_ff_post"
  )

  lazy val lhuc = new LHUC(config.modelSize, prefix = Some(prefix))

  def apply(
      target: Symbol,
      targetBias: Symbol,
      source: Symbol,
      sourceBias: Symbol,
      cache: Option[Map[String, Option[Symbol]]] = None
  ): Symbol = {

    // self attention
    val targetSelfAtt =
      selfAttention(inputs = preSelfAttention(target, None), bias = Some(targetBias), cache = cache)

    val targetPostSelfAtt = postSelfAttention(targetSelfAtt._1, Some(target))

    // encoder attention
    val targetEncAtt = encAttention(
      queries = preEncAttention(targetPostSelfAtt, None),
      memory = source,
      bias = Some(sourceBias)
    )

    val (targetPostEncAtt, updatedCache) = (targetEncAtt, target)

    // feed-forward
    val targetFF     = ff(preFF(targetPostEncAtt, None))
    val targetPostFF = postFF(targetFF, Some(targetPostEncAtt))

    config.useLHUC match {
      case true  => lhuc(targetPostEncAtt)
      case false => targetPostFF
    }

  }
}
