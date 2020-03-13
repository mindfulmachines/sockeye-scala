package io.mindfulmachines.sockeye.transformer

import javax.swing.text.Position.Bias
import org.apache.mxnet.{DType, NDArray, Shape, Symbol}

/**
  *
  * @param numLayers Number of layers.
  * @param weight Weight of this loss.
  */
final case class LengthRatioConfig(numLayers: Int, weight: Float)

/**
  * @param queries Attention queries. Shape: (n, lq, d).
  * @param keys Attention keys. Shape: (n, lk, d).
  * @param values Attention values. Shape: (n, lk, dv).
  */
final case class QKV(queries: Symbol, keys: Symbol, values: Symbol)

/**
  * @param lengths Optional sequence lengths of the keys. Shape: (n,).
  * @param dropout Dropout probability.
  * @param bias Optional 3d bias tensor.
  * @param prefix Optional prefix.
  */
final case class DotAttentionParams(
    lengths: Option[Symbol] = None,
    dropout: Float = Constants.defaultSystemDropout,
    bias: Option[Symbol] = None,
    prefix: Option[String] = None
)

/**
  * @param prefix Attention prefix.
  * @param depthAttention Attention depth / number of hidden units.
  * @param heads Number of attention heads.
  * @param depthOut Output depth / number of output units.
  * @param dropout Dropout probability on attention scores.
  */
final case class AttentionParams(
    prefix: String,
    depthAttention: Int = Constants.defaultSystemDepthAttention,
    heads: Int = Constants.defaultSystemAttentionHeads,
    depthOut: Int = Constants.defaultSystemDepthOut,
    dropout: Float = Constants.defaultSystemDropout
)

/**
  * Defines the output layer of Sockeye decoders. Supports weight tying and weight normalization.
  *
  * @param hiddenSize  Decoder hidden size.
  * @param vocabSize Target vocabulary size.
  * @param weightNormalization Whether to apply weight normalization.
  * @param prefix  Prefix used for naming.
  */
final case class OutputParams(
    hiddenSize: Int,
    vocabSize: Int,
    weight: Option[Symbol],
    weightNormalization: Boolean,
    prefix: String = Constants.defaultOutputLayerPrefix,
    name: String = Constants.logitsName
)
object Layers {

  /**
    * Apply custom or standard activation.
    *
    * Custom activation types include:
    *      - Swish-1, also called Sigmoid-Weighted Linear Unit (SiLU): Ramachandran et
    *        al. (https://arxiv.org/pdf/1710.05941.pdf), Elfwing et al.
    * (https://arxiv.org/pdf/1702.03118.pdf)
    *      - Gaussian Error Linear Unit (GELU): Hendrycks and Gimpel
    * (https://arxiv.org/pdf/1606.08415.pdf)
    *
    * @param data Input Symbol of any shape.
    * @param actType Type of activation.
    * @return Output Symbol with same shape as input.
    */
  def activation(data: Symbol, actType: String): Symbol = {
    actType match {
      case Constants.swish1 => data * Symbol.api.Activation(Some(data), act_type = "sigmoid")
      case Constants.gelu   =>
        // val a created for readability
        val a = Symbol.api.Activation(
          Some((data + ((Symbol.pow(data, 3)) * 0.044715)) * (math.sqrt(2 / math.Pi))),
          act_type = "tanh"
        ) + 1
        // Approximation of x * gaussian_cdf(x) used by Hendrycks and Gimpel
        data * a * 0.5
      case _ => Symbol.api.Activation(Some(data), act_type = actType)
    }
  }

  /**
    * Returns a symbol with head dimension folded into batch and depth divided by the number of heads.
    *
    * @param x Symbol of shape (batch, length, depth).
    * @param depthPerHead Depth per head.
    * @param heads Number of heads.
    * @return Symbol of shape (batch * heads, length, depthPerHeads
    */
  def splitHeads(x: Symbol, depthPerHead: Int, heads: Int): Symbol = {

    // (batch, length, heads, depthPerHead)
    val xReshape =
      Symbol.api.reshape(data = Some(x), shape = Some(Shape(0, -1, heads, depthPerHead)))

    // (batch, heads, length, depth/heads)
    val xTranspose = Symbol.api.transpose(data = Some(xReshape), axes = Some(Shape(0, 2, 1, 3)))

    // (batch * heads, length, depth/heads)
    Symbol.api.reshape(data = Some(xTranspose), shape = Some(Shape(-3, -1, depthPerHead)))

  }

  /**
    * Returns a symbol with both batch & length, and head & depth dimensions combined.
    * @param x Symbol of shape (batch * heads, length, depth_per_head).
    * @param depthPerHead Depth per head.
    * @param heads Number of heads.
    * @return Symbol of shape (batch, length, depth).
    */
  def combineHeads(x: Symbol, depthPerHead: Int, heads: Int): Symbol = {

    // (batch, heads, length, depthPerHead)
    val xReshape =
      Symbol.api.reshape(data = Some(x), shape = Some(Shape(-4, -1, heads, 0, depthPerHead)))

    // (batch, length, heads, depthPerHead)
    val xTranspose = Symbol.api.transpose(data = Some(xReshape), axes = Some(Shape(0, 2, 1, 3)))

    // (batch, length, depth)
    Symbol.api.reshape(data = Some(xTranspose), shape = Some(Shape(-1, 0, depthPerHead * heads)))

  }

  /**
    * Broadcasts batch-major input of shape (batch, d1 ... dn-1) to (batch*heads, d1 ... dn-1).
    *
    * @param x Batch-major input. Shape: (batch, d1 ... dn-1).
    * @param numHeads Number of heads.
    * @param nDim Number of dimensions in x.
    * @param foldHeads Whether to fold heads dimension into batch dimension.
    * @return Tensor with each sample repeated heads-many times.
    *         Shape: (batch * heads, d1 ... dn-1) if fold_heads == True, (batch, heads, d1 ... dn-1) else.
    */
  def broadcastToHeads(x: Symbol, numHeads: Int, nDim: Int, foldHeads: Boolean = true): Symbol = {

    // Fill List with 0s nDims-1 number of times
    val dims = List.fill(nDim - 1)(0)

    // x: (batch, 1)
    val xExpandDims = Symbol.api.expand_dims(data = Some(x), axis = 1)

    // x: (batch, heads, dims....)
    val xBroadcast = Symbol.api
      .broadcast_to(data = Some(xExpandDims), shape = Some(Shape(List(0, numHeads) ++ dims)))

    // x: (batch, heads, dims....)
    foldHeads match {
      case true =>
        Symbol.api.reshape(data = Some(xBroadcast), shape = Some(Shape(List(-3) ++ dims)))
      case false => xBroadcast
    }

  }

  /**
    * @param qkv References parameter object `QKV` for queries, keys, values in Attention architecture
    * @param d References parameter object `DotAttentionParams`.
    * @return 'Context' vectors for each query. Shape: (n, lq, dv).
    */
  def dotAttention(
      qkv: QKV,
      d: DotAttentionParams
  ): Symbol = {
// todo why this need to  be commented for transformers to successfully run?
/*    assert(
      d.lengths.nonEmpty || d.bias.nonEmpty,
      "Must provide either length or bias argument for masking"
    )*/

    // (number of heads, length of queries, length of keys)
    val logits = Symbol.api.batch_dot(
      lhs = Some(qkv.queries),
      rhs = Some(qkv.keys),
      transpose_b = Some(true),
      name = s"${d.prefix}_dot"
    )

    val lengthForMasking = d.lengths match {
      case Some(_) =>
        val logitsTranspose = Symbol.api.transpose(data = Some(logits), axes = Some(Shape(2, 0, 1)))
        val logitsSequenceMask = Symbol.api.SequenceMask(
          data = Some(logitsTranspose),
          use_sequence_length = Some(true),
          sequence_length = d.lengths,
          value = Some(Constants.largeNegativeValue)
        )
        Symbol.api.transpose(data = Some(logitsSequenceMask), axes = Some(Shape(1, 2, 0)))
      case None => logits
    }

    val biasForMasking = d.bias match {
      case Some(_) =>
        Symbol.api.broadcast_add(Some(lengthForMasking), d.bias, name = s"${d.prefix}_bias_add")
      case None => lengthForMasking
    }

    val probs = Symbol.api.softmax(data = Some(biasForMasking), axis = Some(-1))
    val probsDropout =
      if (d.dropout > 0.0) {
        Symbol.api.Dropout(data = Some(probs), p = Some(d.dropout))
      } else probs

    // (n, lq, lk) x (n, lk, dv) -> (n, lq, dv)
    Symbol.api.batch_dot(
      lhs = Some(probsDropout),
      rhs = Some(qkv.values),
      name = s"${d.prefix}_context"
    )

  }

}

/**
  * Implements Ba et al, Layer Normalization (https://arxiv.org/abs/1607.06450).
  * @param prefix Prefix of layer name.
  * @param scale Optional variable for scaling of shape (numHidden). Will be created if None.
  * @param shift Optional variable for shifting of shape (numHidden). Will be created if None.
  * @param scaleInit Initial value of scale variable if scale is None. Default 1.0.
  * @param shiftinit Initial value of shift variable if shift is None. Default 0.0.
  */
class LayerNormalization(
    prefix: String = "layerNorm",
    scale: Option[Symbol] = None,
    shift: Option[Symbol] = None,
    scaleInit: Float = 1.0f,
    shiftinit: Float = 0.0f
) {

  val defScale = scale match {
    case Some(s) => s
    // todo how the F do I set initialization value to 1.0f?
    case None => Symbol.Variable(s"${prefix}_gamma")
  }

  val defShift = shift match {
    case Some(s) => s
    // todo how the F do I set initialization value to 0.0f?
    case None => Symbol.Variable(s"${prefix}_beta")

  }

  /**
    * Normalizes hidden units of data as follows:
    *
    * data = scale * (data - mean) / sqrt(var + eps) + shift
    *
    * Normalization is performed over the last dimension of the input data.
    * @param data  Data to normalize. Shape: (d0, ..., dn, numHidden).
    * @param eps  Variance epsilon.
    * @return  Normalized inputs. Shape: (d0, ..., dn, num_hidden).
    */
  def apply(data: Symbol, eps: Float = 1e-06f): Symbol = {
    Symbol.api.LayerNorm(
      data = Some(data),
      gamma = scale,
      beta = shift,
      axis = Some(-1),
      eps = Some(eps),
      output_mean_var = Some(false),
      name = prefix
    )
  }

}

/**
  * Learning Hidden Unit Contribution
  *
  * David Vilar. "Learning Hidden Unit Contribution for Adapting Neural
  * Machine Translation Models" NAACL 2018
  *
  * @param numHidden Number of hidden units of the layer to be modified.
  * @param weight Optional parameter vector.
  * @param prefix Optional prefix for created parameters (if not given as weight).
  */
class LHUC(numHidden: Int, weight: Option[Symbol] = None, prefix: Option[String] = None) {

  //todo how the F do I initialize this to mx.init.Uniform(0.1)???
  val params = weight match {
    case None =>
      Symbol.Variable(
        name = s"${prefix}_${Constants.lhucName}",
        shape = Shape(Some(numHidden)),
        dType = DType.Float32
      )
    case Some(w) => w
  }

  /**
    * We use a sigmoid with amplitude 2 for weighting the hidden units. The
    * activation is dampened when the value of the sigmoid is close to 0, and
    * strengthened when it's close to 2 (see also original paper)
    */
  def apply(inputs: Symbol, name: Option[String] = None): Symbol = {

    val weightVector = Symbol.api.Activation(
      data = Some(params),
      act_type = "sigmoid"
    ) * 2

    val out = Symbol.api.broadcast_mul(
      lhs = Some(weightVector),
      rhs = Some(inputs),
      name = name.getOrElse(throw new RuntimeException("apply method requires name parameter"))
    )
    out

  }

}

/**
  * Base class for Multi-Head attention.
  */
class MultiHeadAttentionBase(a: AttentionParams) {

  val depth        = a.depthAttention
  val depthPerHead = (depth / a.heads).intValue()

  val w_h2o = Symbol.Variable(s"${a.prefix}_hidden_to_output_weight")

  /**
    * Returns context vectors of multi-head dot attention.
    *
    * @param qkv     References `QKV` parameter object composed of:
    *                queries: Query tensor. Shape: (batchSize, queryMaxLength, depth).
    *                keys: Keys. Shape: (batchSize, memoryMaxLength, depth).
    *                values: Values. Shape: (batchSize, memoryMaxLength, depth).
    * @param lengths Optional lengths of keys. Shape: (batchSize).
    * @param bias    Optional 3d bias.
    * @return Context vectors. Shape: (batch_size, query_max_length, output_depth).
    */
  def getAttentionContext(
      qkv: QKV,
      lengths: Option[Symbol] = None,
      bias: Option[Symbol] = None
  ): Symbol = {

    // scale by sqrt(depthPerHead)
    val queries = qkv.queries * math.pow(depthPerHead, -0.5)

    //(batch*heads, length, depth/heads)
    val shQueries = Layers.splitHeads(queries, depthPerHead, a.heads)
    val shKeys    = Layers.splitHeads(qkv.keys, depthPerHead, a.heads)
    val shValues  = Layers.splitHeads(qkv.values, depthPerHead, a.heads)

    val length = lengths match {
      case Some(l) =>
        Some(
          Layers
            .broadcastToHeads(
              l,
              a.heads,
              nDim = 1,
              foldHeads = true
            )
        )
      case None => None
    }

    // (batch*heads, queryMaxLength, depthPerhead)
    val dotAttentionContexts = Layers
      .dotAttention(
        QKV(shQueries, shKeys, shValues),
        DotAttentionParams(
          lengths = length,
          dropout = a.dropout,
          bias = bias,
          prefix = Some(a.prefix)
        )
      )

    //(batch, queryMaxDepth, depth)
    val combineHeadsContext =
      Layers.combineHeads(dotAttentionContexts, depthPerHead, a.heads)

    Symbol.api.FullyConnected(
      data = Some(combineHeadsContext),
      weight = Some(w_h2o),
      no_bias = Some(true),
      num_hidden = a.depthAttention,
      flatten = Some(false)
    )

  }
}

/**
  * Multi-head self-attention. Independent linear projections of inputs serve as
  * queries, keys, and values for the attention.
  */
class MultiHeadSelfAttention(a: AttentionParams) extends MultiHeadAttentionBase(a) {

  val w_i2h = Symbol.Variable(s"${a.prefix}_i2h_weight")

  /**
    * Computes multi-head attention on a set of inputs, serving as queries, keys, and values.
    * If sequence lengths are provided, they will be used to mask the attention scores.
    * A bias mask may also be used to mask the attention scores.
    * May also use a cache of previously computed inputs.
    * Returns a symbol of shape (batch, maxLength, outputDepth).
    *
    * @param inputs Input Data. Shape: (batch, max_length, input_depth).
    * @param inputLengths Optional lengths of inputs to mask attention scores. Shape: (batch, 1).
    * @param bias Optional 3d bias tensor to mask attention scores.
    * @param cache Optional dictionary of previously computed keys and values.
    * @return Symbol of shape (batch, maxLength, outputDepth) and updated cache.
    */
  def apply(
      inputs: Symbol,
      inputLengths: Option[Symbol] = None,
      bias: Option[Symbol] = None,
      cache: Option[Map[String, Option[Symbol]]] = None
  ): (Symbol, Option[Map[String, Option[Symbol]]]) = {

    //combined: (batch, maxLength, depth * 3)
    val combined = Symbol.api.FullyConnected(
      data = Some(inputs),
      weight = Some(w_i2h),
      no_bias = Some(true),
      num_hidden = depth * 3,
      flatten = Some(false),
      name = s"${a.prefix}_qkv_transform"
    )

    //split into query, keys and values
    //(batch, maxLength, depth)

    val qkvList = Symbol.api.split(data = Some(combined), num_outputs = 3, axis = Some(2))

    val qkvObject = QKV(qkvList.get(0), qkvList.get(1), qkvList.get(2))

    //append new keys & values to cache, update the cache
    val (keys, values, updatedCache) = cache match {

      case Some(c) =>
        val valueOfkey   = c.get("k").flatten
        val valueOfValue = c.get("v").flatten
        val theKey = valueOfkey match {
          case Some(k) =>
            val data = Array(k, qkvObject.keys)
            Symbol.api.concat(data, num_args = data.length, dim = Some(1))
          case None => qkvObject.keys

        }
        val theValue = valueOfValue match {
          case Some(v) =>
            val data = Array(v, qkvObject.values)
            Symbol.api.concat(data, num_args = data.length, dim = Some(1))
          case None => qkvObject.values
        }
        val theCache = c + ("k" -> Some(theKey), "v" -> Some(theValue))
        (theKey, theValue, Some(theCache))

      case None => (qkvObject.keys, qkvObject.values, cache)
    }

    (
      super.getAttentionContext(QKV(qkvObject.queries, keys, values), inputLengths, bias),
      updatedCache
    )

  }

}

/**
  * Multi-head attention layer for queries independent from keys/values.
  *
  * @param a References `AttentionParams` parameter object composed of:
  *           -- prefix: Attention prefix.
  *           -- depth_att: Attention depth / number of hidden units.
  *           -- heads: Number of attention heads.
  *           -- depth_out: Output depth / number of output units.
  *           -- dropout: Dropout probability on attention scores
  */
class MultiHeadAttention(a: AttentionParams) extends MultiHeadAttentionBase(a) {

  val w_q2h = Symbol.Variable(s"${a.prefix}_q2h_weight")
  val w_k2h = Symbol.Variable(s"${a.prefix}_k2h_weight")
  val w_v2h = Symbol.Variable(s"${a.prefix}_v2h_weight")

  /**
    * Computes multi-head attention for queries given a memory tensor.
    * If sequence lengths are provided, they will be used to mask the attention scores.
    * A bias mask may also be used to mask the attention scores.
    * Returns a symbol of shape (batch, max_length, output_depth).
    *
    * @param queries  Query tensor. Shape: (batch, queryMaxLength, inputDepth).
    * @param memory   Memory data to attend to. Shape: (batch, memoryMaxLength, inputDepth).
    * @param memoryLengths Optional lengths of memory to mask attention scores. Shape: (batch, 1).
    * @param bias  Optional 3d bias tensor to mask attention scores.
    * @return Symbol of shape (batch, querySequenceLength, outputDepth).
    */
  def apply(
      queries: Symbol,
      memory: Symbol,
      memoryLengths: Option[Symbol] = None,
      bias: Option[Symbol] = None
  ): Symbol = {

    // (batch, queryMaxLength, depth)
    val fcQueries = Symbol.api
      .FullyConnected(
        data = Some(queries),
        weight = Some(w_q2h),
        no_bias = Some(true),
        num_hidden = depth,
        flatten = Some(false),
        name = s"${a.prefix}_q_transform"
      )

    // (batch, memoryMaxLength, depth)
    val fcKeys = Symbol.api
      .FullyConnected(
        data = Some(memory),
        weight = Some(w_k2h),
        no_bias = Some(true),
        num_hidden = depth,
        flatten = Some(false),
        name = s"${a.prefix}_k_transform"
      )

    // (batch, memoryMaxLength, depth)
    val fcValues = Symbol.api
      .FullyConnected(
        data = Some(memory),
        weight = Some(w_v2h),
        no_bias = Some(true),
        num_hidden = depth,
        flatten = Some(false),
        name = s"${a.prefix}_v_transform"
      )

    super.getAttentionContext(QKV(fcQueries, fcKeys, fcValues), memoryLengths, bias = bias)
  }
}

/**
  * Dot attention layer for queries independent from keys/values.
  * @param prefix Attention prefix.
  * @param numHidden Attention depth / number of hidden units.
  */
class ProjectedDotAttention(prefix: String, numHidden: Int) {

  val w_q2h  = Symbol.Variable(s"${prefix}_q2h_weight")
  val b_q2h  = Symbol.Variable(s"${prefix}_q2h_bias")
  val w_kv2h = Symbol.Variable(s"${prefix}_kv2h_weight")
  val b_kv2h = Symbol.Variable(s"${prefix}_kv2h_bias")

  /**
    *  Apply project, apply dot attention and return new context vectors.
    *
    * @param queries        Symbol of shape (batch, queriesMaxLength, inputNumHidden).
    * @param memory         Symbol of shape (batch, memoryMaxLength, inputNumHidden).
    * @param memoryLengths  Symbol of shape (batch, 1).
    * @return               Symbol of shape (batch, queriesMaxLength, numHidden).
    */
  def apply(queries: Symbol, memory: Symbol, memoryLengths: Symbol): Symbol = {

    // (batch, memory_max_length, num_hidden * 2)
    val combinedKV = Symbol.api.FullyConnected(
      data = Some(memory),
      weight = Some(w_kv2h),
      bias = Some(b_kv2h),
      num_hidden = numHidden * 2,
      flatten = Some(false),
      name = s"${prefix}_kv_transform"
    )

    // split into keys and values

    val splitKV = Symbol.api
      .split(
        data = Some(combinedKV),
        num_outputs = 2,
        axis = Some(2)
      )

    val (keys, values) = (splitKV.get(0), splitKV.get(1))

    // (batch, queriesMaxLength, numHidden)
    val fcQueries = Symbol.api.FullyConnected(
      data = Some(queries),
      weight = Some(w_q2h),
      bias = Some(b_q2h),
      num_hidden = numHidden,
      flatten = Some(false),
      name = s"${prefix}_q_transform"
    )

    // scale by sqrt(numHidden)
    val scaledQueries = fcQueries * math.pow(numHidden, -0.5)

    // (batch, queriesMaxLength, numHidden)
    val contexts = Layers.dotAttention(
      QKV(queries = scaledQueries, keys = keys, values = values),
      d = DotAttentionParams(Some(memoryLengths))
    )

    contexts

  }
}

/**
  * Dot attention layer for queries independent from keys/values.
  */
class PlainDotAttention {

  /**
    * Returns a Symbol of shape (batch, maxLength, outputDepth).
    * @param queries Symbol of shape (batch, queriesMaxLength, inputDepth).
    * @param memory  Symbol of shape (batch, memoryMaxLength, inputDepth).
    * @param memoryLengths Symbol of shape (batch, 1).
    * @return Symbol of shape (batch, queriesMaxLength, outputDepth).
    */
  def apply(queries: Symbol, memory: Symbol, memoryLengths: Symbol): Symbol = {

    val contexts = Layers.dotAttention(
      qkv = QKV(queries = queries, keys = memory, values = memory),
      d = DotAttentionParams(Some(memoryLengths))
    )

    contexts
  }
}

/**
  * Implements Weight Normalization, see Salimans & Kingma 2016 (https://arxiv.org/abs/1602.07868).
  * For a given tensor the normalization is done per hidden dimension.
  *
  * @param weightSymbol  Weight tensor of shape: (numHidden, d1, d2, ...).
  * @param numHidden Size of the first dimension.
  * @param nDim  The total number of dimensions of the weight tensor.
  * @param prefix  The prefix used for naming.
  */
class WeightNormalization(
    weightSymbol: Symbol,
    numHidden: Int,
    nDim: Int = 2,
    prefix: Option[String] = None
) {

  //TODO how the F do i set the initialization value to init=mx.init.Constant(value=1.0)
  val scaleSymbol = Symbol.Variable(
    name = s"${prefix}wn_scale",
    shape = Shape(List(numHidden) ::: List.fill(nDim - 1)(1))
  )

  /**
    * Normalize each hidden dimension and scale afterwards
    * @return weight normalized weight tensor.
    */
  def apply(
      weightNDArray: Option[NDArray] = None,
      scaleNDArray: Option[NDArray] = None
  ): Either[Symbol, NDArray] = {

    val wFinal: Either[Symbol, NDArray] = (weightNDArray, scaleNDArray) match {
      case (None, None) =>
        Left(
          Symbol.api.broadcast_mul(
            lhs = Some(Symbol.api.L2Normalization(Some(weightSymbol), mode = Some("instance"))),
            rhs = Some(scaleSymbol),
            name = s"${prefix}wn_scale"
          )
        )
      case (Some(wnd), Some(snd)) =>
        Right(
          NDArray.api.broadcast_mul(
            lhs = NDArray.api.L2Normalization(wnd, mode = Some("instance")),
            rhs = snd
          )
        )
    }
    wFinal
  }
}

/**
  * Defines the output layer of Sockeye decoders. Supports weight tying and weight normalization.
  *
  * @param p parameter object containing params for OutputLayer, contains:
  *           -- hiddenSize  Decoder hidden size.
  *           -- vocabSize Target vocabulary size.
  *           -- weightNormalization Whether to apply weight normalization.
  *           -- prefix  Prefix used for naming.
  */
class OutputLayer(p: OutputParams) {

  val w = p.weight match {
    case Some(w) => w
    case None =>
      Symbol.Variable(
        name = s"${p.prefix}weight",
        shape = Shape(p.vocabSize, p.hiddenSize),
        dType = DType.Float32
      )
  }

  val weightNormalized = p.weightNormalization match {
    case true =>
      new WeightNormalization(w, numHidden = p.vocabSize, nDim = 2, prefix = Some(p.prefix))
        .apply() match {
        case Left(wn) => wn
        case Right(_) => throw new Exception("Impossible")
      }
    case false => w
  }

  val b = Symbol.Variable(s"${p.prefix}bias")

  /**
    * Linear transformation to vocab size. Returns logits.
    *
    * @param hidden  Decoder representation for n elements. Shape: (n, numHidden).
    * @return  Logits. Shape(n, vocabSize)
    */
  def apply(
      hidden: Either[Symbol, NDArray],
      weight: Option[NDArray] = None,
      bias: Option[NDArray] = None
  ): Either[Symbol, NDArray] = {
    (hidden, weight, bias) match {
      case (Left(x), _, _) =>
        Left(
          Symbol.api
            .FullyConnected(
              data = Some(x),
              num_hidden = p.vocabSize,
              weight = Some(weightNormalized),
              bias = Some(b),
              flatten = Some(false),
              name = p.name
            )
        )
      case (Right(y), Some(wnd), Some(bnd)) =>
        Right(
          NDArray.api
            .FullyConnected(
              data = y,
              num_hidden = bnd.shape.get(0),
              weight = wnd,
              bias = y,
              flatten = Some(false)
            )
            .get
        )

    }
  }
}

/**
  * Defines the length-ratio prediction layer of Sockeye.
  *
  * @param hiddenSize  Encoder hidden size.
  * @param numLayers Number of layers.
  * @param prefix  Prefix used for naming.
  */
class LengthRatio(
    hiddenSize: Int,
    numLayers: Int,
    prefix: String = Constants.lenratiosOutputLayerPrefix
) {
  Utils.checkCondition(numLayers >= 1, "LengthRatio's num_layers has to be >=1.")

  /**
    * Transformation to the length ratio. Returns a vector.
    *
    * @param sourceEncoded Encoder representation for n elements. Shape: (n, sourceEncodedLength, hiddenSize).
    * @param sourceEncodedLength A vector of encoded sequence lengths. Shape: (n,).
    * @return  Predictions of the ratio length(hypothesis)/length(reference). Shape(n, 1).
    */
  def apply(sourceEncoded: Symbol, sourceEncodedLength: Symbol): Symbol = {

    // data: (n, hiddenSize)
    val data = averageSources(sourceEncoded, sourceEncodedLength)

    // MLP
    val layeredTanh = (0 until numLayers).foldLeft(data) {
      case (d, l) =>
        val fcTanh = Symbol.api.FullyConnected(
          data = Some(d),
          num_hidden = hiddenSize
        )
        val layersTanh = Symbol.api.Activation(
          data = Some(fcTanh),
          act_type = "tanh",
          name = prefix + s"dense${l}_"
        )

        layersTanh

    }

    // SoftReLU activation to ensure positiveness of the predicted length ratio
    val fcSoftReLu = Symbol.api.FullyConnected(
      data = Some(layeredTanh),
      num_hidden = 1
    )
    // layeredDataSoftReLu: (n,1)
    val layeredDataSoftReLu = Symbol.api.Activation(
      data = Some(fcSoftReLu),
      act_type = "softrelu",
      name = prefix + s"dense${numLayers - 1}_"
    )

    layeredDataSoftReLu

  }

  /**
    * Calculate the average of encoded sources taking into account their lengths.
    * @param sourceEncoded Encoder representation for n elements. Shape: (n, sourceEncodedLength, hiddenSize).
    * @param sourceEncodedLength  A vector of encoded sequence lengths. Shape: (n,).
    * @return  Average vectors. Shape(n, hidden_size).
    */
  def averageSources(sourceEncoded: Symbol, sourceEncodedLength: Symbol): Symbol = {

    //sourceMasked: (n, sourceEncodedLength, hiddenSize)
    val sourceMasked = Symbol.api.SequenceMask(
      data = Some(sourceEncoded),
      axis = Some(1),
      sequence_length = Some(sourceEncodedLength),
      use_sequence_length = Some(true),
      value = Some(0.0f)
    )

    val averaged = Symbol.api.broadcast_div(
      lhs = Some(
        Symbol.api.sum(data = Some(sourceMasked), axis = Some(Shape(1)), keepdims = Some(false))
      ),
      rhs = Some(Symbol.api.reshape(Some(sourceEncodedLength), shape = Some(Shape(-1, 1))))
    )

    averaged
  }
}
