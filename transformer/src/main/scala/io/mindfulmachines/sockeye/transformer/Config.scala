package io.mindfulmachines.sockeye.transformer

sealed trait PrePost

/**
  * layer normalization
  */
case object N extends PrePost

/**
  * residual connection
  */
case object R extends PrePost

/**
  * dropout
  */
case object D extends PrePost

sealed trait PositionalEmbeddingType
case class SinCosEmbeddingType()  extends PositionalEmbeddingType
case class LearnedEmbeddingType() extends PositionalEmbeddingType
case class NoOpEmbeddingType()    extends PositionalEmbeddingType

sealed trait Config

case class TransformerConfig(
    modelSize: Int,
    attentionHeads: Int,
    feedForwardNumHidden: Int,
    actType: String,
    numLayers: Int,
    dropoutAttention: Float,
    dropoutAct: Float,
    dropoutPrePost: Float,
    preProcessSequence: List[PrePost],
    postProcessSequence: List[PrePost],
    maxSeqLenTarget: Int,
    maxSeqLenSource: Int,
    useLHUC: Boolean = false,
    dtype: String,
    embeddingType: PositionalEmbeddingType
) extends Config
