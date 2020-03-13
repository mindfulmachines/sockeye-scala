package io.mindfulmachines.sockeye.transformer

object Constants {

  val padId = 0
  // positional embeddings
  val noPositionalEmbedding      = "none"
  val fixedPositionalEmbedding   = "fixed"
  val learnedPositionalEmbedding = "learned"
  val positionalEmbeddingTypes =
    List(noPositionalEmbedding, fixedPositionalEmbedding, learnedPositionalEmbedding)

  val dTypeFloatPrecision16 = "float16"
  val dTypeFloatPrecision32 = "float32"

  val largeNegativeValue = -99999999f
  val largePositiveValue = 99999999f

  //Something at the middle of 32768<x<65519. Will be rounded to a multiple of 32.
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_integer_values
  // second element in map will be rounded to 1.0e8.
  //    // https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Precision_limits_on_integer_values.
  val largeValues =
    Map(dTypeFloatPrecision16 -> 49152.0f, dTypeFloatPrecision32 -> largePositiveValue)

  // default system values
  val defaultSystemDropout        = 0.0f
  val defaultSystemDepthAttention = 512
  val defaultSystemDepthOut       = 512
  val defaultSystemAttentionHeads = 8

  // Activation types
  // Gaussian Error Linear Unit (https://arxiv.org/pdf/1606.08415.pdf)
  val gelu = "gelu"
  // Gated Linear Unit (https://arxiv.org/pdf/1705.03122.pdf)
  val glu      = "glu"
  val relu     = "relu"
  val sigmoid  = "sigmoid"
  val softRelu = "softrelu"
  //Swish-1/SiLU (https://arxiv.org/pdf/1710.05941.pdf, https://arxiv.org/pdf/1702.03118.pdf)
  val swish1                     = "swish1"
  val tanh                       = "tanh"
  val transformerActivationTypes = List(gelu, relu, swish1)
  val CnnActivationType          = List(glu, relu, sigmoid, softRelu, tanh)

  val lhucName = "lhuc"
  // lhuc application points
  val lhucEncoder   = "encoder"
  val lhucDecoder   = "decoder"
  val lhucStateInit = "state_init"
  val lhucAll       = "all"
  val lhucChoices   = List(lhucEncoder, lhucDecoder, lhucStateInit, lhucAll)

  val sourceEncodedName = "encoded_source"
  val sourceLengthName  = "source_length"

  //data layout for strings
  val batchMajor = "NTC"

  // weight tying components
  val weightTyingSrc     = "src"
  val weightTyingTrg     = "trg"
  val weightTyingSoftmax = "softmax"
  // weight tying types (combinations of above components):
  val weightTyingTrgSoftmax    = "trg_softmax"
  val weightTyingSrcTrg        = "src_trg"
  val weightTyingSrcTrgSoftmax = "src_trg_softmax"

  // source factors
  val sourceFactorsCombineSum     = "sum"
  val sourceFactorsCombineConcat  = "concat"
  val sourceFactorsCombineChoices = List(sourceFactorsCombineSum, sourceFactorsCombineConcat)

  val transformerDecoderPrefix = "transformer_"

  // default I/O variable names
  val sourceName              = "source"
  val sourceLength            = "source_length"
  val targetName              = "target"
  val targetLabelName         = "target_label"
  val lenratioLabelName       = "length_ratio_label"
  val lenratioLabelOutputName = "length_ratio_label" + "_output"
  val lenratioName            = "length_ratio"
  val lenratioLossName        = lenratioName + "_loss"
  val lenratioOutputName      = lenratioName + "_output"
  val lexiconName             = "lexicon"

  val encoderPrefix              = "encoder_"
  val decoderPrefix              = "decoder_"
  val embeddingPrefix            = "embed_"
  val attentionPrefix            = "att_"
  val coveragePrefix             = "cov_"
  val bidirectionalRNNPrefix     = encoderPrefix + "birnn_"
  val stackedRNNPrefix           = encoderPrefix + "rnn_"
  val forwardPrefix              = "forward_"
  val reversePrefix              = "reverse_"
  val transformerEncoderPrefix   = encoderPrefix + "transformer_"
  val cnnEncoderPrefix           = encoderPrefix + "cnn_"
  val charSeqEncoderPrefix       = encoderPrefix + "char_"
  val defaultOutputLayerPrefix   = "target_output_"
  val lenratiosOutputLayerPrefix = "length_ratio_layer_"

  // embedding prefixes
  val sourceEmbeddingPrefix           = "source_" + embeddingPrefix
  val sourcePositionalEmbeddingPrefix = "source_pos_" + embeddingPrefix
  val targetEmbeddingPrefix           = "target_" + embeddingPrefix
  val targetPositionalEmbeddingPrefix = "target_pos_" + embeddingPrefix
  val sharedEmbeddingPrefix           = "source_target_" + embeddingPrefix

  val logitInputsName   = "logit_inputs"
  val logitsName        = "logits"
  val softmaxName       = "softmax"
  val softmaxOutputName = softmaxName + "_output"

  // loss
  val crossEntropy       = "cross-entropy"
  val lenratioRegression = "length-ratio-regression"

  val linkNormal       = "normal"
  val linkPoisson      = "poisson"
  val lengthTaskRatio  = "ratio"
  val lengthTaskLength = "length"

  val lossNormBatch = "batch"
  val lossNormValid = "valid"

  val targetMaxLengthFactor        = 2
  val defaultNumStdMaxOutputLength = 2

  // metrics

  val lenrationMSE = "length-ratio-mse"
}
