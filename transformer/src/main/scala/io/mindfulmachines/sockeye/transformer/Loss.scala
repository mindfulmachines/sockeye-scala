package io.mindfulmachines.sockeye.transformer

import org.apache.mxnet.{EvalMetric, NDArray, Shape, Symbol}

/**
  * Loss configuration.
  *
  * @param name                name: Loss name.
  * @param vocabSize           vocabSize: Target vocab size.
  * @param normalizationType   normalizationType: How to normalize the loss.
  * @param labelSmoothing      labelSmoothing: Optional smoothing constant for label smoothing.
  * @param lengthTaskLink      link: Link function.
  * @param lengthTaskWeight    weight: Loss weight.
  */
case class LossConfig(
    name: String,
    vocabSize: Option[Int] = None,
    normalizationType: Option[String] = None,
    labelSmoothing: Float = 0.0f,
    lengthTaskLink: Option[String] = None,
    lengthTaskWeight: Float = 1.0f
)

object Loss {

  /**
    * Returns a Loss instance.
    * @param config  Loss configuration.
    * @return  Instance implementing the Loss.
    */
  def getLoss(config: LossConfig): Loss = {

    config.name match {
      case x if x.equals(Constants.crossEntropy) =>
        new CrossEntropyLoss(
          config,
          outputNames = List(Constants.softmaxOutputName),
          labelNames = List(Constants.targetLabelName)
        )
      case _ => throw new Exception(s"Unknown loss name: ${config.name}")
    }
  }

  /**
    * Returns a Loss instance.
    *
    * @param config  Loss configuration.
    * @return  Instance implementing Loss.
    */
  def getLengthTaskLoss(config: LossConfig): Loss = {

    (config.lengthTaskLink) match {
      case Some(x) if x.equals(Constants.linkNormal) =>
        new MSELoss(
          config,
          outputNames = List(Constants.lenratioOutputName),
          labelNames = List(Constants.lenratioLabelName)
        )
    }
  }

}

/**
  * Generic Loss interface.
  * getLoss() method should return a loss symbol.
  * The softmax outputs (named Constants.softmaxName) are used by EvalMetrics to compute various metrics,
  *     e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
  * provides softmax outputs for forward() AND crossEntropy() gradients for backward().
  */
trait Loss {

  def lossConfig: LossConfig
  def outputNames: List[String]
  def labelNames: List[String]

  /**
    * Returns loss and softmax output symbols given logits and integer-coded labels.
    *
    * @param logits  Shape: (batchSize * targetSeqLength, targetVocabSize).
    * @param labels  Shape: (batchSize * targetSeqLength,).
    * @return  Loss symbol.
    */
  def getLoss(logits: Symbol, labels: Symbol): Symbol
  def createMetric: EvalMetric
}

/**
  * Computes the cross-entropy loss.
  */
class CrossEntropyLoss(
    val lossConfig: LossConfig,
    val outputNames: List[String],
    val labelNames: List[String],
    name: String = Constants.softmaxName,
    ignoreLabel: Int = Constants.padId
) extends Loss {

  /**
    * Returns loss symbol given logits and integer-coded labels.
    * @param logits  Shape: (batchSize * targetSeqLength, targetVocabSize).
    * @param labels  Shape: (batchSize * targetSeqLength,).
    * @return  Loss symbol.
    */
  override def getLoss(logits: Symbol, labels: Symbol): Symbol = {

    val normalization = lossConfig.normalizationType match {
      case Some(x) if x.equals(Constants.lossNormValid) => "valid"
      case Some(x) if x.equals(Constants.lossNormBatch) => "null"
      case Some(x)                                      => throw new Exception(s"Unknown loss normalization type: $x")
    }

    Symbol.api.SoftmaxOutput(
      data = Some(logits),
      label = Some(labels),
      ignore_label = Some(ignoreLabel),
      use_ignore = Some(true),
      normalization = Some(normalization),
      smooth_alpha = Some(lossConfig.labelSmoothing),
      name = name
    )

  }

  override def createMetric: EvalMetric =
    new CrossEntropyMetric(lossConfig, Some(outputNames), Some(labelNames), name)

}

/**
  * Version of the cross entropy metric that ignores padding tokens.
  * @param outputNames  Name of this metric instance for display.
  * @param labelNames  Name of predictions that should be used when updating with update_dict.
  * @param name  Name of labels that should be used when updating with update_dict.
  */
class CrossEntropyMetric(
    lossConfig: LossConfig,
    outputNames: Option[List[String]] = None,
    labelNames: Option[List[String]] = None,
    name: String = Constants.softmaxName
) extends EvalMetric(name) {

  def crossEntropy(logprob: NDArray, label: NDArray): NDArray = {
    val ce = -NDArray.api.pick(logprob, label)
    ce
  }

  def crossEntropySmoothed(
      logprob: NDArray,
      label: NDArray,
      alpha: Float,
      numClasses: Int
  ): NDArray = {
    val ce = crossEntropy(logprob, label)
    //gain for each incorrect class
    val perClassGain = alpha / (numClasses - 1)
    // discounted loss for correct class
    val discountedLoss = ce * (1 - alpha - perClassGain)
    // add gain for incorrect classes to total cross-entropy
    val totalCE = discountedLoss - NDArray.api.sum(
      logprob * perClassGain,
      axis = Some(Shape(-1)),
      keepdims = Some(false)
    )
    totalCE
  }

  //todo either properly implement update or replace with alternative

  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = ()

}

/**
  * Computes the Poisson regression loss. MSEMetric for this loss will be reporting the mean square error between
  * lengths, not length ratios!!!
  */
class PoissonLoss(
    val lossConfig: LossConfig,
    val outputNames: List[String],
    val labelNames: List[String],
    name: String = Constants.lenratioLossName
) extends Loss {

  /**
    * Returns loss symbol given logits and integer-coded labels.
    * @param pred  Predictions. shape: (batchSize, 1)
    * @param labels   Target integers. Shape: (batchSize,).
    * @return  Loss symbol.
    */
  override def getLoss(pred: Symbol, labels: Symbol): Symbol = {

    val labelsReshape = Symbol.api.reshape(Some(labels), shape = Some(Shape(-1, 1)))
    val lossValue     = pred - labels * Symbol.api.log(Some(Symbol.max(1e-10, pred)))

    // MakeLoss scales only the gradient, so scaling explicitly
    val lossValueScale = lossValue * lossConfig.lengthTaskWeight
    val lossValueBatch = Symbol.api.MakeLoss(
      data = Some(lossValueScale),
      normalization = Some("batch"),
      name = name
    )
    lossValueBatch
  }

  override def createMetric: EvalMetric = new MSEMetric(Some(outputNames), Some(labelNames), name)
}

/**
  * Computes the Mean Squared Error loss. MSEMetric for this loss will be reporting the mean square error between
  * length ratios.
  *
  * @param lossConfig  The configuration used for the corresponding loss.
  * @param outputNames  Name of this metric instance for display.
  * @param labelNames  Name of predictions that should be used when updating with updateDict.
  * @param name  Name of labels that should be used when updating with updateDict.
  */
class MSELoss(
    val lossConfig: LossConfig,
    val outputNames: List[String],
    val labelNames: List[String],
    name: String = Constants.lenratioLossName
) extends Loss {

  /**
    *
    * @param pred  Shape: (batchSize * 1).
    * @param labels  Shape: (batchSize).
    * @return  Loss symbol.
    */
  override def getLoss(pred: Symbol, labels: Symbol): Symbol = {

    val labelsReshape = Symbol.api.reshape(
      data = Some(labels),
      shape = Some(Shape(-1, 1))
    )

    val lossValue = Symbol.api.square(Some(pred - labelsReshape)) * lossConfig.lengthTaskWeight / 2

    val lossValueBatch = Symbol.api.MakeLoss(
      data = Some(lossValue),
      normalization = Some("batch"),
      name = name
    )

    lossValueBatch
  }

  override def createMetric: EvalMetric =
    new MSEMetric(Some(outputNames), Some(labelNames), name = Constants.lenrationMSE)

}

/**
  * Version of the MSE metric that ignores padding tokens.
  * @param outputNames  Name of this metric instance for display.
  * @param labelNames  Name of predictions that should be used when updating with updateDict.
  * @param name  Name of labels that should be used when updating with updateDict.
  */
class MSEMetric(
    outputNames: Option[List[String]] = None,
    labelNames: Option[List[String]] = None,
    name: String
) extends EvalMetric(name) {

  //todo either properly implement update or replace with alternative
  /**
    *
    * @param labels List of (batchSize,)-shaped NDArrays.
    * @param preds List of (batchSize,1)-shaped NDArrays.
    */
  override def update(labels: IndexedSeq[NDArray], preds: IndexedSeq[NDArray]): Unit = ()
}
