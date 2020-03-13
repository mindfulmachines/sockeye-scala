package io.mindfulmachines.sockeye.transformer

/**
  * Stores data statistics relevant for inference.
  */
//todo create config class dataconfig should extend it
case class DataConfig(
    dataStatistics: DataStatistics,
    maxSeqLengthSource: Int,
    maxSeqLengthTarget: Int,
    numSourceFactors: Int,
    sourceWithEOS: Boolean = false
)

case class DataStatistics(
    numSents: Int,
    numDiscarded: Int,
    numTokensSource: Int,
    numTokensTarget: Int,
    numUnksSource: Int,
    numUnksTarget: Int,
    maxObservedLenSource: Int,
    maxObservedLenTarget: Int,
    sizeVocabSource: Int,
    sizeVocabTarget: Int,
    lengthRatioMean: Float,
    lengthRatioStd: Float,
    buckets: List[(Int, Int)],
    numSentsPerBucket: List[Int],
    meanLenTargetPerBucket: List[Option[Float]],
    lengthRatioStatsPerBucket: Option[List[(Option[Float], Option[Float])]] = None
)
