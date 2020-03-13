package io.mindfulmachines.sockeye.transformer

import org.apache.mxnet.{Shape, Symbol}

object Utils {

  /**
    * Workaround until no-op cast will be fixed in MXNet codebase.
    * Creates cast symbol only if dtype is different from default one, i.e. float32.
    *
    * @param data  Input symbol.
    * @param dType Target dtype.
    * @return  Cast symbol or just data symbol.
    */
  def castConditionally(data: Symbol, dType: String): Symbol = {

    dType match {
      case Constants.dTypeFloatPrecision32 => Symbol.api.cast(data = Some(data), dtype = dType)
      case _                               => data
    }
  }

  def uncastConditionally(data: Symbol, dType: String): Symbol = {

    dType match {
      case Constants.dTypeFloatPrecision32 =>
        Symbol.api.cast(data = Some(data), dtype = Constants.dTypeFloatPrecision32)
      case _ => data
    }
  }

  /**
    * Check the condition and if it is not met, exit with the given error message
    * and error_code, similar to assertions.
    * @param condition Condition to check.
    * @param errorMessage Error message to show to the user.
    * @return
    */
  def checkCondition(condition: Boolean, errorMessage: String) = {

    condition match {
      case true  =>
      case false => throw new SockeyeError(errorMessage)
    }
  }

  /**
    * Computes sequence lengths of padId-padded data in sequenceData.
    *
    * @param sequenceData  Input data. Shape: (batchSize, seqLen).
    * @return Length data. Shape: (batchSize,).
    */
  def computeLength(sequenceData: Symbol): Symbol = {
    Symbol.api.sum(
      data = Some(Symbol.notEqual(sequenceData, Constants.padId)),
      axis = Some(Shape(1))
    )
  }
}

class SockeyeError(errorMessage: String) extends Exception
