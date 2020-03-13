import sbt._

object Deps {
  object versions {
    val scalameta  = "4.2.0"
    val mockito    = "3.0.0"
    val scalatest  = "3.0.8"
    val mxnet      = "1.6.0-SNAPSHOT"
    val scalagraph = "1.13.0"
  }

  val scalameta = Seq(
    "org.scalameta" %% "scalameta" % versions.scalameta
  )


  val scalagraph = Seq(
    "org.scala-graph" %% "graph-core"        % versions.scalagraph,
    "org.scala-graph" %% "graph-constrained" % versions.scalagraph
  )
  val mxnet = Seq(
    "org.apache.mxnet" % "mxnet-full_2.12-osx-x86_64-cpu" % versions.mxnet % "provided"
  )

  val json4s =
    Seq("org.json4s" %% "json4s-core" % "3.6.7", "org.json4s" %% "json4s-native" % "3.6.7")

  val commonDeps = scalameta ++ json4s ++ scalagraph
}
