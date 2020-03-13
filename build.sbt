import sbt.Keys.libraryDependencies
import Deps._

name := "sockeye-scala"

lazy val commonSettings = Seq(
  scalaVersion        := "2.12.8",
  organization        := "io.mindfulmachines",
  version             := "0.0.1",
  libraryDependencies ++= commonDeps,
  resolvers += "Local Maven Repository" at "file://" + Path.userHome.absolutePath + "/.m2/repository",
  publishTo := Some(Resolver.file("file", new File(Path.userHome.absolutePath + "/releases/sockeye-scala")))
)

lazy val `sockeye-scala` = (project in file("."))
  .aggregate(transformer)
  .settings(commonSettings)


lazy val transformer: Project = (project in file("transformer"))
  .settings(commonSettings, libraryDependencies ++= mxnet)


lazy val examples = (project in file("examples"))
  .settings(commonSettings, libraryDependencies ++= mxnet)
  .dependsOn(transformer)
