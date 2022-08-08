name := "recommenderV2"

version := "0.1"

scalaVersion := "2.13.8"

import Dependencies._

libraryDependencies ++= Seq(
  SparkCore,
  SparkMLlib
)
