/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.filters

import com.salesforce.op.OpParams
import com.salesforce.op.features.types.Text
import com.salesforce.op.features.{OPFeature, TransientFeature}
import com.salesforce.op.readers.DataFrameFieldNames
import com.salesforce.op.stages.impl.feature.{HashAlgorithm, TextTokenizer}
import com.salesforce.op.test.{Passenger, PassengerSparkFixtureTest}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.io.Source

@RunWith(classOf[JUnitRunner])
class RawFeatureFilterTest extends FlatSpec with PassengerSparkFixtureTest with FiltersTestData {
  val txt1 = Seq("a", "b", "c", "c", "c")
  val txt2 = Seq("a", "d", "b", "a", "b")
  def calculateJaccard(seq1: Seq[String], seq2: Seq[String]): Double = {
    val bagOfWords1 = seq1.groupBy(identity).mapValues(_.size)
    val bagOfWords2 = seq2.groupBy(identity).mapValues(_.size)
    val intersection = bagOfWords1.map { case (str, ct) =>
      if (bagOfWords2.get(str).isEmpty) 0 else Math.min(bagOfWords2(str), ct)
    }.sum * 2
    val union = bagOfWords1.values.sum + bagOfWords2.values.sum
    1 - intersection.toDouble / union
  }
  it should "not be very small for random text" in {
    val text1 = RandomText.strings(1,10).take(100000).map(_.value.get).toSeq
    val text2 = RandomText.strings(1,10).take(100000).map(_.value.get).toSeq
    val hasher: HashingTF = new HashingTF(numFeatures = 10000)
      .setBinary(false)
      .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)
    val hash1 = hasher.transform(text1).toArray
    val hash2 = hasher.transform(text2).toArray
    val dist1 = FeatureDistribution("text", None, 100000L, 0, hash1, Array())
    val dist2 = FeatureDistribution("text", None, 100000L, 0, hash2, Array())
    val js = dist1.jsDivergence(dist2)
    println(js)
    println(calculateJaccard(text1, text2))
  }

  it should "not be very large for selected text" in {
    val text1 = RandomText.pickLists(domain = List("alpha", "beta", "gamma", "delta"), List(0.05, 0.1, 0.15, 1.0))
      .take(100000).map(_.value.get).toSeq
    val text2 = RandomText.pickLists(domain = List("alpha", "beta", "gamma", "delta"), List(0.1, 0.3, 0.7, 1.0))
      .take(100000).map(_.value.get).toSeq
    val hasher: HashingTF = new HashingTF(numFeatures = 100)
      .setBinary(false)
      .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)
    val hash1 = hasher.transform(text1).toArray
    val hash2 = hasher.transform(text2).toArray
    val dist1 = FeatureDistribution("text", None, 1000L, 0, hash1, Array())
    val dist2 = FeatureDistribution("text", None, 1000L, 0, hash2, Array())
    val js = dist1.jsDivergence(dist2)
    println(js)
    println(calculateJaccard(text1, text2))
  }

  it should "be large for different text" in {
    val text1 = Source.fromFile("src/test/resources/text1.txt").getLines.toList.map(Text(_))
      .flatMap(TextTokenizer.tokenize(_).tokens.value)
    println(text1.size)
    val text2 = Source.fromFile("src/test/resources/text2.txt").getLines.toList.map(Text(_))
      .flatMap(TextTokenizer.tokenize(_).tokens.value)
    println(text2.size)
    val hasher: HashingTF = new HashingTF(numFeatures = 100)
      .setBinary(false)
      .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)
    val hash1 = hasher.transform(text1).toArray
    val hash2 = hasher.transform(text2).toArray
    val dist1 = FeatureDistribution("text", None, 1000L, 0, hash1, Array())
    val dist2 = FeatureDistribution("text", None, 1000L, 0, hash2, Array())
    val js = dist1.jsDivergence(dist2)
    println(js)
    println(calculateJaccard(text1, text2))
  }

  Spec[RawFeatureFilter[_]] should "compute feature stats correctly" in {
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.8, Double.PositiveInfinity, 0.7, 1.0)
    val allFeatureInfo = filter.computeFeatureStats(passengersDataSet, features)

    allFeatureInfo.responseSummaries.size shouldBe 1
    allFeatureInfo.responseSummaries.headOption.map(_._2) shouldEqual Option(Summary(0, 1))
    allFeatureInfo.responseDistributions.size shouldBe 1
    allFeatureInfo.predictorSummaries.size shouldBe 12
    allFeatureInfo.predictorDistributions.size shouldBe 12

    val surv = allFeatureInfo.responseDistributions(0)
    surv.name shouldBe survived.name
    surv.key shouldBe None
    surv.count shouldBe 6
    surv.nulls shouldBe 4
    surv.distribution.sum shouldBe 2

    val ageF = allFeatureInfo.predictorDistributions.filter(_.name == age.name)(0)
    ageF.name shouldBe age.name
    ageF.key shouldBe None
    ageF.count shouldBe 6
    ageF.nulls shouldBe 2
    ageF.distribution.sum shouldBe 4

    val strMapF =
      allFeatureInfo.predictorDistributions.filter(d => d.name == stringMap.name && d.key == Option("Female"))(0)

    strMapF.name shouldBe stringMap.name
    if (strMapF.key.contains("Female")) strMapF.nulls shouldBe 3 else strMapF.nulls shouldBe 4

    val strMapM =
      allFeatureInfo.predictorDistributions.filter(d => d.name == stringMap.name && d.key == Option("Male"))(0)

    strMapM.name shouldBe stringMap.name
    if (strMapM.key.contains("Male")) strMapM.nulls shouldBe 4 else strMapM.nulls shouldBe 3
  }

  it should "correctly determine which features to exclude based on the stats of training fill rate" in {
    // only fill rate matters
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedTrainF, excludedTrainMK) =
      filter.getFeaturesToExclude(trainSummaries, Seq.empty, Map.empty)
    excludedTrainF.toSet shouldEqual Set("B", "D")
    excludedTrainMK.keySet shouldEqual Set("C")
    excludedTrainMK.head._2 shouldEqual Set("2")
  }

  it should "correctly determine which features to exclude based on the stats of training and scoring fill rate" in {
    // only fill rate matters

    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedBothF, excludedBothMK) =
      filter.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothF.toSet shouldEqual Set("B", "D")
    excludedBothMK.keySet shouldEqual Set("C")
    excludedBothMK.head._2 shouldEqual Set("2")
  }

  it should "correctly determine which features to exclude based on the stats of relative fill rate" in {
    // relative fill rate matters
    val filter2 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 0.5, Double.PositiveInfinity, 1.0, 1.0)
    val (excludedBothRelF, excludedBothRelMK) =
      filter2.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothRelF.toSet shouldEqual Set("A")
    excludedBothRelMK.isEmpty shouldBe true
  }

  it should "correctly determine which features to exclude based on the stats of fill rate ratio" in {
    // relative fill ratio matters
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, 2.0, 1.0, 1.0)
    val (excludedBothRelFR, excludedBothRelMKR) =
      filter4.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothRelFR.toSet shouldEqual Set("D", "A", "B")
    excludedBothRelMKR.isEmpty shouldBe true
  }

  it should "correctly determine which features to exclude based on the stats of js distance" in {
    // js distance
    val filter3 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, Double.PositiveInfinity, 0.5, 1.0)
    val (excludedBothDistF, excludedBothDistMK) =
      filter3.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothDistF.isEmpty shouldEqual true
    excludedBothDistMK.keySet shouldEqual Set("C")
    excludedBothDistMK.head._2 shouldEqual Set("1")
  }

  it should "correctly determine which features to exclude based on all the stats" in {
    // all
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.5, Double.PositiveInfinity, 0.5, 1.0)
    val (excludedBothAllF, excludedBothAllMK) =
      filter4.getFeaturesToExclude(trainSummaries, scoreSummaries, Map.empty)
    excludedBothAllF.toSet shouldEqual Set("A", "B", "C", "D")
    excludedBothAllMK.isEmpty shouldBe true
  }

  it should "correctly clean the dataframe returned and give the features to blacklist" in {
    val params = new OpParams()
    val survPred = survived.copy(isResponse = false)
    val features: Array[OPFeature] =
      Array(survPred, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.0, 1.0, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.isEmpty shouldBe true
    filteredRawData.mapKeysToDrop.isEmpty shouldBe true
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields

    val filter1 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData1 = filter1.generateFilteredRaw(features, params)
    filteredRawData1.featuresToDrop should contain theSameElementsAs Array(survPred)
    filteredRawData1.mapKeysToDrop should contain theSameElementsAs Map("numericMap" -> Set("Male"),
      "booleanMap" -> Set("Male"), "stringMap" -> Set("Male"))
    filteredRawData1.cleanedData.schema.fields.exists(_.name == survPred.name) shouldBe false
    filteredRawData1.cleanedData.collect(stringMap).
      foreach(m => if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
  }

  it should "not drop response features" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0, 1.0)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.isEmpty shouldBe true
    filteredRawData.cleanedData.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields
    filteredRawData.cleanedData.collect(stringMap)
      .foreach(m => if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
  }


  it should "not drop protected features" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.1, 0.1, 2, 0.2, 0.9)
    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.toSet shouldEqual Set(age, gender, height, weight, description, boarded)
    filteredRawData.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Array(DataFrameFieldNames.KeyFieldName, survived.name)

    val filter2 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.1, 0.1, 2, 0.2, 0.9,
      protectedFeatures = Set(age.name, gender.name))
    val filteredRawData2 = filter2.generateFilteredRaw(features, params)
    filteredRawData2.featuresToDrop.toSet shouldEqual Set(height, weight, description, boarded)
    filteredRawData2.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Array(DataFrameFieldNames.KeyFieldName, survived.name, age.name, gender.name)
  }

  it should "not drop JS divergence-protected features based on JS divergence check" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, boardedTime, boardedTimeAsDateTime)
    val filter = new RawFeatureFilter(
      trainingReader = dataReader,
      scoreReader = Some(simpleReader),
      bins = 10,
      minFill = 0.0,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 0.0,
      maxCorrelation = 1.0,
      jsDivergenceProtectedFeatures = Set(boardedTime.name, boardedTimeAsDateTime.name))

    val filteredRawData = filter.generateFilteredRaw(features, params)
    filteredRawData.featuresToDrop.toSet shouldEqual Set(age, gender, height, weight, description, boarded)
    filteredRawData.cleanedData.schema.fields.map(_.name) should contain theSameElementsAs
      Seq(DataFrameFieldNames.KeyFieldName, survived.name, boardedTime.name, boardedTimeAsDateTime.name)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.9" in {
    val expectedDropped = Seq(boarded, weight, gender)
    val expectedMapKeys = Seq("Female", "Male")
    val expectedDroppedMapKeys = Map[String, Set[String]]()
    nullLabelCorrelationTest(0.9, expectedDropped, expectedMapKeys, expectedDroppedMapKeys)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.6" in {
    val expectedDropped = Seq(boarded, weight, gender, age)
    val expectedMapKeys = Seq("Female", "Male")
    val expectedDroppedMapKeys = Map[String, Set[String]]()
    nullLabelCorrelationTest(0.6, expectedDropped, expectedMapKeys, expectedDroppedMapKeys)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.4" in {
    val expectedDropped = Seq(boarded, weight, gender, age, description)
    val expectedMapKeys = Seq("Male")
    val expectedDroppedMapKeys = Map("booleanMap" -> Set("Female"), "stringMap" -> Set("Female"),
      "numericMap" -> Set("Female"))
    nullLabelCorrelationTest(0.4, expectedDropped, expectedMapKeys, expectedDroppedMapKeys)
  }

  it should "correctly drop features based on null-label leakage correlation greater than 0.3" in {
    val expectedDropped = Seq(boarded, weight, gender, age, description, booleanMap, numericMap, stringMap)
    // all the maps dropped
    val expectedDroppedMapKeys = Map[String, Set[String]]()
    nullLabelCorrelationTest(0.3, expectedDropped, Seq(), expectedDroppedMapKeys)
  }

  private def nullLabelCorrelationTest(
    maxCorrelation: Double,
    expectedDropped: Seq[OPFeature],
    expectedMapKeys: Seq[String],
    expectedDroppedMapKeys: Map[String, Set[String]]
  ): Unit = {
    def getFilter(maxCorrelation: Double): RawFeatureFilter[Passenger] = new RawFeatureFilter(
      trainingReader = dataReader,
      scoreReader = Some(simpleReader),
      bins = 10,
      minFill = 0.0,
      maxFillDifference = 1.0,
      maxFillRatioDiff = Double.PositiveInfinity,
      maxJSDivergence = 1.0,
      maxCorrelation = maxCorrelation)

    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val FilteredRawData(df, dropped, droppedKeyValue) = getFilter(maxCorrelation).generateFilteredRaw(features, params)

    dropped should contain theSameElementsAs expectedDropped.toSeq
    droppedKeyValue should contain theSameElementsAs expectedDroppedMapKeys
    df.schema.fields.map(_.name) should contain theSameElementsAs
      DataFrameFieldNames.KeyFieldName +: features.diff(dropped).map(_.name)
    if (expectedMapKeys.nonEmpty) {
      df.collect(booleanMap).map(_.value.keySet).reduce(_ + _) should contain theSameElementsAs expectedMapKeys
      df.collect(numericMap).map(_.value.keySet).reduce(_ + _) should contain theSameElementsAs expectedMapKeys
      df.collect(stringMap).map(_.value.keySet).reduce(_ + _) should contain theSameElementsAs expectedMapKeys
    } else {
      intercept[IllegalArgumentException] { df.collect(booleanMap) }
      intercept[IllegalArgumentException] { df.collect(numericMap) }
      intercept[IllegalArgumentException] { df.collect(stringMap) }
    }
  }
}
