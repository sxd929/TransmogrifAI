title=Quick Start
type=doc
status=published
~~~~~~

##Running Optimus Prime from spark shell

Start up your spark shell:

```scala
./spark-shell --jars optimus-prime-3.3.0-all.jar
```

Create your spark session:

```scala
// Set up a SparkSession as normal
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

val conf = new SparkConf().setAppName("TitanicPrediction")
implicit val spark = SparkSession.builder.config(conf).getOrCreate()
```

Import optimus prime:
```scala
import com.salesforce.op._
```

Now follow along with the rest of the code from the Titanic example found [here](https://github.salesforceiq.com/einstein/optimus-prime/wiki/Example:-Titanic).

---

##Bootstrap Your First Project

We provide a convenient way to bootstrap you first project with Optimus Prime using the OP CLI.
As an illustration, let's generate a binary classification model with the Titanic passenger data.

### Prerequisites
* Java 1.8
* Spark 2.2.1 - [Download](https://spark.apache.org/downloads.html), unzip it and then set an environment variable: `export SPARK_HOME=<SPARK_FOLDER>`
* Generate Github api token [here](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line) and then set it as an environment variable: `export GITHUB_API_TOKEN=<MY_TOKEN>`

### Quick Start
Clone the Optimus Prime repo:
```bash
git clone git@github.salesforceiq.com:einstein/optimus-prime.git
```

Build the OP CLI by running:
```bash
cd ./optimus-prime
./gradlew cli:shadowJar
alias op="java -cp `pwd`/cli/build/libs/\* com.salesforce.op.cli.CLI"
```
Finally generate your Titanic model project (follow the instructions on screen):
```
op gen --input `pwd`/test-data/PassengerDataAll.csv \
  --id passengerId --response survived \
  --schema `pwd`/test-data/PassengerDataAll.avsc Titanic
```  

If you run this command more than once, two important command line arguments will be useful:
- `--overwrite` will allow to overwrite an existing project; if not specified, the generator will fail
- `--answers answers file` will provide answers to the questions that the generator asks.

e.g.
```
op gen --input `pwd`/test-data/PassengerDataAll.csv --id passengerId --response survived --schema `pwd`/test-data/PassengerDataAll.avsc --answers cli/passengers.answers Titanic --overwrite
```
will do the generation without asking you anything.

Here we have specified the schema of the input data as an Avro schema. Avro is the schema format that the OP CLI understands. Note that when writing up your machine learning workflow by hand, you can always use case classes instead.  

Your Titanic model project is ready to go. 

You will notice a default set of [FeatureBuilders](Documentation#featurebuilders) generated from the provided Avro schema. You are encouraged to edit this code to customize feature generation and take full advantage of the Feature types available (selecting the appropriate type will improve automatic feature engineering steps).
 
The generated code also uses the ```.vectorize()``` shortcut to apply default feature transformations to the raw features and create a single feature vector. This is in essence the [automatic feature engineering Stage](AutoML-Stages#vectorizers) of Optimus Prime. Once again, you can customize and expand on the feature manipulations applied by acting directly on individual features before applying ```.vectorize()```. You can also choose to completely discard ```.vectorize()``` in favor of hand-tuned feature engineering and manual vector creation using the VectorsCombiner Estimator (short-hand ```Vectorizers.combine()```) if you desire to have complete control over the feature engineering.

For convenience we have provided a simple `OpAppWithRunner` (and a more customizable `OpApp`) which takes in a workflow and allows you to run spark jobs from the command line rather than creating your own Spark App.

```scala
object Titanic extends OpAppWithRunner with TitanicWorkflow {
   def runner(opParams: OpParams): OpWorkflowRunner =
    new OpWorkflowRunner(
      workflow = titanicWorkflow,
      trainingReader = trainingReader,
      scoringReader = scoringReader,
      evaluator = evaluator,
      scoringEvaluator = None,
      featureToComputeUpTo = featureVector,
      kryoRegistrator = classOf[TitanicKryoRegistrator]
    )
}
```

This app is generated as part of the template and can be run like this:

```bash
cd titanic
./gradlew installDist
./gradlew sparkSubmit -Dmain=com.salesforce.app.Titanic -Dargs="--run-type=train --model-location=/tmp/titanic-model --read-location Passenger=`pwd`/../test-data/PassengerDataAll.csv"
```


To generate a project for any other dataset, simply modify the parameters to point to your specific data and its schema.

Happy modeling!

___