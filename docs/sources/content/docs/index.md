title=Overview
type=doc
status=published
~~~~~~

abstraction for creating and running machine learning workflows. The abstraction is made up of Features, Stages, Workflows and Readers which interact as shown in the diagram below.

<img src="../img/AbstractionDiagram-cropped.png" alt="TransmogrifAI Overview" width="750px" text-align="center"/>


**Features**: The primary abstraction that users of TransmogrifAI interact with are Features. Features are essentially type-safe pointers to data columns with additional metadata built in. Features are the elements which users manipulate and interact with in order to define all steps in the machine learning workflow. In our abstraction Features are acted on by Stages in order to produce new Features. Part of the metadata contained in Features is strict type information about the column. This is used both to determine which Stages *can* be called on a given Feature and which Stages *should* be called when automatic feature engineering Stages are used. Because the output of every Stage is a Feature or set of Features, any sequence of type safe operations can be strung together to create a machine learning workflow. 

**Stages**: Stages define actions that you wish to perform on Features in your workflow. Those familiar with Spark ML will recognize the idea of Stages being either **Transformers** or **Estimators**. Transformers provide functions for transforming one or more Features in your data to one or more *new* Features. Estimators provide algorithms which when applied to one or more Features produce Transformers. The TransmogrifAI Transformers and Estimators extend Spark ML Transformers and Estimators and can be used as standard Spark stages if desired. In both Spark ML and TransmogrifAI when Stages are used within a workflow the user does not need to distinguish between types of Stages (Estimator or Transformer), this distinction is only important for developers developing new Estimators or Transformers.

**Workflows and Readers**: Once the final desired Feature, or Features, have been defined they are materialized by feeding the final Features into a Workflow. The Workflow will trace back how the final Features were created and make an optimized DAG of Stage executions in order to produce the final Features. The Workflow must also be provided a DataReader. The DataReader can do complex data pre-processing steps or simply load a dataset. The key component of the DataReader is that the type of the data produced by the reader must match the type of the data expected by the initial feature generation stages.

___