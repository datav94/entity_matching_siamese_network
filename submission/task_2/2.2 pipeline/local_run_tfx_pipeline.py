# TODO: The components below Trainer need to be examined again

"""
Running this pipeline will recreate the errors as rising
by Resolver node that needs to be corrected for the Channel

The pipeline run has be checked until transform to be 
running as required.

The trainer component lacks dimensional linkage that needs
to be cleared at the LSTM layer input from the embedding
Input.

The model run in local env using jupyter can be seen in 
entity_matching.ipynb
"""


import os
import tfx
import tensorflow as tf
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import SchemaGen
from tfx.components import ExampleValidator
from tfx.components import Transform
from tfx.v1.dsl import Resolver
from tfx.v1.orchestration import LocalDagRunner
from tfx.v1.types.standard_artifacts import Model, ModelBlessing
from tfx.proto import example_gen_pb2
from tfx.v1.proto import Output, SplitConfig
from tfx.proto import trainer_pb2

_MODULE_FILE = "../pipeline/components/transform.py"
TRAINING_STEPS = 1000
EVALUATION_STEPS = 100
_pipeline_root = "../pipeline"
trainer_file = "../pipeline/components/module.py"
_INPUT_DATA_DIR = "../data"
_serving_model_dir = os.path.join(_pipeline_root, 'serving_model')
# _serving_model_dir_lite = os.path.join(_pipeline_root, 'serving_model_lite')

# Setup the DAG
context = LocalDagRunner()._configure_context(pipeline_name="entity_matching_pipeline")

with context:

    # Define split strategy
    output = Output(
        split_config = SplitConfig(
            splits=[
                SplitConfig.Split(name="train", hash_buckets=6),
                SplitConfig.Split(name="eval", hash_buckets=2),
                SplitConfig.Split(name="test", hash_buckets=2),
            ]
        )
    )
    
    example_gen = CsvExampleGen(
    input_base=os.path.join(os.getcwd(), _INPUT_DATA_DIR),
    output_config=output
    )

    # Statistics Generator
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )

    # Schema Generator
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True
    )

    # Example Validator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )

    # Tensorflow Transform Component
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=_MODULE_FILE
    )

    # Trainer
    trainer = tfx.components.Trainer(
        module_file=trainer_file,
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAINING_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVALUATION_STEPS)
    )

    # Model Resolver
    model_resolver = Resolver(
        instance_name="latest_blessed_model_resolver",
        strategy_class=tfx.v1.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.v1.dsl.Channel(type=Model),
        model_blessing=tfx.v1.dsl.Channel(type=ModelBlessing))

    # Setup the serving components
    infra_validator = tfx.components.InfraValidator(
        model=trainer.outputs['model'],
        examples=example_gen.outputs['examples'],
        serving_spec=tfx.proto.serving_pb2.InfraValidatorSpec(),
        instance_name="infra_validator")

    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        infra_blessing=infra_validator.outputs['blessing'],
        push_destination=tfx.proto.pusher_pb2.PushDestination(
            filesystem=tfx.proto.pusher_pb2.PushDestination.Filesystem(
                base_directory=_serving_model_dir)),
        instance_name="pusher")

# Run the pipeline
LocalDagRunner().run(context)