"""
ML Engg: Viral Gorecha

Reference: Building Machine Learning Pipelines - O'Reilly"

The below pipeline can be further diversified
to include its run on GCP Vertex AI or using the
old name GCP AI Platform

The Vertex AI Infrastructure configurations can be 
included in the below pipeline for better scaling
of the whole architecture
"""

import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
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



TRAIN_STEPS = 50000
EVAL_STEPS = 10000


def init_components(
    data_dir,
    module_file,
    training_steps=TRAIN_STEPS,
    eval_steps=EVAL_STEPS,
    serving_model_dir=None
):

    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=9),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )

    output = Output(
    split_config = SplitConfig(
            splits=[
                SplitConfig.Split(name="train", hash_buckets=6),
                SplitConfig.Split(name="eval", hash_buckets=2),
                SplitConfig.Split(name="test", hash_buckets=2),
            ]
        )
    )

    # Example Generator
    example_gen = CsvExampleGen(
        input_base=os.path.join(os.getcwd(), data_dir),
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

    # TFT Transform component
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )

    # Training Keyword Arguments
    training_kwargs = {
        "module_file": module_file,
        "examples": transform.outputs["transformed_examples"],
        "schema": schema_gen.outputs["schema"],
        "transform_graph": transform.outputs["transform_graph"],
        "train_args": trainer_pb2.TrainArgs(num_steps=training_steps),
        "eval_args": trainer_pb2.EvalArgs(num_steps=eval_steps),
    }

    # Trainer component
    trainer = tfx.components.Trainer(**training_kwargs)


    # Model resolver component
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


    # Model Evaluation Configuration
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_default",
                label_key="consumer_disputed",
                # preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[tfma.SlicingSpec(), tfma.SlicingSpec(feature_keys=["product"])],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.65}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                ],
            )
        ],
    )

    # Model Evaluator
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    if serving_model_dir:
        pusher = tfx.components.Pusher(
            model=trainer.outputs["model"],
            model_blessing=evaluator.outputs["blessing"],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=serving_model_dir
                )
            ),
        )
    else:
        raise NotImplementedError(
            "Provide serving_model_dir."
        )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        infra_validator
        pusher,
    ]
    return components
