"""
Kindly Note: This pipeline is a sketch work
architecture / boilerplate code / scratch pad
that can be explored and / or exploited as per needs

ML Engg: Viral Gorecha

Reference: Building Machine Learning Pipelines - O'Reilly"

"""

import kfp.dsl as dsl
import kfp.components as comp

tfx_image = "your-tfx-image:latest"
tfx_pipeline_op = comp.load_component_from_file("tfx_pipeline_op.yaml")

# Define Kubeflow pipeline
@dsl.pipeline(
    name="Entity Matching Pipeline",
    description="End-to-End ML Pipeline for Entity Matching"
)
def entity_matching_pipeline():
    # Data Cleaning Beam component
    data_cleaning = dsl.ContainerOp(
        name='data-cleaning',
        image='your-apache-beam-image:latest',
        command=['python', '/apache-beam/data_cleaning_beam.py'],
        arguments=[
            '--input_file', '/path/to/raw_dataset.csv',
            '--output_file', '/path/to/cleaned_dataset',
            '--stopwords', 'inc,ltd,llc'
        ],
        file_outputs={'output_file': '/apache-beam/cleaned_dataset-00000-of-00001'}
    )

    # ExampleGen component
    example_gen = tfx_pipeline_op(
        'CsvExampleGen',
        image=tfx_image,
        component_name='ExampleGen',
        input_base=data_cleaning.outputs['output_file'],
    )

     # StatisticsGen component
    statistics_gen = tfx_pipeline_op(
        'StatisticsGen',
        image=tfx_image,
        component_name='StatisticsGen',
        examples=example_gen.outputs['examples'],
    )

    # SchemaGen component
    schema_gen = tfx_pipeline_op(
        'SchemaGen',
        image=tfx_image,
        component_name='SchemaGen',
        statistics=statistics_gen.outputs['statistics'],
    )

    # Transform component
    transform = tfx_pipeline_op(
        'Transform',
        image=tfx_image,
        component_name='Transform',
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
    )

    # Trainer component
    trainer = tfx_pipeline_op(
        'Trainer',
        image=tfx_image,
        component_name='Trainer',
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
    )

    # Model Resolver component
    model_resolver = tfx_pipeline_op(
        'ModelResolver',
        image=tfx_image,
        component_name='ModelResolver',
        model=trainer.outputs['model'],
        model_blessing=trainer.outputs['model_blessing'],
    )

    # InfraValidator component
    infra_validator = tfx_pipeline_op(
        'InfraValidator',
        image=tfx_image,
        component_name='InfraValidator',
        model=trainer.outputs['model'],
        examples=example_gen.outputs['examples'],
    )

    # Pusher component
    pusher = tfx_pipeline_op(
        'Pusher',
        image=tfx_image,
        component_name='Pusher',
        model=trainer.outputs['model'],
        infra_blessing=infra_validator.outputs['blessing'],
        push_destination='/path/to/serving_model',
    )

# Compile and deploy the pipeline
pipeline_filename = 'entity_matching_pipeline.yaml'
kfp.compiler.Compiler().compile(entity_matching_pipeline, pipeline_filename)
