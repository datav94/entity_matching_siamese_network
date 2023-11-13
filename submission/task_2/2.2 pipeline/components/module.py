"""
ML Engg: Viral Gorecha

The below module is structured according to 
the Model section of the book
"Building Machine Learning Pipelines - O'Reilly"

The module has been locally tested on an interactive tfx pipeline

It functions efficiently with just a dimensionality error to be resolved 
in the get_model() function
"""


import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_transform as tft
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda, Dropout

LABEL_KEY = "tag"
os.environ["TFHUB_CACHE_DIR"] = "./tmp/tfhub"

def get_model(show_summary: bool = True) -> tf.keras.models.Model:

    embedding_dim=50
    lstm_units=100
    vocab_size = 500

    input_a = Input(shape=(None,), name=transformed_name("entity_1"))
    input_b = Input(shape=(None,), name=transformed_name("entity_2"))

    # Embedding layer for mapping tokens to vectors
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

    # Shared LSTM layer
    shared_lstm = LSTM(lstm_units)

    embedding_a = embedding_layer(tf.reshape(input_a, [-1]))
    embedding_b = embedding_layer(tf.reshape(input_b, [-1]))

    # Process each input through the embedding and LSTM layers
    output_a = shared_lstm(embedding_a)
    output_b = shared_lstm(embedding_b)

    # Calculate L1 distance between the two representations
    l1_distance = Lambda(lambda x: tf.abs(x[0] - x[1]))([output_a, output_b])

    # Dense layer to make the final prediction
    dense1 = Dense(50, activation='relu')(l1_distance)
    droput = Dropout(0.2)(dense1)
    dense2 = Dense(20, activation='relu')(droput)
    prediction = Dense(1, activation='sigmoid')(dense2)

    # Create the siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=prediction)

    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(),
        ],
    )

    if show_summary:
        siamese_model.summary()

    return siamese_model



def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")



def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn



def transformed_name(key: str) -> str:
    return key + "_xf"



def _input_fn(file_pattern, tf_transform_output, batch_size=500):
    """Generates features and label for tuning/training.

    Args:
    file_pattern: input tfrecord file pattern.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of
          label indices.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def run_fn(fn_args):
    """Train the model based on given args.

    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 64)

    print(train_dataset)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    callbacks = [tensorboard_callback]

    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)