"""
ML Engg: Viral Gorecha

Kindly Note: This module has been tested 
on the local run of the interactive tfx pipeline
and has been found to function as expected
using the TFTransform component.
"""

import tensorflow as tf
import tensorflow_transform as tft

def transformed_name(key: str) -> str:
    return key + "_xf"

def preprocessing_fn(inputs: tf.Tensor) -> tf.Tensor:
    
    entity_1 = inputs['entity_1']
    entity_2 = inputs['entity_2']

    # Removal of extra spaces at ends
    entity_1 = tf.strings.strip(entity_1)
    entity_2 = tf.strings.strip(entity_2)

    # lowering of all cases
    lowered_entity_1 = tft.tf.strings.lower(entity_1)
    lowered_entity_2 = tft.tf.strings.lower(entity_2)

    # Short names removal (assumes names with length <= 3 on both sides are short)
    lowered_entity_1 = tft.tf.where(tft.tf.strings.length(lowered_entity_1) > 3, lowered_entity_1, '')
    lowered_entity_2 = tft.tf.where(tft.tf.strings.length(lowered_entity_2) > 3, lowered_entity_2, '')

    lowered_entity_1 = tf.strings.regex_replace(lowered_entity_1, pattern="[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", rewrite='')
    lowered_entity_2 = tf.strings.regex_replace(lowered_entity_2, pattern="[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]", rewrite='')

    transformed_entity_1 = tft.compute_and_apply_vocabulary(lowered_entity_1, top_k = 10000)
    transformed_entity_2 = tft.compute_and_apply_vocabulary(lowered_entity_2, top_k = 10000)

    # tokenized_entity_1 = 
    # tokenized_entity_2

    # max_length = tf.math.maximum(
    #     tf.reduce_max(tft.word_count(tokenzied_entity_1)),
    #     tf.reduce_max(tft.word_count(tokenzied_entity_2))
    # )

    # Padding
    # padded_entity_1 = tf.keras.preprocessing.sequence.pad_sequences(transformed_entity_1, maxlen=max_length)
    # padded_entity_2 = tf.keras.preprocessing.sequence.pad_sequences(transformed_entity_2, maxlen=max_length)

    return {
        transformed_name("entity_1"): transformed_entity_1,
        transformed_name("entity_2"): transformed_entity_2,
        transformed_name("tag"): inputs["tag"]
    }

