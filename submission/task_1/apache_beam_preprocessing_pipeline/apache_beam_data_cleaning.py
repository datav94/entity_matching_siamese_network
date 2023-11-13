import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from imblearn.over_sampling import SMOTE
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sklearn.model_selection import train_test_split

class CleanTextFn(beam.DoFn):
    def __init__(self, stopwords=None):
        self.stopwords = stopwords or set()
        self.lemmatizer = WordNetLemmatizer()

    def process(self, element):
        entity_1, entity_2 = element['entity_1'], element['entity_2']

        # Standardize formatting
        entity_1 = entity_1.lower().strip()
        entity_2 = entity_2.lower().strip()

        # Remove stop words
        entity_1 = ' '.join([word for word in entity_1.split() if word not in self.stopwords])
        entity_2 = ' '.join([word for word in entity_2.split() if word not in self.stopwords])

        # Remove special characters
        entity_1 = re.sub(r'[^\w\s]', '', entity_1)
        entity_2 = re.sub(r'[^\w\s]', '', entity_2)

        # Tokenization and lemmatization
        entity_1 = ' '.join([self.lemmatizer.lemmatize(word) for word in word_tokenize(entity_1)])
        entity_2 = ' '.join([self.lemmatizer.lemmatize(word) for word in word_tokenize(entity_2)])

        yield {
            'entity_1': entity_1,
            'entity_2': entity_2,
            'label': element['tag']
        }

def preprocess_for_smote(element):
    # Convert label to integer for SMOTE
    element['label'] = int(element['label'])
    return element

def run_data_cleaning_beam(input_file, output_file, stopwords=None):
    with beam.Pipeline(options=PipelineOptions()) as p:
        # Read data from CSV
        data = (
            p
            | 'ReadFromCSV' >> beam.io.ReadFromText(input_file, skip_header_lines=1)
            | 'ParseCSV' >> beam.Map(lambda line: dict(zip(['entity_1', 'entity_2', 'label'], line.split(','))))
            | 'CleanText' >> beam.ParDo(CleanTextFn(stopwords=stopwords))
        )

        # Split data into train and test sets
        train_data, test_data = (
            data
            | 'TrainTestSplit' >> beam.Partition(lambda _, i: i % 2 == 0, 2)
        )

        # Apply SMOTE to the training set
        smote_train_data = (
            train_data
            | 'PreprocessForSMOTE' >> beam.Map(preprocess_for_smote)
            | 'MapToKV' >> beam.Map(lambda x: (x['label'], x))
            | 'GroupByLabel' >> beam.GroupByKey()
            | 'ApplySMOTE' >> beam.Map(lambda  elements: SMOTE().fit_resample(pd.DataFrame(elements)))
            | 'FlattenSMOTE' >> beam.FlatMap(lambda elements: elements.to_dict(orient='records'))
        )

        # # Combine the SMOTE-treated training data with the original test data
        # final_train_data = (
        #     smote_train_data
        #     | 'FinalTrainData' >> beam.Flatten()
        #     | 'CombineWithTest' >> beam.Flatten(test_data)
        # )

        # Write final train data and test data to files
        train_data | 'WriteFinalTrainData' >> beam.io.WriteToCsv(output_file + '_train.csv')
        test_data | 'WriteTestData' >> beam.io.WriteToCsv(output_file + '_test.csv')


if __name__ == '__main__':
    input_file = './data/ds_challenge_alpas.csv'
    cleaned_output_file = './data/ds_challenge_alpas_cleaned.csv'
    stopwords = set(['inc', 'ltd', 'llc', "corp"])

    run_data_cleaning_beam(input_file, cleaned_output_file, stopwords=stopwords)
