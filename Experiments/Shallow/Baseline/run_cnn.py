import pandas as pd
from text_based import TextBased

training_data_path = '../input_datasets_updated/feature_extraction_train_updated.csv'
testing_data_path = '../input_datasets_updated/feature_extraction_test_updated.csv'

train_df = pd.read_csv(training_data_path, encoding='latin-1')
test_df = pd.read_csv(testing_data_path, encoding='latin-1')
tb = TextBased(train_df, test_df)

tb.pre_process_data()
tb.text_to_sequences()
tb.train_val_split()
cnn_model = tb.build_model_FeedForward()
tb.test_model(cnn_model, 'cnn', 'output/')