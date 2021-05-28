import pandas as pd

df_fakes_train = pd.read_csv('input/feature_extraction_train_updated.csv')
df_fakes_test = pd.read_csv('input/feature_extraction_test_updated.csv')

print('\nFA-KES training: Label Distribution')
print(df_fakes_train.label.value_counts() / len(df_fakes_train))
print('\nFA-KES testing: Label Distribution')
print(df_fakes_test.label.value_counts() / len(df_fakes_test))

df_buzz_train = pd.read_csv('input/buzzfeed_feature_extraction_train_80_updated.csv')
df_buzz_test = pd.read_csv('input/buzzfeed_feature_extraction_test_20_updated.csv')

print('\nBuzzFeed training: Label Distribution')
print(df_buzz_train.label.value_counts() / len(df_buzz_train))
print('\nBuzzFeed testing: Label Distribution')
print(df_buzz_test.label.value_counts() / len(df_buzz_test))