import pandas as pd

# the original FA-KES datasets
fakes_train_df = pd.read_csv('input_datasets_old/feature_extraction_train.csv', encoding='latin-1')
fakes_test_df = pd.read_csv('input_datasets_old/feature_extraction_test.csv', encoding='latin-1')

# merging
df_fakes = pd.concat([fakes_train_df, fakes_test_df], ignore_index=True)
print(df_fakes.label.value_counts())

# 1    431
# 0    373
# Name: label, dtype: int64

# After merging, you better split the merged dataset into two datasets (True dataset and Fake Dataset).
# Then choose an equal number of trues and an equal number of fakes to throw
# in a “Test” dataset. Whatever remains is the “Train” dataset.

df_fakes_1 = df_fakes[df_fakes['label'] == 1]
df_fakes_0 = df_fakes[df_fakes['label'] == 0]

# length of the testing data of FA-KES (20%)
fakes_20 = int(round(0.2 * len(df_fakes)))
# Then choose an equal number of trues and an equal number of fakes to throw
# fakes_20 = 161, 161/2 = 80.5 ==> so, we will do: 80 labels 0, 81 labels 1
num_labels_half_0s = int(fakes_20/2)
num_labels_half_1s = fakes_20 - num_labels_half_0s

feature_extraction_test_1 = df_fakes_1[: num_labels_half_1s]
feature_extraction_train_1 = df_fakes_1[num_labels_half_1s:]

feature_extraction_test_2 = df_fakes_0[: num_labels_half_0s]
feature_extraction_train_2 = df_fakes_0[num_labels_half_0s:]

feature_extraction_test = pd.concat([feature_extraction_test_1, feature_extraction_test_2], ignore_index=True)
feature_extraction_train = pd.concat([feature_extraction_train_1, feature_extraction_train_2], ignore_index=True)

# shuffle the rows of both the training and the testing
feature_extraction_test = feature_extraction_test.sample(frac=1).reset_index(drop=True)
feature_extraction_train = feature_extraction_train.sample(frac=1).reset_index(drop=True)

feature_extraction_test.to_csv('../input_datasets_updated/feature_extraction_test_updated.csv', index=False)
feature_extraction_train.to_csv('../input_datasets_updated/feature_extraction_train_updated.csv', index=False)

# reading old buzzfeed datsets
buzzfeed_train_df = pd.read_csv('input_datasets_old/buzzfeed_feature_extraction_80_training.csv')
buzzfeed_test_df = pd.read_csv('input_datasets_old/buzzfeed_feature_extraction_20_testing.csv')

# merging them together
df_buzzfeed = pd.concat([buzzfeed_train_df, buzzfeed_test_df], ignore_index=True)
print(df_buzzfeed.label.value_counts())

# 1    1090
# 0      64
# Name: label, dtype: int64

# the 0s and the 1s alone
df_buzzfeed_0 = df_buzzfeed[df_buzzfeed['label'] == 0]
df_buzzfeed_1 = df_buzzfeed[df_buzzfeed['label'] == 1]

# approximately, 94% of labels are 1, 6% of labels are 0

# length of the testing data of buzzfeed
buzzfeed_20 = int(round(0.2 * len(df_buzzfeed)))

# percentage of 1s and 0s in the dataset
perct_1 = len(df_buzzfeed_1)/len(df_buzzfeed)
perct_0 = len(df_buzzfeed_0)/len(df_buzzfeed)

# apply same percentage for the testing (i.e ensure 94% 1s and 6% 0s to be found in the testing)
num_labels_1 = int(round(perct_1 * buzzfeed_20))
num_labels_0 = int(round(perct_0 * buzzfeed_20))

buzzfeed_feature_extraction_test_1 = df_buzzfeed_1[:num_labels_1]
buzzfeed_feature_extraction_train_1 = df_buzzfeed_1[num_labels_1:]

buzzfeed_feature_extraction_test_0 = df_buzzfeed_0[:num_labels_0]
buzzfeed_feature_extraction_train_0 = df_buzzfeed_0[num_labels_0:]

buzzfeed_feature_extraction_test = pd.concat([buzzfeed_feature_extraction_test_1, buzzfeed_feature_extraction_test_0], ignore_index=True)
buzzfeed_feature_extraction_train = pd.concat([buzzfeed_feature_extraction_train_1, buzzfeed_feature_extraction_train_0], ignore_index=True)

# shuffle the rows to avoid all 0s coming after all 1s
buzzfeed_feature_extraction_test = buzzfeed_feature_extraction_test.sample(frac=1).reset_index(drop=True)
buzzfeed_feature_extraction_train = buzzfeed_feature_extraction_train.sample(frac=1).reset_index(drop=True)

buzzfeed_feature_extraction_test.to_csv('../input_datasets_updated/buzzfeed_feature_extraction_test_20_updated.csv', index=False)
buzzfeed_feature_extraction_train.to_csv('../input_datasets_updated/buzzfeed_feature_extraction_train_80_updated.csv', index=False)


# fakes_20 = int(0.2 * len(df_fakes))
# fakes_80 = int(0.8 * len(df_fakes))
# buzzfeed_20 = int(0.2 * len(df_buzzfeed))
# buzzfeed_80 = int(0.8 * len(df_buzzfeed))
#
#
# print('20 percent of FA-KES: %d' % fakes_20)
# print('80 percent of FA-KES: %d' % fakes_80)
#
# print('20 percent of Buzzfeed: %d' % buzzfeed_20)
# print('80 percent of Buzzfeed: %d' % buzzfeed_80)

# 20 percent of FA-KES: 160
# 80 percent of FA-KES: 643
# 20 percent of Buzzfeed: 230
# 80 percent of Buzzfeed: 923

