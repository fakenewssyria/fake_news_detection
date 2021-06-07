import pandas as pd
from feature_selection_code import FeatureSelection
import os

if __name__ == '__main__':
    df_train = pd.read_csv('input/feature_extraction_train_updated.csv', encoding='latin-1')
    df_test = pd.read_csv('input/feature_extraction_test_updated.csv', encoding='latin-1')
    df = pd.concat([df_train, df_test])

    col_len = len(list(df.columns))

    output_folder = 'output/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fs = FeatureSelection(df, target_variable='label',
                          output_folder=output_folder,
                          cols_drop=['article_title', 'article_content', 'source', 'source_category', 'unit_id'],
                          scale=True,
                          scale_input=True,
                          scale_output=False,
                          output_zscore=False,
                          output_minmax=False,
                          output_box=False,
                          output_log=False,
                          input_zscore=(0, len(df.drop(['article_title', 'article_content', 'source', 'source_category', 'unit_id'], axis=1).columns) - 1),
                          input_minmax=None,
                          input_box=None,
                          input_log=None,
                          regression=False)

    fs.drop_zero_std()
    fs.drop_low_var()
    fs.drop_high_correlation()
    fs.feature_importance(xg_boost=True, extra_trees=True)
    fs.univariate()
    fs.rfe()
