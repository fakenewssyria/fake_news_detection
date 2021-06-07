import pandas as pd
import time
from models_hyperparams import models_to_test, non_probabilistic_models_to_test
from models_hyperparams import hyperparameters_per_model as hyperparameters
from cross_validation import LearningModelCrossVal


if __name__ == '__main__':
    training_data_path = '../input/feature_extraction_train_updated.csv'
    testing_data_path = '../input/feature_extraction_test_updated.csv'

    train_df = pd.read_csv(training_data_path, encoding='latin-1')
    test_df = pd.read_csv(testing_data_path, encoding='latin-1')

    cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']
    lm = LearningModelCrossVal(train_df, test_df, output_folder='output_xgboost_poslabel0/',
                               cols_drop=cols_drop,
                               over_sample=False,
                               standard_scaling=True,
                               minmax_scaling=False,
                               learning_curve=False)

    for model in models_to_test:
        model_name = models_to_test[model]
        print('\n********** Results for %s **********' % model_name)
        t0 = time.time()
        lm.cross_validation(model, hyperparameters[model_name], model_name, 10,
                            lm.test_df, 10, probabilistic=True, pos_class_label=0)
        t1 = time.time()
        print('function took %.3f min\n' % (float(t1 - t0) / 60))

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')