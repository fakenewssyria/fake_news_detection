import pickle
import os


def get_results_dementia(trained_models_dir, metric='f2'):
    best_f2 = 0.0
    best_model = ''
    for root, dirs, files in os.walk(trained_models_dir):
        if 'error_metrics.p' in files and 'std_metrics.p' in files:
            with open(os.path.join(root, 'error_metrics.p'), 'rb') as f:
                error_metrics = pickle.load(f)

            if float(error_metrics[metric]) > float(best_f2):
                best_f2 = error_metrics[metric]
                # best_model = root.split('_')[4]
                best_model = root

    print('best {} so far: {} in model {}'.format(metric, best_f2, best_model))


if __name__ == '__main__':
    dir1 = 'fake_news/'

    print('dir: {}'.format(dir1))
    get_results_dementia(dir1, metric='accuracy')
    print('======================')
    # print('dir: {}'.format(dir2))
    # get_results_dementia(dir2, metric='f2')
    # print('======================')
    # print('dir: {}'.format(dir3))
    # get_results_dementia(dir3, metric='f2')
    # print('======================')
