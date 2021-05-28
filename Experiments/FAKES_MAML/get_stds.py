import pickle

with open('trained_models/model_3/std_metrics.p', 'rb') as f:
    std_metrics = pickle.load(f)

for k,v in std_metrics.items():
    print('{}: {}'.format(k, v))