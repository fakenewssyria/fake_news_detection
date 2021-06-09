# Experiments

This directory contains all experiments regarding the FA-KES Dataset

## FAKES_MAML
Contains the code used for running MAML algorithm on the FAKES dataset. The code is originally taken from [here](https://github.com/cbfinn/maml). Minor modifications to the code are made so that it could:
  * Be run on Structure/Tabular 2D datasets
  * Take as input the training and testing datasets

## Shallow
Contains the code used for running cross validation with hyper parameter tuning on FAKES datasets.

## Advanced ML Evaluations
In order to assess **qualitatively** the predictions of the meta learners and the shallow models, we need to do some risk analysis. More specifically:

### Mean Emprical Risks
Good performing models should not only have good accuracy scores, they should be able to assign a high probability to data points which are most likely to belong to a certain class. Luckilly, some sklearn models are probabilistic, also, in the meta learning exercises we are able to get a probability score for each prediction by applyying softmax function to the logits, which are the raw predictions made by the model. 
Mean Empirical Risk Curves help us assess this. More specifically, we rank the news by descending order of their estimated risk scores (probability of +ve class). 
The, we then group news into bins based on the percentiles they fall into when categorized using risk scores. In our experiments, we choose to create 10 bins. 
The bottom 10\% of news who have the least risk are grouped into a single bin. Those that rank between 10th and 20th percentile are grouped in the next bin and so on. 
For each such bin, we compute the **empirical risk score** _which is the fraction of news from that bin who actually, as per ground truth, are positive._ A good model would be classifying news correctly if the _empirical risk curve_ is monotonically non-decreasing.
If the empirical risk curve is non-monotonic for some models, it implies that the classification using the model's risk scores may result in scenarios where news with lower risk scores are more likely to be fake compared to news with higher risk scores.
![Mean Empirical Risks](https://github.com/fakenewssyria/fake_news_detection/blob/master/Experiments/advanced_ml_plots_all/mean_empirical_risks.png)

### Precision + Recall at Top K
Precision and Recall at Top K news that are at risk of being Positive.

![Precision Top K](https://github.com/fakenewssyria/fake_news_detection/blob/master/Experiments/advanced_ml_plots_all/precisions_topK.png)
![Recall Top K](https://github.com/fakenewssyria/fake_news_detection/blob/master/Experiments/advanced_ml_plots_all/recalls_topK.png)

### ROC-AUC Curves
![ROC AUC Curve](https://github.com/fakenewssyria/fake_news_detection/blob/master/Experiments/advanced_ml_plots_all/roc_curves.png)
