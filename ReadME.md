# Automatic Detection of Fake News in the Syrian War
This repository contains the code and input needed to reproduce our work.
  * feature_extraction, and learning_model are the source code directories that contain the code behind the work
  * main.py is the file needed to reproduce our work.
  * feature_selection/ is the directory that contains feature selection code. It accounts for both regression and classification, can run various FS methods, can take any training/testing dataset, can scale data if specified

# Testing feature extraction:
**In order to test the feature extraction approach in our work, you need to first download the 'stanford-corenlp-full-2018-02-27' Stanford dependency parser which should consist of the following**:
1. stanford-corenlp-3.9.1.jar
2. stanford-corenlp-3.9.1-models.jar

It can be downloaded using the following link: http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

**You will also need to download the StanfordNERTagger 'stanford-ner-2018-02-27' which consists of the following**:
1. classifiers/english.all.3class.distsim.crf.ser.gz
2. stanford-ner.jar

It can be downloaded using the following link: https://nlp.stanford.edu/software/stanford-ner-2018-02-27.zip

**Download the above two requirements and save them in this directory.**
Inside main.py, uncomment test_feature_extraction() from the main function in order to test feature extraction. Feature extraction csv files will be outputted into the output directory

# Testing machine learning models:
By running the main.py function, you run the best found machine learning model on the FA-KES train and test datasets
You can change the paths in the ```test_features_model()``` function in the main.py file in order to test it on other datasets. The other datasets are found in the other_datasets directory (Buzzfeed train and test features extraction are there)
comment the ```test_features_model()``` in the main function call in the main.py file if you do not want to test the feature-based machine learning model.

# Testing baseline text-based deep learning model:
  * You need to download the Glove file glove.6B.300d.txt and place it in the same directory as the main.py file
  * It can be downloaded from this link: http://nlp.stanford.edu/data/glove.6B.zip
  * uncomment the ```test_text_model()``` in the main function call in the main.py file if you want to test the text-based deep learning model.
  * You can change the paths in the ```test_text_model()``` function in the main.py file in order to test it on other datasets. The other datasets are found in the other_datasets directory (Buzzfeed train and test features extraction are there)

# Meta-Learning vs. Shallow Models 
  * We have applied meta learning to the FAKES dataset. More specifically, we experimented with MAML
  * We have compared performance to shallow models
  * All Experiments can be found [here](https://github.com/fakenewssyria/fake_news_detection/tree/master/Experiments)

# Adavnced ML Evaluations: Meta Learning Vs. Shallow
We have also run advanced machine learning evaluations to assess the quality of the predictions of each of the best meta learners and best performing shallow model:
  * Results can be found [here](https://github.com/fakenewssyria/fake_news_detection/tree/master/Experiments/advanced_ml_plots_all)
  * Code for Running Advanced Evaluations can be found [here](https://github.com/fakenewssyria/fake_news_detection/blob/master/Experiments/AdvanedEvaluation.py)

# Crowd Sourced Annotations
 * Can be found [here](https://github.com/fakenewssyria/fake_news_detection/blob/master/crowdsourced-annotations.xlsx)
