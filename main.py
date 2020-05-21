from feature_extraction.feature_extraction import FeatureExtraction
from exploratory_analysis.exploratory_analysis import ExploratoryAnalysis
from learning_model.feature_based import FeatureBased
import matplotlib.pyplot as plt
from learning_model.text_based import TextBased
import os

def test_feature_extraction():

    # some dependencies needed for feature extraction
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    
    # it is assumed that the stanford parsers are located in the same directory as the cloned github repository
    # if this is not the case, change the path_to_stanford_parsers variable below to the absolute path of the stanford parsers
    # e.g. /Users/username/pythonworkspace/fake-news-detection
    path_to_stanford_parsers = os.getcwd()
    
    feature_extraction = FeatureExtraction(path_to_stanford_parsers)
    feature_extraction.read_clustering_output("input/fakes.csv")
    feature_extraction.read_lexicons(assertive_lexicon_path="input/lexicons/assertive.txt",
                                         factive_lexicon_path="input/lexicons/factive.txt",
                                         implicative_lexicon_path="input/lexicons/implicative.txt",
                                         hedges_lexicon_path="input/lexicons/hedges.txt",
                                         sectarian_lexicon_path="input/lexicons/sectarian_language_lexicon.txt",
                                         report_lexicon_path="input/lexicons/report_verbs.txt",
                                         bias_lexicon_path="input/lexicons/bias.txt",
                                         subjclues_lexicon_path="input/lexicons/subjclues.txt",
                                         negative_lexicon_path="input/lexicons/negative_words.txt",
                                         positive_lexicon_path="input/lexicons/positive_words.txt")    

    feature_extraction.calculate_features()
    feature_extraction.export_features_csv("output/feature_extraction.csv")

def test_features_model():
    
    learning_model = FeatureBased()
    learning_model.read_split_dataset("input/feature_extraction_train.csv", "input/feature_extraction_test.csv")
    learning_model.create_decision_tree()

def test_text_model():
    text_based = TextBased()
    text_based.run("input/feature_extraction.csv", "glove.6B.300d.txt")
    
if __name__ == '__main__':
    print("uncomment your experiment of choice in the main function of main.py")
    #test_feature_extraction()
    test_features_model()    
    # test_text_model()