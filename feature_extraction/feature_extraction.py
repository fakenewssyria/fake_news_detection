from feature_extraction.lexicon_features import LexiconFeatures
from feature_extraction.consistency_score import ConsistencyScore
from feature_extraction.quoted_sources import QuotedSourcesFeature

import pandas as pd
import numpy as np


class FeatureExtraction:
    
    def __init__(self, path_to_parsers):
        
        self.lexicon_features = LexiconFeatures()
        self.consistency_score = ConsistencyScore()
        self.quoted_sources = QuotedSourcesFeature(path_to_parsers)
        
        self.features_output = pd.DataFrame()
         
    def read_clustering_output(self, clustering_output_path):
        
        ''' function that reads our labeled dataset '''
        self.clustering_output = pd.read_csv(clustering_output_path, encoding='latin-1', header='infer')
        
    def calculate_features(self):
        
        ''' function that calculates the values of each of our features '''
        
        self.calculate_lexicons_features()
        self.calculate_consistency_score()    
        self.set_description_of_sources_quoted()
        
    def calculate_consistency_score(self):
        
        print("calculating consistency score feature")

        ''' function that calculates the consistency score feature '''
        self.consistency_score.calculate_consistency_score(self.clustering_output)
    
    def set_description_of_sources_quoted(self):
        
        ''' function that sets the description of sources quoted feature '''
        
        print("calculating description of quoted sources feature")
        articles = self.clustering_output["article_content"]
        self.quoted_sources.set_description_of_sources_quoted(articles, self.lexicon_features.report_lexicon)
        
    def read_lexicons(self, assertive_lexicon_path, factive_lexicon_path, implicative_lexicon_path,
                      hedges_lexicon_path, sectarian_lexicon_path, report_lexicon_path,
                      bias_lexicon_path, subjclues_lexicon_path, negative_lexicon_path, positive_lexicon_path):
        
        ''' function that reads lexicons from files '''
        
        self.lexicon_features.read_assertive_lexicon(assertive_lexicon_path)
        self.lexicon_features.read_factive_lexicon(factive_lexicon_path)
        self.lexicon_features.read_implicative_lexicon(implicative_lexicon_path)
        self.lexicon_features.read_hedges_lexicon(hedges_lexicon_path)
        self.lexicon_features.read_sectarian_lexicon(sectarian_lexicon_path)
        self.lexicon_features.read_report_lexicon(report_lexicon_path)
        self.lexicon_features.read_bias_subj_lexicon(bias_lexicon_path, subjclues_lexicon_path, negative_lexicon_path, positive_lexicon_path)
    
    def calculate_lexicons_features(self):
        
        ''' function that calculates the lexicon features '''
        print("calculating lexicon features")
        articles = self.clustering_output["article_content"]
        self.lexicon_features.calculate_lexicons_features(articles)
    
    def export_features_csv(self, features_path):
        
        features_output = pd.DataFrame()
        
        ''' clustering input and output '''
        
#         features_output["unit_id"] = np.array(self.clustering_output["unit_id"])
#         features_output["article_title"] = np.array(self.clustering_output["article_title"])
        features_output["article_content"] = np.array(self.clustering_output["article_content"])
#         features_output["source"] = np.array(self.clustering_output["source"])

        features_output["label"] = np.array(self.clustering_output["label"])
        
        ''' feature extraction output '''
         
        features_output["assertive_verbs"] = self.lexicon_features.assertive_frequency
        features_output["factive_verbs"] = self.lexicon_features.factive_frequency
        features_output["implicative_verbs"] = self.lexicon_features.implicative_frequency
        features_output["hedges"] = self.lexicon_features.hedges_frequency
        features_output["report_verbs"] = self.lexicon_features.report_frequency
        features_output["bias"] = self.lexicon_features.bias_frequency
        features_output["sectarian_language"] = self.lexicon_features.sectarian_frequency
        
#         features_output["source_category"] = self.consistency_score.clustering_output["source_category"]
        features_output["consistency_score"] = self.consistency_score.consistency_score_array
         
        features_output["quoted_sources"] = self.quoted_sources.quoted_source_labels
         
        features_output.to_csv(features_path)
