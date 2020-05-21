import numpy as np


class ConsistencyScore:
    
    def calculate_consistency_score(self, clustering_output):
        self.clustering_output = clustering_output
        
        # comment the following two lines if testing a dataset other than FA-KES
        self.set_source_category()
        self.init_arrays()
        
        self.consistency_score_array = self.calculate_score()
        
    def set_source_category(self):
        ''' set the category of each source to compute consistency score based on categories'''
                
        source_to_cat_dict = {"arabiya": "against",
                              "jordantimes": "against",
                              "ahram": "against",
                              "asharqalawsat": "against",
                              "nna": "against",
                              "sana": "pro",
                              "alalam": "pro",
                              "manar": "pro",
                              "sputnik": "pro",
                              "tass": "pro",
                              "reuters": "neutral",
                              "etilaf": "against",
                              "alaraby": "against",
                              "trt": "against",
                              "dailysabah": "against"}
        source_categories = []

        for _, article in self.clustering_output.iterrows():
            source_categories.append(source_to_cat_dict.get(article["source"]))
        
        self.clustering_output["source_category"] = np.array(source_categories)    

    def init_arrays(self):
        
        ''' get the min, max, and ranges arrays '''
            
        self.max_of_distances = np.zeros(6)
        self.min_of_distances = np.zeros(6)
        self.ranges_of_distances = np.zeros(6)
        
        ''' max of values '''
        self.max_of_distances[0] = self.clustering_output["nb_civilians"].max()
        self.max_of_distances[1] = self.clustering_output["nb_children"].max()
        self.max_of_distances[2] = self.clustering_output["nb_women"].max()
        self.max_of_distances[3] = self.clustering_output["nb_noncivilians"].max()
        self.max_of_distances[4] = 1
        self.max_of_distances[5] = 1
        
        ''' mins of values '''
        self.min_of_distances[0] = self.clustering_output["nb_civilians"].min()
        self.min_of_distances[1] = self.clustering_output["nb_children"].min()
        self.min_of_distances[2] = self.clustering_output["nb_women"].min()
        self.min_of_distances[3] = self.clustering_output["nb_noncivilians"].min()
        self.min_of_distances[4] = -1
        self.min_of_distances[5] = -1
        
        ''' ranges of values '''
        
        for i in range(len(self.min_of_distances)):
                
            if np.isnan(self.max_of_distances[i]):
                self.max_of_distances[i] = 0
                    
            if np.isnan(self.min_of_distances[i]):
                self.min_of_distances[i] = 0
                    
            if self.max_of_distances[i] != 0:
                self.ranges_of_distances[i] = 1 - self.min_of_distances[i] / self.max_of_distances[i]
            else:
                self.ranges_of_distances[i] = 0
                
    def get_distances_array(self, article_features, relevant_article):

        distances = np.zeros(6)
            
        distances[0] = abs(article_features["nb_civilians"] - relevant_article["nb_civilians"]) 
        
        distances[1] = abs(article_features["nb_children"] - relevant_article["nb_children"]) 
        
        distances[2] = abs(article_features["nb_women"] - relevant_article["nb_women"]) 
        
        distances[3] = abs(article_features["nb_noncivilians"] - relevant_article["nb_noncivilians"]) 
        
        if article_features["actor"] != article_features["actor"]:
            distances[4] = 1
        else:
            distances[4] = 0
    
        if article_features["cause_of_death"] != article_features["cause_of_death"]:
            distances[5] = 1
        else:
            distances[5] = 0
        
        return distances
    
    ''' gower distance function '''
    
    def get_gower_distance(self, article_features, relevant_article):
        
        distances_array = self.get_distances_array(article_features, relevant_article)
        
        sum_sij = 0.0
                    
        for col in range(len(distances_array)):
                    
            value_xi = distances_array[col]
    
            if col < 4:
                
                if self.max_of_distances[col] != 0:
                    value_xi = value_xi / self.max_of_distances[col]
                else:
                    value_xi = 0
    
                if self.ranges_of_distances[col] != 0:
                    sij = value_xi / self.ranges_of_distances[col]
                else:
                    sij = 0
            
            else:
                sij = value_xi
    
            sum_sij += sij
                
        return sum_sij
    
    def calculate_score(self):

        return -1 
        # consistency_score = np.zeros(len(self.clustering_output))
            
        # for index, article_features in self.clustering_output.iterrows():
            
        #     # get all other articles that report this same event
        #     relevant_articles = self.clustering_output.loc[(self.clustering_output.month == article_features["month"])
        #                                     & (self.clustering_output.year == article_features["year"])
        #                                     & (self.clustering_output.source_category == article_features["source_category"])
        #                                       & (self.clustering_output.article_content != article_features["article_content"])]
            
        #     if len(relevant_articles) == 0:  # article was not matched with any other article 
        #         consistency_score[index] = -1
        #     else:
        #         for _, relevant_article in relevant_articles.iterrows():
        #             consistency_score[index] += self.get_gower_distance(article_features, relevant_article)
            
        #     if len(relevant_articles) != 0:
        #         consistency_score[index] /= len(relevant_article)
                
        # return consistency_score
    
