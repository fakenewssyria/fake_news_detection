import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.linear_model import LinearRegression


class ExploratoryAnalysis():
    
    def __init__(self, features_file):
        self.features_file = features_file
    
    def split_data(self, train_data_file, test_data_file):
        
        ''' function that splits our dataset into 80% for training and 20% for testing '''
        
        data = pd.read_csv(self.features_file, encoding='latin-1')

        features = np.array(data.iloc[:, 0:-1])
        labels = np.array(data["label"])
        
        X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.20)
        
        train_df = pd.DataFrame(X_train)
        train_df["label"] = Y_train
        train_df.to_csv(train_data_file, encoding="latin-1")  # save the training dataset
        
        test_df = pd.DataFrame(X_test)
        test_df["label"] = Y_test
        test_df.to_csv(test_data_file, encoding="latin-1")  # save the testing dataset

    def perform_analysis(self, train_data_file):
        
        ''' function that performs exploratory analysis '''
        
        df_train = pd.read_csv(train_data_file, encoding='latin-1')  # read training dataset
        
#         self.scatter_plots(df_train)
        self.fake_true_percentages(df_train, "big_scale_event")

        sources = ['ahram', 'alaraby', 'arabiya', 'asharqalawsat', 'dailysabah', 'etilaf', 'jordantimes', 'nna', 'trt',
                   'alalam', 'manar', 'sana', 'sputnik', 'tass', 'reuters']
        
        colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'r', 'r', 'r', 'r', 'g']
        self.fake_true_percentages(df_train, "source", sources, colors)
        
        categories = ['against', 'pro' , 'neutral']
        self.fake_true_percentages(df_train, "source_category", categories)

        df_fake = df_train.loc[df_train["label"] == 0]  # articles labeled as fake
        df_true = df_train.loc[df_train["label"] == 1]  # articles labeled as true        
        
        self.plot_label_stats(df_train)
        self.plot_feature_stats(df_fake, "fake")
        self.plot_feature_stats(df_true, "true")
    
    def plot_label_stats(self, df):
        
        labels = {"true": (np.count_nonzero(np.array(df["label"]))) / len(df["label"]) * 100,
                  "fake": (len(df["label"]) - np.count_nonzero(np.array(df["label"]))) / len(df["label"]) * 100}
        
        self.plot_figure(labels, "label", "% of articles", "true_fake_dist", colors=['r', 'g'])
        
    def fake_true_percentages(self, train_df, column, categories=None, colors=['b', 'r', 'g']):
        
        ''' function that calculates % true and % fake for a given column in the entire train dataset'''
        
        if categories == None:
            categories = np.unique(train_df[column])  # get unique values
        
        nb_total_per_category = {}  # dictionary of category:nb articles
        nb_fake_per_category = {}  # dictionary of category:nb fake articles
        nb_true_per_category = {}  # dictionary of category:nb true articles
        
        for category in categories:
            if category != "none":
                nb_total_per_category[category] = 0
                nb_fake_per_category[category] = 0
                nb_true_per_category[category] = 0
        
        for _, row in train_df.iterrows():
            
            if row[column] != "none":
                nb_total_per_category[row[column]] += 1
                if row["label"] == 0:  # fake
                    nb_fake_per_category[row[column]] += 1
                elif row["label"] == 1:  # true
                    nb_true_per_category[row[column]] += 1    
        
        for category in categories:
            if category != "none":
                nb_fake_per_category[category] = (nb_fake_per_category[category] / nb_total_per_category[category]) * 100 
                nb_true_per_category[category] = (nb_true_per_category[category] / nb_total_per_category[category]) * 100
        
        self.plot_figure(nb_total_per_category, column, "nb articles in our dataset from this " + column, column + "_total.png", colors)
        self.plot_figure(nb_true_per_category, column, "% of this " + column + "'s articles that was labeled true", column + "_true.jpg", colors)
        self.plot_figure(nb_fake_per_category, column, "% of this " + column + "'s articles that was labeled fake", column + "_fake.jpg", colors)
           
    def plot_figure(self, array_to_plot, xlabel, ylabel, title, colors):
                 
        plt.figure(figsize=(10, 12))    
        plt.bar(range(len(array_to_plot)), list(array_to_plot.values()), align='center', color=colors)
        plt.xticks(range(len(array_to_plot)), list(array_to_plot.keys()))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if xlabel == "source_category" or xlabel == "source":  # color columns based on source's category
            against = mpatches.Patch(color='blue', label='against')
            pro = mpatches.Patch(color='red', label='pro')
            neutral = mpatches.Patch(color='green', label='neutral')
      
            plt.legend(handles=[against, pro, neutral])
             
        plt.savefig(title)      
           
        plt.figure()    
        
    def scatter_plots(self, df):
        
        ''' function that plots the scatter plots of each pair of features '''
        
#         df.loc[df['label'] == 0, ['label']] = "fake"
#         df.loc[df['label'] == 1, ['label']] = "true"
        
        corr = df.drop(columns=['unit_id']).corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        # Set up the matplotlib figure
        f, ax = plt.subplots()
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
#         plt.setp(ax.get_xticklabels(), ha="right")
#         plt.setp(ax.get_yticklabels(), ha="right")

        plt.savefig("scatter_plot.png")
        
#         
# #         column_pairs = list(product(df.columns, df.columns))
#         column_pairs = [('sectarian_language', 'quoted_sources'),
#                         ('sectarian_language', 'bias'),
#                         ('sectarian_language', 'factive_verbs'),
#                         ('sectarian_language', 'implicative_verbs'),
#                         ('sectarian_language', 'hedges'),
#                         ('sectarian_language', 'report_verbs'),
#                         ('sectarian_language', 'assertive_verbs'),
#                         ('sectarian_language', 'consistency_score'),
#                         ('quoted_sources', 'sectarian_language'),
#                         ('quoted_sources', 'bias'),
#                         ('quoted_sources', 'factive_verbs'),
#                         ('quoted_sources', 'implicative_verbs'),
#                         ('quoted_sources', 'hedges'), ('quoted_sources', 'report_verbs'),
#                         ('quoted_sources', 'assertive_verbs'), ('quoted_sources', 'consistency_score'),
#                         ('bias', 'sectarian_language'), ('bias', 'quoted_sources'), ('bias', 'factive_verbs'), ('bias', 'implicative_verbs'), ('bias', 'hedges'), ('bias', 'report_verbs'), ('bias', 'assertive_verbs'), ('bias', 'consistency_score'),
#                         ('factive_verbs', 'sectarian_language'), ('factive_verbs', 'quoted_sources'), ('factive_verbs', 'bias'), ('factive_verbs', 'implicative_verbs'), ('factive_verbs', 'hedges'), ('factive_verbs', 'report_verbs'), ('factive_verbs', 'assertive_verbs'), ('factive_verbs', 'consistency_score'),
#                         ('implicative_verbs', 'sectarian_language'), ('implicative_verbs', 'quoted_sources'), ('implicative_verbs', 'bias'), ('implicative_verbs', 'factive_verbs'), ('implicative_verbs', 'hedges'), ('implicative_verbs', 'report_verbs'), ('implicative_verbs', 'assertive_verbs'), ('implicative_verbs', 'consistency_score'),
#                          ('hedges', 'sectarian_language'), ('hedges', 'quoted_sources'), ('hedges', 'bias'), ('hedges', 'factive_verbs'), ('hedges', 'implicative_verbs'), ('hedges', 'report_verbs'), ('hedges', 'assertive_verbs'), ('hedges', 'consistency_score'),
#                          ('report_verbs', 'sectarian_language'), ('report_verbs', 'quoted_sources'), ('report_verbs', 'bias'), ('report_verbs', 'factive_verbs'), ('report_verbs', 'implicative_verbs'), ('report_verbs', 'hedges'), ('report_verbs', 'assertive_verbs'), ('report_verbs', 'consistency_score'),
#                          ('assertive_verbs', 'sectarian_language'), ('assertive_verbs', 'quoted_sources'), ('assertive_verbs', 'bias'), ('assertive_verbs', 'factive_verbs'), ('assertive_verbs', 'implicative_verbs'), ('assertive_verbs', 'hedges'), ('assertive_verbs', 'report_verbs'), ('assertive_verbs', 'consistency_score'),
#                          ('consistency_score', 'sectarian_language'), ('consistency_score', 'quoted_sources'), ('consistency_score', 'bias'), ('consistency_score', 'factive_verbs'), ('consistency_score', 'implicative_verbs'), ('consistency_score', 'hedges'), ('consistency_score', 'report_verbs'), ('consistency_score', 'assertive_verbs')
#                          ]
# 
#         print(column_pairs)
#         for pair in column_pairs:
#             sns.pairplot(df[[pair[0], pair[1], "label"]], hue="label")
#             plt.savefig("new_scatter_plots/%s_%s.png" % (pair[0], pair[1]))

    def plot_feature_stats(self, df, title):
        
        ''' function that performs exploratory analysis on the features in our dataset '''
        
        file = open("stats_for_" + title, 'w+')
        
        file.write("stats for " + title)
        file.write("total number of articles: %d" % len(df))
             
        ''' numerical features '''
            
        for col in ["quoted_sources"]:
            file.write("stats for " + col)
            file.write(str(df[col].describe()))
            plt.figure(figsize=(9, 8))
            
            cut1 = 0
            cut2 = 0
            cut3 = 0
            cut_off = 0.05
            for val in df[col]:
                if val == 0:
                    cut1 += 1
                elif val == 0.5:
                    cut2 += 1
                else:
                    cut3 += 1
                    
            cut1 /= len(df[col])
            cut2 /= len(df[col])
            
            cut1 *= 100
            cut2 *= 100
            print(cut1, cut2)
            
            min_val = df[col].min()
            max_val = df[col].max()
            
            str1 = "%.2f to %.2f" % (min_val, cut_off)
            str2 = "%.2f to %.2f" % (cut_off, max_val)
            
            plt.bar(["0", "0.5", "1"], [cut1, cut2, cut3], color='g')
            
            plt.xlabel(col)
            plt.ylabel('% ' + title + ' articles')
            
            plt.ylim(0, 100)
            plt.grid(None)
            plt.savefig("output/exploratory_analysis_plots/" + col + "_" + title + ".png")
        
        file.close()
        
    def plot_learning_curve(self, training_data_file, learning_curve_file):
    
        '''read data from file'''
        
        data = pd.read_csv(training_data_file, encoding='latin-1') 
        
        Y = np.array(data["label"]) 
        X = np.array(data.iloc[:, 6:10])  

        '''fit a linear regression classifier to the training data'''
        
        regr = LinearRegression()
        regr = regr.fit(X, Y)
        
        '''plot the learning curve'''
        
        # set the cross-validation factor to be used in evaluating learning curve
        cv = ShuffleSplit(n_splits=len(Y), test_size=0.2)  # 20% for validation
        
        train_sizes, train_scores, test_scores = learning_curve(regr, X, Y, cv=cv)  # calculate learning curve values
        
        # mean of the results of the training and testing
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
         
        '''plot results'''
        
        plt.figure()
        plt.xlabel("Number of Training Points")
        plt.ylabel("Error")
        
        plt.plot(train_sizes, train_scores_mean, color="r", label="Ein")
        plt.plot(train_sizes, test_scores_mean, color="g", label="Eval")
        
        frame = plt.gca()
#         frame.axes.yaxis.set_ticklabels([])
        frame.invert_yaxis()
        
        plt.legend(loc="best")
        plt.savefig(learning_curve_file, bbox_inches='tight')
