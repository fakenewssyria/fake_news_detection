import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize


class LexiconFeatures:
    
    ''' read the lexicons from their respective files '''
    
    def read_assertive_lexicon(self, assertive_lexicon_path):
        ''' Assertive Verbs Lexicon '''
        self.assertive_lexicon = [line.rstrip('\n') for line in open(assertive_lexicon_path)]

    def read_factive_lexicon(self, factive_lexicon_path):
        ''' Factive Verbs Lexicon '''
        self.factive_lexicon = [line.rstrip('\n') for line in open(factive_lexicon_path)]

    def read_implicative_lexicon(self, implicative_lexicon_path):
        ''' Implicative Verbs Lexicon '''
        self.implicative_lexicon = [line.rstrip('\n') for line in open(implicative_lexicon_path)]

    def read_hedges_lexicon(self, hedges_lexicon_path):
        ''' Hedges Lexicon '''
        self.hedges_lexicon = [line.rstrip('\n') for line in open(hedges_lexicon_path)]

    def read_sectarian_lexicon(self, sectarian_lexicon_path):
        ''' Sectarian Language Lexicon '''
        self.sectarian_lexicon = [line.rstrip('\n') for line in open(sectarian_lexicon_path)]

    def read_report_lexicon(self, report_lexicon_path):
        self.report_lexicon = [line.rstrip('\n') for line in open(report_lexicon_path)]

    def read_bias_subj_lexicon(self, bias_lexicon_path, subjclues_lexicon_path, negative_lexicon_path, positive_lexicon_path):
        
        ''' Read bias, subjectivity, positive words, negative words into one lexicon '''
        self.bias_subj_lexicon = [line.rstrip('\n') for line in open(bias_lexicon_path)]
        self.bias_subj_lexicon.extend([line.rstrip('\n') for line in open(negative_lexicon_path, encoding='latin-1')])
        self.bias_subj_lexicon.extend([line.rstrip('\n') for line in open(positive_lexicon_path, encoding='latin-1')])
        self.bias_subj_lexicon.extend([line.rstrip('\n') for line in open(subjclues_lexicon_path)])
        
        self.bias_subj_lexicon = np.unique(self.bias_subj_lexicon)
    
    ''' calculate the lexicon features '''

    def calculate_occurrence(self, article_content, lexicon, frequency=True):
        
        ''' Calculates the occurrence of the words of a lexicon in the article content'''

        freq = 0
        
        article_content = word_tokenize(article_content)  # split article into words
        while "" in article_content: article_content.remove("")  # remove empty strings
        
        article_content = [w for w in article_content if not w in self.stop_words]  # filter out the stop words from the article

        article_content = [self.stemmer.stem(w) for w in article_content]  # stem each word
         
        for word, tag in pos_tag(article_content):

            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            
            if not wntag:
                lemma = word
            else:
                lemma = self.lemmatizer.lemmatize(word, wntag)  # lemmatize each word

            if lemma.lower() in lexicon:  # if this word is in this lexicon, increment the count
                freq += 1
        
        if frequency:
            freq /= len(article_content)
            
        return freq

    def calculate_lexicons_features(self, articles):
        
        self.stemmer = SnowballStemmer("english")         
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.assertive_frequency = np.zeros(len(articles))
        self.factive_frequency = np.zeros(len(articles))
        self.implicative_frequency = np.zeros(len(articles))
        self.hedges_frequency = np.zeros(len(articles))
        self.sectarian_frequency = np.zeros(len(articles))
        self.report_frequency = np.zeros(len(articles))
        self.bias_frequency = np.zeros(len(articles))
        
        for i in range(len(articles)):
            
            article_content = articles[i].replace(".", " ")  # prepare for split
            
            # get frequency of related work features 
            
            self.assertive_frequency[i] = self.calculate_occurrence(article_content, self.assertive_lexicon)
            self.factive_frequency[i] = self.calculate_occurrence(article_content, self.factive_lexicon)
            self.implicative_frequency[i] = self.calculate_occurrence(article_content, self.implicative_lexicon)
            self.hedges_frequency[i] = self.calculate_occurrence(article_content, self.hedges_lexicon)
            self.report_frequency[i] = self.calculate_occurrence(article_content, self.report_lexicon)
            self.bias_frequency[i] = self.calculate_occurrence(article_content, self.bias_subj_lexicon)
            
            self.sectarian_frequency[i] = self.calculate_occurrence(article_content, self.sectarian_lexicon)
