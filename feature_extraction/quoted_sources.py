from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordNERTagger
from nltk.stem.wordnet import WordNetLemmatizer

import json

class QuotedSourcesFeature:
    
    def __init__(self, path_to_parsers):
        
        self.dependency_parser = StanfordDependencyParser('%s/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar' % path_to_parsers,
                                                          '%s/stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar' % path_to_parsers)
        self.ner_tagger = StanfordNERTagger('%s/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz' % path_to_parsers,
                       '%s/stanford-ner-2018-02-27/stanford-ner.jar' % path_to_parsers,
                       encoding='utf-8')

        ''' set label values '''
        
        self.NO_SOURCES = 0  # no sources
        self.NO_REAL_ATTRIBUTION = 1  # unnamed source is no real attribution
        self.REAL_ATTRIBUTION = 2  # real attribution is named the quoted organization, activist, or source
        
    def set_description_of_sources_quoted(self, articles, report_verbs_lexicon):
        
        ''' function that sets the description of sources quoted feature for all articles '''
        
        self.report_verbs_lexicon = report_verbs_lexicon
        quoted_source_labels = []

        for i in range(len(articles)):
            
            print(i)
            article = articles[i]
            label = -1
            try:
                subjects = self.get_report_verbs_subjects(article)  # extract the subject of the report verbs
                label = self.get_subjects_label(subjects)  # label the subject of the report verb
            except:   
                print("an error occurred in parsing")
            quoted_source_labels.append(label)

        self.quoted_source_labels = quoted_source_labels  # return the array of labels of all the articles
    
    def get_report_verbs_subjects(self, article):
        
        ''' function that gets the report verb subjects for one article '''
        
        report_verbs_subjects = []
        lemmatizer = WordNetLemmatizer()
        
        article_content = article.replace("<br>", ".")
        article_sentences = article_content.split(".")  # end of sentence
            
        for sentence in article_sentences:
                
            if sentence == "":  # skip empty sentences in split array
                continue
                
            sent = [lemmatizer.lemmatize(x, 'v') for x in sentence.split(' ')]  # get verb lemma to find report verbs
            sentence = " ".join(sent)

            for verb in self.report_verbs_lexicon:
                if verb in sentence.lower():
                    dependencies = self.dependency_parser.raw_parse(sentence)  # build dependencies tree 
                    dependencies = dependencies.__next__()
                    subjects = self.traverse(dependencies, 0, verb)  # traverse dependency tree looking for the report verb
                    if subjects != []:
                        report_verbs_subjects.extend(subjects)  # add to list of report verbs subjects
                            
        return report_verbs_subjects
    
    def traverse(self, deps, addr, verb):
        
        ''' function that traverses the dependency tree '''
        
        subjects = []  # list of subjects of this verb
        
        dep = deps.get_by_address(addr)
        
        if dep["word"] == verb:
                s = str(dep['deps'])[len("defaultdict(<class 'list'>, "):-1].replace("'", "\"")
                if s != "{}":
                    d = json.loads(s)
                    if "nsubj" in d:  # get indices of subjects of the verb from the tree
                        for subj_index in d["nsubj"]:
                            subjects.append(deps.get_by_address(subj_index)["word"])  # get the subject from its index from the tree
        
        for d in dep['deps']:
            for addr2 in dep['deps'][d]:
                sub_subjects = self.traverse(deps, addr2, verb)  # recursively traverse the tree to make sure we find the subjects of the report verb
                subjects.extend(sub_subjects)  # append the subjects returned by the recursive call
                
        return subjects  # return all possible subjects of this report verb
    
    def get_subjects_label(self, subjects):
        
        if len(subjects) == 0:  # no sources quoted
            label = self.NO_SOURCES 
        else:
            label = self.NO_REAL_ATTRIBUTION  # sources quoted but not person or organization

        for subject in subjects:
            
            tagged_subject = self.ner_tagger.tag([subject])
            tag = tagged_subject[0][1]  # organization, person, location, or other                
            if tag == "ORGANIZATION" or tag == "PERSON":  # quoted a person or an organization
                label = self.REAL_ATTRIBUTION
        
        return label
