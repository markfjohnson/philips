# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:43:50 2017

@author: vinodkumar
"""

import InputReader as Reader
import numpy as np

class DataReader(object):
    """
    DataReader class file to read and convert the given data into required format and generate
    word2id and id2word features.
    """
    
    def __init__(self, train_file="GENIAcorpus3.02.pos.txt", test_file=None, max_words=30, split=.15, batch_size=32):
        
        self.train_file = train_file
        self.test_file = test_file
        self.max_words = max_words
        self.split = split
        self.batch_size = batch_size
        
        # maps
        self.word2id = dict()
        self.id2word = []
        self.tag2id=dict()
        self.id2tag=[]
        
    def load_tagged_sents(self, filename, textformat, delimiter=None):
        """
        Load the tagged sents from the train file.
        Parameters:
            filename: train file
            textformat : The text is present with/without delimiter.
            delimiter : delimiter character
        """
        tagged_sents = Reader.DataReader().formatConv(filename, textformat, delimiter)
        return tagged_sents
        
    def prepare_vocabs(self):
        """
        Prepare word2id, tag2id and id2word, id2tag files
        """
        #tagged_sents = addStartEndTags(tagged_sents)
        tagged_sents = self.load_tagged_sents(self.train_file, textformat = "delimiter", delimiter = "/")
        self.word2id["DUMMY_WORD"] = 0
        next_word_key = 1
        
        self.word2id["UNK"] = next_word_key
        next_word_key = next_word_key + 1
    
        self.tag2id["DUMMY_TAG"] = 0
        next_tag_key = 1
    
        for sent in tagged_sents:
            for (word, tag) in sent:
                word = word.encode('utf-8').lower()
    
                # insert word into dictionary
                if not self.word2id.has_key(word):
                    self.word2id[word] = next_word_key
                    next_word_key += 1
    
                # insert tag into dictionary
                if not self.tag2id.has_key(tag):
                    self.tag2id[tag] = next_tag_key
                    next_tag_key += 1
    
        print ("Total Unique Words: {0}, Total Tags: {1}".format(len(self.word2id), len(self.tag2id)))
    
        self.id2word = [0] * len(self.word2id)
        for word, i in self.word2id.iteritems():
            self.id2word[i] = word
    
        self.id2tag = [0] * len(self.tag2id)
        for tag, i in self.tag2id.iteritems():
            self.id2tag[i] = tag
            
            
    def get_data(self, filename, textformat, delimiter):
        """
        Convert the given text data into numeric form
        """
        tagged_sents = self.load_tagged_sents(filename, textformat, delimiter)
        n_sents = len(tagged_sents)
        # @todo: Add sent start and sent end tokens
    
        x = np.zeros((n_sents, self.max_words), dtype=np.int32)
        x_f = np.zeros((n_sents, self.max_words), dtype=np.int32)
        y = np.zeros((n_sents, self.max_words), dtype=np.int32)
    
        for i, sent in enumerate(tagged_sents):
            for j, (word, tag) in enumerate(sent):
                if j >= self.max_words:
                    break
                if word[0].isupper():
                    x_f[i, j] = 1
                word = word.encode('utf-8').lower()
                x[i, j] = self.word2id[word]
                y[i, j] = self.tag2id[tag]
    
        return x,x_f, y

    def test_data(self, textformat, delimiter):
        """
        Prepare the test data
        """
        #self.prepare_vocabs()
        #tagged_sents = self.load_tagged_sents(self.test_file, textformat, delimiter)
        print self.test_file
        print type(self.test_file)
        if type(self.test_file) == type(list()):
            tagged_sents = [self.test_file]
        else:
            tagged_sents = [map(str.strip, (self.test_file).strip('[]').split(','))]
            print tagged_sents
        n_sents = len(tagged_sents)

        x = np.zeros((n_sents, self.max_words), dtype=np.int32)
        x_f = np.zeros((n_sents, self.max_words), dtype=np.int32)
        y = np.zeros((n_sents, self.max_words), dtype=np.int32)
        
        for i, sent in enumerate(tagged_sents):
            for j, (word) in enumerate(sent):
                if j >= self.max_words:
                    break
                if word[0].isupper():
                    x_f[i, j] = 1
                word = word.encode('utf-8').lower()
                #assert self.id2word[x[i, j]] == word, "[{0},{1}] {2} {3}".format(i, j, self.id2word[x[i, j]], word)
                if self.word2id.has_key(word):
                    x[i, j] = self.word2id[word]
                else:
                    x[i,j] = self.word2id["UNK"]
        return x,x_f,y

    def get_test_labels(self, test_x, y_hat):
        x_shape = test_x.shape
        mainlist = list()
        predict_labels = np.argmax(y_hat, axis=2)
        for i in xrange(0,x_shape[0]):
            sentlist=list()
            for j in xrange(0, x_shape[1]):
                if self.id2word[test_x[i,j]] != "DUMMY_WORD":
                    sentlist.append(tuple([self.id2word[test_x[i,j]], self.id2tag[predict_labels[i,j]]]))
            mainlist.append(sentlist)
        return mainlist
