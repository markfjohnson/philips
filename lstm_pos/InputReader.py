# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:09:38 2017

@author: vinodkumar
"""
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize


# Preprocessing code for pos_tagger code


class DataReader(object):
    """
    Convert the given text to required format based on the text format( plain text or delimited text)
    """
    
    def formatConv(self, inputFile, texformat = None, delimiter=None):
        if texformat == "delimiter":
            result = self.delimConv(inputFile, delimiter)
        else:
            result = self.textConv(inputFile)
        return result
    
    def delimConv(self, inputFile, delimiter = "/"):
        with open(inputFile) as fp:
            geniaData = fp.read().splitlines()
            
        convertedData = list()
        sentlist = list()
        
        for each_line in geniaData:
            wordTuple = tuple(each_line.rsplit(delimiter,1))
            if len(wordTuple) > 1:
                sentlist.append(wordTuple)
            elif len(wordTuple) == 1 and len(sentlist) > 0:
                #print(wordTuple)
                convertedData.append(sentlist)
                sentlist = list()
        return convertedData
    
    def textConv(self, inputFile):
        
        convertedData = list()
        
        tokenizeSent = sent_tokenize(inputFile)
        for each_sent in tokenizeSent:
            convertedData.append(word_tokenize(each_sent))
        
        return convertedData
            
        
        
    
    
    
