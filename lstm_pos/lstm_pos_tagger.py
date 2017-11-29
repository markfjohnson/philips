# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:24:46 2017

@author: vinodkumar
"""

import numpy as np
from keras.layers.merge import concatenate
from keras.models import Sequential,load_model, Model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Activation,Input,merge, Dropout 
import cPickle as pickle
import dataPreprocessing as dataPreprtn
import os
import argparse, pkg_resources

#DATA_PATH = pkg_resources.resource_filename('nlpservices', 'data/posLstm2/')
DATA_PATH="."
DATA_PATH = os.getcwd()+"/"
print DATA_PATH
class Configuration(object):
    """
    It defines the configuration settings required for lstm pos tagger like
    - model_file, embedding_size, num_steps, batch_size
    """
    
    def __init__(self):

        self.save_path = DATA_PATH
        
        # The model path
        self.model_file = "posKerasModel.h5"

        # Batch size
        self.batch_size = 32
        
        #embedding_size
        self.embedding_size = 300
        
        # LSTM time steps (Unrolled LSTM size)
        self.num_steps = 30 # average size of words in a sentences
    
class LstmTagger(object):
    """
    Class that applies lstm method that does pos tagging
    """
    
    def __init__(self, train_file, test_file,config):
        self.train_file = train_file
        self.test_file = test_file
        self.config_values = config
        
    def get_model_sequential(self,input_vocab_size, input_length, output_size):
        """
        Initialize the sequential keras model for lstm
        """
        model = Sequential()
        encoding_size = self.config_values.embedding_size
        model.add(Embedding(input_vocab_size, encoding_size, input_length=input_length)) #, batch_input_shape=(BATCH_SIZE, input_length)))
    
        model.add(LSTM(128, return_sequences=True, stateful=False, unroll=True))
        model.add(LSTM(32, return_sequences=True, stateful=False, unroll=True))
        model.add(TimeDistributed(Dense(output_size, activation='softmax')))
    
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def get_model(self, input_vocab_size, input_length, output_size):
        """
        Initialize the keras model based on the Input size for lstm
        """
        
        encoding_size = self.config_values.embedding_size
        
        embedding_input = Input(shape=(input_length,))
        model_embedding = Embedding(input_vocab_size, encoding_size, input_length=input_length)(embedding_input) #, batch_input_shape=(BATCH_SIZE, input_length)))
        #print(model_embedding.output_shape)
    
        features_input = Input((input_length, 1))
        #print(model_features.output_shape)
    
        merged =  concatenate([model_embedding, features_input])
        
        
        lstm  = LSTM(128, return_sequences=True, stateful=False)(merged)
        lstm = TimeDistributed(Dense(output_size, activation='softmax'))(lstm)
        
        model = Model(inputs=[embedding_input, features_input], outputs=[lstm])
    
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    
    def encode_output(self, ys, tag2id):
        """
        Perform one hot encoding
        """
        Y = np.zeros((len(ys), self.config_values.num_steps, len(tag2id)))
        for i, y in enumerate(ys):
            for j, o in enumerate(y):
                Y[i, j, o] = 1
        return Y
        
def train(train_file):
    """
    Train the lstm model 
    """
    
    config = Configuration()
    tagger = LstmTagger(train_file = train_file, test_file = None, config = config)
    
    preObj = dataPreprtn.DataReader(train_file=train_file, test_file=None,
                                    max_words=config.num_steps, split = 0.15, 
                                    batch_size = config.batch_size)
    preObj.prepare_vocabs()
    X,X_f, y = preObj.get_data(preObj.train_file, textformat="delimiter" , delimiter="/")
    Y = tagger.encode_output(y, preObj.tag2id)
    
    shuffle = np.random.permutation(len(X))
    X, Y = X[shuffle], Y[shuffle]
    X_f = X_f[shuffle]
    
    split = int(len(X)* (1-preObj.split))
    train_X = X[0:split]
    train_Y = Y[0:split]
    train_X_f = X_f[0:split]
    
    val_X = X[split:]
    val_Y = Y[split:]
    val_X_f = X_f[split:]
    
    model = tagger.get_model(len(preObj.word2id), preObj.max_words, len(preObj.tag2id))
    print (model.summary())
    
    train_X_f = train_X_f.reshape(len(train_X_f), -1, 1)
    val_X_f = val_X_f.reshape(len(val_X_f), -1, 1)
    
    modelName = config.model_file

    model.fit([train_X, train_X_f], train_Y, preObj.batch_size, epochs=3, 
              validation_data=([val_X, val_X_f], val_Y), shuffle=False)    
    model.save(config.save_path +  modelName)
    
    word2id_fileName = config.save_path +  "word2id.pkl"
    with open(word2id_fileName, 'wb') as pklFp:
        pickle.dump(preObj.word2id, pklFp, protocol=pickle.HIGHEST_PROTOCOL)
    
    postags = config.save_path +  "posTags.pkl"
    with open(postags,'wb') as pklTg:
        pickle.dump(preObj.id2tag, pklTg, protocol=pickle.HIGHEST_PROTOCOL)
        
#    text = "This is an input to pos tagger. Check for correct tags"
#    tagger_test = LstmTagger(train_file = None, test_file = text, config = config)
#    preObj_test = dataPreprtn.DataReader( test_file=text,
#                                    max_words=config.num_steps, split = 0.15, 
#                                    batch_size = config.batch_size)
#    X_test,X_f_test, y_test = preObj_test.test_data(textformat="text", delimiter=None)
#    X_f_test = X_f_test.reshape(len(X_f), -1, 1)
#    Y = tagger_test.encode_output(y_test, preObj_test.tag2id)
#    pmodel = load_model(config.save_path+  config.model_file)
#    Y_hat = pmodel.predict([X, X_f])
#    resultList = preObj_test.get_test_labels(X,Y_hat)
#    print resultList

def test(text_array, vocabFile=DATA_PATH + "word2id.pkl"):
    """
    Function that takes the given text and predicts the pos_tag for given text
    Parameters:
        text_array : The given text
        vocabFile = vocabulary file that has word2id mapping.
    Return:
         List : It contains Pos_tagging for the given text
    """
    
    config = Configuration()
    print config.save_path
    tagger = LstmTagger(train_file = None, test_file = text_array, config = config)
    preObj = dataPreprtn.DataReader( test_file=text_array,
                                    max_words=config.num_steps, split = 0.15, 
                                    batch_size = config.batch_size)
    #Word2id and id2ord conversions
    with open(vocabFile, 'rb') as pklFp:
        preObj.word2id = pickle.load(pklFp)
    
    #preObj.word2id["tagger"] = len(preObj.word2id)
    #preObj.word2id["check"] = len(preObj.word2id)
    
    preObj.id2word = [0] * len(preObj.word2id)
    for word, i in preObj.word2id.iteritems():
        preObj.id2word[i] = word
        
    X,X_f, y = preObj.test_data(textformat="text", delimiter=None)
    #Y = tagger.encode_output(y, preObj.tag2id)
    print config.save_path +  config.model_file
    pmodel = load_model(config.save_path+  config.model_file)
    X_f = X_f.reshape(len(X_f), -1, 1)
    Y_hat = pmodel.predict([X, X_f])
    
    #id2tag and tag2id conversions
    id2tag = DATA_PATH + "posTags.pkl"
    with open(id2tag, "rb") as pklTg:
        preObj.id2tag = pickle.load(pklTg)
    resultList = preObj.get_test_labels(X,Y_hat)
    #print resultList
    return resultList
    
def predict_labels(val_X, val_Y, val_Y_hat, id2word, id2tag):
    """
    Write the predict labels to the file test_predicted_labels.txt during trainig.
    """
    x_shape = val_X.shape
    outfile = "test_predicted_labels.txt"
    orig_labels = np.argmax(val_Y,axis=2)
    predict_labels = np.argmax(val_Y_hat, axis=2)
    with open(outfile,"w") as op:
        op.write('"TEXT", "TAG", "PREDICTED TAG"' + "\n")
        for i in xrange(0,x_shape[0]):
            for j in xrange(0, x_shape[1]):
                new_str = id2word[val_X[i,j]] + ", " + id2tag[orig_labels[i,j]] + ", " + id2tag[predict_labels[i,j]]
                op.write(new_str)
                op.write("\n")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", help="train or test")
    parser.add_argument("--train_file", help = "Enter the file name for training")
    parser.add_argument("--test_array", help = "Enter the filename for testing")
    parser.add_argument("--vocab_file", help = "Enter teh word2id file")
    
    args = parser.parse_args()
    if (args.operation).lower() == "train":
        args.test_file=None
        train(args.train_file)
    elif (args.operation).lower() == "test":
        args.train_file =None
        y_hat = test(args.test_array, args.vocab_file)
    
