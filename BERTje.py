'''
Based on code submission for TMD 2021 VU- Dyon van der Ende, Myrthe Buckens, Eva den Uijl
and https://github.com/cltl/ma-ml4nlp-labs/blob/main/code/assignment1/basic_system.ipynb
'''
import sys
import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, pipeline, AutoModel, TFAutoModel
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


#constants
MODEL_NAME = "GroNLP/bert-base-dutch-cased" #huggingface transformers model
MAX_LEN = 75                    #max token length of sentence
STEP_SIZE = 1                   #short sentence by STEP_SIZE until MAX_LEN tokens
                                #higher value is faster feature extraction
    
def extract_features(df, classifier):
    """
    Function to extract features from data.
    :param df: a pandas dataframe
    :returns: the selected features and the gold labels in the data
    
    """
    #initializing tokenizer DUTCH BERT model
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
    
    #creating lists for gold data and features
    gold = []
    features = []
     
    num_rows = df.shape[0]
    current_row = 1
    for index, row in df.iterrows():
        #print progress
        print(f'feature extraction: \t{str(current_row/num_rows*100)[:4]}%', end='\r')
        
        sentence = row['sentence']       

        #decrease sentence length until max number of tokens is reached
        while len(tokenizer.tokenize(sentence)) > MAX_LEN -2:
            sentence = sentence[:-STEP_SIZE]
        
        
        features = classifier(sentence)  #vectorize sentences by RoBERTa model
        sentence_len = len(features[0])
                                       
        vector = features[0]
        vector_length = len(vector[0])

        for i in range(sentence_len, MAX_LEN):
            vector.append(vector_length*[0])
        
        out = np.array(vector).flatten()

        assert len(vector) == MAX_LEN, "too much tokens"

        features.append(out)  
        gold.append(row['discrimination_label'])
        current_row+=1
        
    print()
   
    return np.array(features), np.array(gold)

def create_classifier(train_features, train_targets):
    """
    Function to create a classifier. Variable 'model' denotes type of classifier.
    
    :param train_features: list with features extracted from training data
    :param train_targets: list with gold labels from training data 
    :type train_features: list
    :type train_targets: list 
    :return: trained model and vectors for features 
    
    """
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    
    print("fitting model")
       
    model.fit(train_features, train_targets)

    return model     
         
        
def run_classifier(train_set, test_set):
    """
    Function to run the classifier and get the predicted labels. 
    
    :param train_set: training data 
    :param test_set: test data
    :type train_set: pandas dataframe
    :type test_set: pandas dataframe
    :return: predictions (list with predicted labels)
    
    """
    
    classifier = pipeline('feature-extraction', model=MODEL_NAME)
    
    print("train data")
    train_features, train_gold = extract_features(train_set, classifier)
    
    
    model = create_classifier(train_features, train_gold)
    
    #free up memory
    train_features = None
    
    print("test data")
    test_features, goldlabels = extract_features(test_set, classifier)
    
    predictions = model.predict(test_features)
    
    
    return predictions


def main():
    
    
    #reading in data 
    train_set = pd.read_pickle('data/validation_data.pkl')
    test_set = pd.read_pickle('data/test_mini.pkl')
    
    #running classifier and generating statistics on performance 
    predictions = run_classifier(train_set, test_set)
    
    #writing the predictions to a new file
    test['prediction'] = predictions
    test.to_csv('data/predictions.csv', sep = ',', index = False)
    

if __name__ == '__main__':
    main()
    
    
