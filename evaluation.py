"""
Script to evaluate the predictions
(based on the code submission Text Mining Domains
Van der Ende, Buckens and Den Uijl 2020-2021 VU Amsterdam)
"""

import argparse
import sys
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def read_data(path):
    """
    Function that reads in the data and returns a dataframe. 
    :param path: path to data file
    :type path: string
    :return: pandas dataframe 
    """
    
    with open(path, encoding = 'utf-8') as infile:
        df = pd.read_csv(infile, delimiter=',')
    
        predictions = df['prediction']
        goldlabels = df['discrimination_label']
        
    return predictions, goldlabels

def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix
    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings
    :returns: confusion matrix
    '''

    # based on example from https://datatofish.com/confusion-matrix-python/
    data = {'Gold': goldlabels, 'Predicted': predictions}
    df = pd.DataFrame(data, columns=['Gold', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    
    print('----> CONFUSION MATRIX <----')
    print(confusion_matrix)
    return confusion_matrix


def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score in a complete report
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions: list
    :type goldlabels: list
    
    '''
    report = classification_report(goldlabels,predictions,digits = 3)
    
    print('----> CLASSIFICATION REPORT <----')
    print(report)

    
def main():
   
    predictions, goldlabels = read_data('data/predictions_baseline.csv')
    print_confusion_matrix(predictions, goldlabels)
    print_precision_recall_fscore(predictions, goldlabels)
    
    
if __name__ == '__main__':
    main()