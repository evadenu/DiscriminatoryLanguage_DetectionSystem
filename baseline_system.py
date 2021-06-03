'''
Random baseline 
(based on the code submission Text Mining Domains
Van der Ende, Buckens and Den Uijl 2020-2021 VU Amsterdam)
'''

import argparse
import sys
import os
import pandas as pd
import random

random.seed(3)

def main():
    
    #checking paths for arguments 
    test = pd.read_pickle('data/test_data.pkl')
    output_path = "data/predictions_baseline.csv"
    
    
    #predicting random label class for all instances
    labels = [0, 1, 2]
    
    predictions = []
    for _ in range(len(test)):
        predictions.append(random.choice(labels))
    
    test['prediction'] = predictions
    test.to_csv(output_path, sep = ',', index = False)

if __name__ == '__main__':
    main()