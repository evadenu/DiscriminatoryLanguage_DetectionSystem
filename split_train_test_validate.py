import pandas as pd
import pickle

def split_test_train_validation(filepath):
    """
    Function to split data into test, train and dev set.
    :param df: filepath to pd dataframe
    :returns: three different randomized datasets 
    
    """
    df = pd.read_pickle(filepath)
    df_permutated = df.sample(frac=1)

    train_size = 0.8
    train_end = int(len(df_permutated)*train_size)

    df_train = df_permutated[:train_end]
    df_test = df_permutated[train_end:]

    test_size = 0.5
    test_end = int(len(df_permutated)*test_size)
    df_validation = df_permutated[test_end:]
    
    return df_train, df_test, df_validation


def pickle_dump(output_path, data)

    with open(output_path, "wb") as outfile:
        pickle.dump(data, outfile)
        

def main():
    
    split_test_train_validation('data/discrimination_labels.pkl')
    pickle_dump('data/training_data.pkl', df_train)
    pickle_dump('data/test_data.pkl', df_test)
    pickle_dump('data/validation_data.pkl', df_validation)

if __name__ == '__main__':
    main()
        
