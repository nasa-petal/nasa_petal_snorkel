import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(load_train_labels: bool = False, split_dev_valid: bool = False):
    filename = r'labeled_data_small.csv'
    df = pd.read_csv(filename)
    #lowercase column names
    df.columns = map(str.lower, df.columns)
    #comnine title and abstract columns into one and drop columns
    df['text'] = df['title'] + ' ' + df['abstract']
    df = df.drop(['title', 'abstract'], axis=1)
    
    df_train = df.sample(frac=0.6, random_state=123).reset_index(drop=True)
    df_dev = df_train.sample(frac=0.2, random_state=123)

    if not load_train_labels:
        df_train['label'] = np.ones(len(df_train['label'])) * -1
    
    df_valid_test = df_train.sample(frac=0.5, random_state=123)
    df_valid, df_test = train_test_split(df_valid_test, random_state=123, stratify=df_valid_test.label)

    if split_dev_valid:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_test
    

    


