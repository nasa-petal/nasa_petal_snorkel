# import glob
# import os
# import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(load_train_labels: bool = False, split_dev_valid: bool = False):
    filename = r"C:\Users\ARalevski\Documents\petal_snorkel\David work\labeled_data.csv"
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

    # if os.path.basename(os.getcwd()) == "snorkel-tutorials":
    #     os.chdir("spam")
    # try:
    #     subprocess.run(["bash", "download_data.sh"], check=True, stderr=subprocess.PIPE)
    # except subprocess.CalledProcessError as e:
    #     print(e.stderr.decode())
    #     raise e
    # filenames = sorted(glob.glob(r"C:\Users\ARalevski\Documents\Petal\Snorkel-PeTaL\youtube_spam_keyword_test.csv"))
    # df = pd.read_csv(filenames)
    # # dfs = []
    # # for i, filename in enumerate(filenames, start=1):
    # #     df = pd.read_csv(filename)
    # #     # Lowercase column names
    # #     # df.columns = map(str.lower, df.columns)
    # #     # Remove comment_id field
    # #     # df = df.drop("comment_id", axis=1)
    # #     # Add field indicating source video
    # #     # df["video"] = [i] * len(df)
    # #     # Rename fields
    # #     # df = df.rename(columns={"class": "label", "content": "text"})
    # #     # Shuffle order
    # #     df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # #     dfs.append(df)

    # df_train = df.sample(frac=0.6, random_state=123).reset_index(drop=True)
    # # print(df_train)
    # # df_train = pd.concat(dfs[:2])
    # df_dev = df_train.sample(frac=0.2, random_state=123)
    # # print(df_dev)

    # if not load_train_labels:
    #     df_train["label"] = np.ones(len(df_train["label"])) * -1
    # df_valid_test = df_train.sample(frac=0.5, random_state=123)
    # df_valid, df_test = train_test_split(
    #     df_valid_test, random_state=123, stratify=df_valid_test.label
    #     )

    # print(df_valid)
    # print(df_test)
    

    


