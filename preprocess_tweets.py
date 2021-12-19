import pandas as pd
from pandas.core.frame import DataFrame
import re

def read_annotated_tweets(input_filename: str):
    df = pd.read_csv(input_filename)
    df.rename(columns={'data':'tweet_text'}, inplace=True)
    #remove leading trailing withspaces, pandas alignment problem when printing df to file
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def clean_tweets(df: DataFrame):
    #replace mentions with "USERNAME"
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'(@\w*)','USERNAME', str(x)))
    #remove hashtags
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'#\w+\s*','', str(x)))
    #remove url-s
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)','', str(x)))
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'\b(http[s]?)*','', str(x)))
    #remove special chars (emojis)
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'\W',' ', str(x)))
    #to lowercase
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: str(x).lower())
    #replace multiple spaces with one
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
    #trim
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: str(x).strip())


def main():
    df = read_annotated_tweets('./data/classified/england_italy_tweets_classified.csv')
    clean_tweets(df)

    print(df)
    

if __name__ == "__main__":
    main()

