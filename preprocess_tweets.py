import nltk
import pandas as pd
from pandas.core.frame import DataFrame
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from feature_extraction import tokenize

wordnet_map = {
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "J": wordnet.ADJ,
    "R": wordnet.ADV
}

LEMMATIZER = WordNetLemmatizer()

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
    #remove standalone numbers
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'(?:^| )(\d+)(?:$| )','', str(x)))
    #remove string like "1st" "2nd" "1goal", "65mins"
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'(\d+[A-Za-z]+)','', str(x)))
    #replace multiple spaces with one
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: re.sub(r'\s+',' ', str(x)))
    #trim
    df['tweet_text'] =  df['tweet_text'].apply(lambda x: str(x).strip())
    #lemmatize
    df['tweet_text'] = df['tweet_text'].apply(lambda x: lemmatize_tweet(x))

def pos_tag_wordnet(text):
    pos_tagged_text = nltk.pos_tag(text)

    # map the pos tagging output with wordnet output
    pos_tagged_text = [
        (word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys()
        else (word, wordnet.NOUN)
        for (word, pos_tag) in pos_tagged_text
    ]

    return pos_tagged_text

def lemmatize_tweet(tweet_text: str):
    pos_tagged = pos_tag_wordnet(tokenize(tweet_text))

    result = []
    for x in pos_tagged:
        lemmatized_word = LEMMATIZER.lemmatize(x[0], x[1])
        result.append(lemmatized_word)

    return ' '.join(result)

def main():
    df = read_annotated_tweets('./data/classified/england_italy_tweets_classified1.csv')
    clean_tweets(df)

    print(df)


if __name__ == "__main__":
    main()

