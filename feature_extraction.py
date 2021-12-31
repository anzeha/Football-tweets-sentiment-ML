from typing import Iterable, Tuple
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction import text
from prepare_data import players_stopwords
import json
from nltk.tokenize import TweetTokenizer

def bow_occurrences_df(sentences: Iterable[str], ngram_range: Tuple[int, int]):
    cv = CountVectorizer(stop_words=stopwords(), ngram_range=ngram_range, strip_accents='unicode', tokenizer=tokenize)

    count_matrix = cv.fit_transform(sentences)

    count_array = count_matrix.toarray()

    df = pd.DataFrame(data=count_array,columns = cv.get_feature_names())

    return df

def bow_occurrences(sentences: Iterable[str], ngram_range: Tuple[int, int]):
    cv = CountVectorizer(   stop_words=stopwords(), 
                            ngram_range=ngram_range,
                            strip_accents='unicode',
                            tokenizer=tokenize 
                        )

    count_matrix = cv.fit_transform(sentences)

    return count_matrix

def bow_tfidf(sentences: Iterable[str], ngram_range: Tuple[int, int]):
    tfidf = TfidfVectorizer(stop_words=stopwords(),
                    ngram_range=ngram_range,
                    tokenizer=tokenize,
                    #min_df=0.05,
                    #max_df=0.95
                    )

    count_matrix = tfidf.fit_transform(sentences)

    return count_matrix

def bow_tfidf_df(sentences: Iterable[str], ngram_range: Tuple[int, int]):
    tfidf = TfidfVectorizer(stop_words=stopwords(),
                    ngram_range=ngram_range,
                    tokenizer=tokenize
                    )

    count_matrix = tfidf.fit_transform(sentences)

    count_array = count_matrix.toarray()

    df = pd.DataFrame(data=count_array,columns = tfidf.get_feature_names())

    return df

def tokenize(text): 
    tknzr = TweetTokenizer(reduce_len=True)
    return tknzr.tokenize(text)

def stopwords():
    return list(set( nltk_stopwords.words('english') ).union( set(text.ENGLISH_STOP_WORDS) ).union(players_stopwords()))

def stopwords_to_json(output_filename: str):

    data = {}
    data['stopwords'] = list(stopwords())

    with open(output_filename, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False)

def main():
    print('main')

    dataset = [ 'Hello my name is james','james this is my python notebook'
                'james trying to create a big dataset',
                'james of words to try differnt',
                'features of count vectorizer']

    dataset = ['my name is james']
    dataset = ['havertz this is my python notebook']

    #print(bow_occurrences_df(dataset, (1,1)))
    #stopwords_to_json('./data/stopwords/stopwords.json')
    

if __name__ == "__main__":
    main()

