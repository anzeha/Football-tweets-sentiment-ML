import pandas as pd

def export_for_doccano(input_filename: str, output_filename: str):
    df = pd.read_csv(input_filename)

    df = df.replace(r'\n', ' ', regex=True)


    with open(output_filename, 'a') as f:
        pd.set_option('display.max_colwidth', None)
        dfAsString = df.to_string(columns=['tweet_text'],header=False, index=False, justify='center')
        df.style.set_properties(subset=['tweet_text'],**{'text-align': 'left'})
        f.write(dfAsString)


def main():
    export_for_doccano('./data/raw/england_italy_tweets.csv', './data/doccano/england_italy_tweets.txt')
    

if __name__ == "__main__":
    main()

