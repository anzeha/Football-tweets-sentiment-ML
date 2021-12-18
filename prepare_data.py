import pandas as pd

def export_for_doccano(input_filename: str, output_filename: str):
    df = pd.read_csv(input_filename)

    with open(output_filename, 'a') as f:
        dfAsString = df.iloc[:,1].to_string(header=False, index=False)
        f.write(dfAsString)


def main():
    export_for_doccano('./data/raw/england_italy_tweets.csv', './data/doccano/england_italy_tweets.txt')
    

if __name__ == "__main__":
    main()

