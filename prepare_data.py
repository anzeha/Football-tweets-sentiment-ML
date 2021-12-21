import pandas as pd
import json
from unidecode import unidecode
import os

#WARNING: remove one dot from path if you want to run this file, path relative to /notebook folder
PLAYERS_TXT_PATH = os.path.realpath('../data/players/euro_players.txt')
PLAYERS_JSON_PATH = os.path.realpath('../data/players/euro_players.json')

def export_for_doccano(input_filename: str, output_filename: str):
    df = pd.read_csv(input_filename)

    df = df.replace(r'\n', ' ', regex=True)


    with open(output_filename, 'a') as f:
        pd.set_option('display.max_colwidth', None)
        dfAsString = df.to_string(columns=['tweet_text'],header=False, index=False, justify='center')
        df.style.set_properties(subset=['tweet_text'],**{'text-align': 'left'})
        f.write(dfAsString)

def parse_players_to_json(input_filename: str = PLAYERS_TXT_PATH, output_filename: str = PLAYERS_JSON_PATH):
    file = open(input_filename)

    lines = file.readlines()

    players = []

    for line in lines:
        player = (line.split(':'))[1].split('(')[0].strip()
        players.append(player)
        if unidecode(player) != player:
            players.append(unidecode(player))

    data = {}
    data['players'] = players

    with open(output_filename, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False)

def players_stopwords(player_txt_path: str = PLAYERS_TXT_PATH, players_json_path: str = PLAYERS_JSON_PATH):
    if os.path.exists(players_json_path) is False:
        parse_players_to_json(player_txt_path)

    file = open(players_json_path)

    data = json.load(file)

    splitted_players = []

    #name and surname is a stopword separately
    for player in data['players']:
        splitted_players.append(list(map(lambda x: str(x).lower(), player.split())))

    splitted_players = sum(splitted_players, [])

    #get rid of duplicates
    splitted_players = list(dict.fromkeys(splitted_players))

    #add country names to stop words
    splitted_players = splitted_players + ['england', 'italy', 'germany', 'croatia']

    #add "username" as all user tags were replaced with "username":  "@random123" -> "username"
    splitted_players = splitted_players + ['username']

    return splitted_players


def main():
    print('main')
    #parse_players_to_json()    
    #print(players_stopwords())

if __name__ == "__main__":
    main()

