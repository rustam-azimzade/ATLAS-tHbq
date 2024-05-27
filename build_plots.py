import json
import pandas as pd
import numpy as np


def main():
    JSON_PATH = 'data.json'
    data = read_json(JSON_PATH)
    print(data['nominal_Loose'].keys())


def read_json(where_json: str):
    # Reading data from a JSON file as a regular array
    with open(where_json, 'r') as json_file:
        data = json.load(json_file)
    print("Data read from JSON as regular dictionary:")

    # Reading data with pandas (if needed)
    # data = pd.read_json(where_json, lines=True)
    #print("Data read from JSON using pandas DataFrame:")

    return data


if __name__ == '__main__':
    main()
