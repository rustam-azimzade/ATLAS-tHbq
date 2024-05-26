import uproot
import json
import pandas as pd
import numpy as np


def main():
    ROOT_PATH = 'single_t_weighted.root'
    JSON_PATH = 'data.json'

    root2json(ROOT_PATH, JSON_PATH)
    data = read_json(JSON_PATH)
    print(data)


def json_serializable(item):
    # Use recursion in every check to decode nested arrays/values as well
    if isinstance(item, pd.DataFrame):
        return json_serializable(item.to_dict(orient='list'))

    if isinstance(item, pd.Series):
        return json_serializable(item.to_list())

    if isinstance(item, (np.integer, np.floating, np.bool_)):
        return json_serializable(item.item())

    if isinstance(item, np.generic):
        return json_serializable(item.item())

    if isinstance(item, np.ndarray):
        return json_serializable(item.tolist())

    return str(item)


def root2json(where_root: str, where_json: str) -> None:
    data = dict()

    # Extracting data from .root file
    with uproot.open(where_root) as root_file:
        for tree_name in root_file.keys():
            print(f"Processing tree: {tree_name}")
            tree = root_file[tree_name]
            tree_data = dict()
            for branch in tree.keys():
                print(f"\tProcessing branch: {branch}")
                branch_data = tree[branch].array(library='np')
                tree_data[branch] = json_serializable(branch_data)
            data[tree_name] = tree_data

    # Writing data to a JSON file
    with open(where_json, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data have been written to {where_json}")


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
