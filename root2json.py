import uproot
import json
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    ROOT_PATH = 'single_t_weighted.root'
    JSON_PATH = ROOT_PATH.replace('.root', '.json')
    root2json(ROOT_PATH, JSON_PATH)


def strip_encoding(name: str) -> str:
    return name.split(';')[0]


def root2json(where_root: str, where_json: str) -> None:
    extracted_data = dict()

    try:
        # Extracting data from .root file
        with uproot.open(where_root) as root_file:
            for tree_name in root_file.keys():
                tree_name = strip_encoding(tree_name)
                logging.info(f"Processing tree: {tree_name}")
                tree = root_file[tree_name]
                tree_data = dict()
                for branch_name in tree.keys():
                    branch_name = strip_encoding(branch_name)
                    logging.info(f"\tProcessing branch: {branch_name}")
                    branch_data = tree[branch_name].arrays()
                    #branch_data = tree[branch_name].arrays().tolist()
                    #branch_data = tree[branch_name].arrays(library="np")
                    # branch_data = tree[branch_name].arrays(library="pd")
                    # branch_data = tree[branch_name].arrays(library="ak")
                    tree_data[branch_name] = branch_data
                extracted_data[tree_name] = tree_data
    except Exception as exception_info:
        logging.error(f"An error occurred: {exception_info}")
        return

    # Writing data to a JSON file
    data_frame = pd.DataFrame(extracted_data)
    data_frame.to_json(where_json, index=False)
    logging.info(f"Data have been written to {where_json}")


if __name__ == '__main__':
    main()
