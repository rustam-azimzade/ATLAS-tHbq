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


def root2json(where_root: str, where_json: str) -> None:
    extracted_data = dict()
    strip_encoding = lambda name: name.split(';')[0]

    try:
        # Extracting data from .root file
        with uproot.open(where_root) as root_file:

            tree_names = root_file.keys()
            for tree_name in tree_names:
                tree_name = strip_encoding(tree_name)
                logging.info(f"Processing tree: {tree_name}")
                tree = root_file[tree_name]
                tree_data = dict()

                branch_names = tree.keys()
                for branch_name in branch_names:
                    branch_name = strip_encoding(branch_name)
                    logging.info(f"\tProcessing branch: {branch_name}")
                    branch_data = tree[branch_name].arrays(library="ak").tolist() # library=None library="np" library="pd"
                    tree_data[branch_name] = branch_data

                extracted_data[tree_name] = tree_data

        # Writing data to a JSON file
        data_frame = pd.DataFrame(extracted_data)
        data_frame.to_json(where_json, index=False)
        logging.info(f"Data have been written to {where_json}")

    except Exception as exception_info:
        logging.error(exception_info)

    # Writing data to a JSON file
    data_frame = pd.DataFrame(extracted_data)
    data_frame.to_json(where_json, index=False)
    logging.info(f"Data have been written to {where_json}")


if __name__ == '__main__':
    main()
