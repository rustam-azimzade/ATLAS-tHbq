from config import *
import uproot
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    ROOT_PATH = 'data/root/single_t_weighted.root'
    JSON_PATH = ROOT_PATH.replace('root', 'json')
    root2json(ROOT_PATH, JSON_PATH)


def root2json(where_root: str, where_json: str) -> None:
    extracted_data = dict()
    strip_encoding = lambda name: name.split(';')[0]

    try:
        # Extracting data from .root file
        with uproot.open(where_root) as root_file:
            for tree_name in root_file.keys():
                tree_name = strip_encoding(tree_name)
                logging.info(f"Processing tree: {tree_name}")
                tree = root_file[tree_name]
                tree_data = dict()
                for branch_name in tree.keys():
                    if branch_name in Config.NECESSARY_BRANCHES:
                        branch_name = strip_encoding(branch_name)
                        logging.info(f"\tProcessing branch: {branch_name}")
                        branch_data = tree[branch_name].array(library="ak").to_list()

                        branch_data_length = len(branch_data)
                        for i in range(branch_data_length):
                            if isinstance(branch_data[i], list):
                                if len(branch_data[i]) == 1:
                                    branch_data[i] = branch_data[i][0]
                                if not branch_data[i]:
                                    branch_data[i] = None

                        tree_data[branch_name] = branch_data
                extracted_data[tree_name] = tree_data

        # Writing data to a JSON file
        with open(where_json, 'w', encoding='utf-8') as json_file:
            json.dump(extracted_data, json_file, ensure_ascii=False)
        logging.info(f"Data have been written to {where_json}")

    except Exception as exception_info:
        logging.error(exception_info)


if __name__ == '__main__':
    main()
