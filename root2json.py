import uproot
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    ROOT_PATH = 'single_t_weighted.root'
    JSON_PATH = 'data.json'

    root2json(ROOT_PATH, JSON_PATH)


def root2json(where_root: str, where_json: str) -> None:
    data = dict()

    try:
        # Extracting data from .root file
        with uproot.open(where_root) as root_file:
            for tree_name in root_file.keys():
                logging.info(f"Processing tree: {tree_name}")
                tree = root_file[tree_name]
                tree_data = dict()
                for branch_name in tree.keys():
                    logging.info(f"\tProcessing branch: {branch_name}")
                    branch_data = tree[branch_name].array().to_list()
                    tree_data[branch_name] = branch_data
                data[tree_name] = tree_data

        # Writing data to a JSON file
        with open(where_json, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        logging.info(f"Data have been written to {where_json}")

    except Exception as exception_info:
        logging.error(f"An error occurred: {exception_info}")


if __name__ == '__main__':
    main()
