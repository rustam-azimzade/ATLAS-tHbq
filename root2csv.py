import uproot
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    ROOT_PATH = 'single_t_weighted.root'
    CSV_PATH = ROOT_PATH.replace('.root', '.csv')
    root2csv(ROOT_PATH, CSV_PATH)


def root2csv(where_root: str, where_csv: str) -> None:

    strip_encoding = lambda name: name.split(';')[0]

    try:
        # Extracting data from .root file
        with uproot.open(where_root) as root_file:
            tree_names = root_file.keys()
            for tree_name in tree_names:
                tree_name = strip_encoding(tree_name)
                logging.info(f"Processing tree: {tree_name}")
                tree = root_file[tree_name]
                branch_names = tree.keys()
                tree_data = tree.arrays(branch_names, library="ak")
                tree_data_frame = pd.DataFrame({branch: tree_data[branch].tolist() for branch in branch_names})
                tree_data_frame.to_csv(where_csv, index=False)
                logging.info(f"Data have been written to {where_csv}")
    except Exception as exception_info:
        logging.error(f"An error occurred: {exception_info}")
        return


if __name__ == '__main__':
    main()
