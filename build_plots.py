import pandas as pd


def main():
    JSON_PATH = 'single_t_weighted.json'
    data = pd.read_json(JSON_PATH)
    print(data['nominal_Loose'])


if __name__ == '__main__':
    main()
