import pandas as pd
from matplotlib import pyplot as plt


def main():
    JSON_PATH = 'single_t_weighted.json'
    data = pd.read_json(JSON_PATH)
    histogram(data, 'nominal_Loose', 'chi2_min_tophad_pt_ttAll', 30)


def histogram(data, tree_name, branch_name, bins_count):
    plt.figure(figsize=(10, 6))
    plt.hist(data[tree_name][branch_name], bins=bins_count, edgecolor="black", label=branch_name)
    plt.title(f"Histogram of {branch_name}")
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
