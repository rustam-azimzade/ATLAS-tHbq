import pandas as pd
from matplotlib import pyplot as plt


def main():
    JSON_PATH = 'single_t_weighted.json'
    TREE_NAME = 'nominal_Loose'
    BRANCHES_INFO = {
        "nnonbjets": {
            "range": (1, 10),
            "bins": 10,
            "value": "another jets"
        },
        "foxWolfram_2_momentum": {
            "range": (1, 10),
            "bins": 10,
            "value": "value"
        },
    }

    BRANCH_INFO = ['nnonbjets', 'foxWolfram_2_momentum', 'chi2_min_higgs_m', 'nbjets', 'njets',
                    'nfwdjets', 'njets_CBT5', 'njets_CBT4', 'sphericity', 'chi2_min_Imvmass_tH',
                    'chi2_min_Whad_m_ttAll', 'chi2_min_tophad_m_ttAll', 'chi2_min_deltaRq1q2',
                    'chi2_min_bbnonbjet_m', 'chi2_min_toplep_pt', 'chi2_min_tophad_pt_ttAll',
                    'bbs_top_m']
    data = pd.read_json(JSON_PATH)

    for branch_name in BRANCH_INFO:
        tree = data[TREE_NAME]
        if branch_name in tree:
            histogram(data, TREE_NAME, branch_name, 10)


def histogram(data, tree_name, branch_name, bins_count):
    branch_data = pd.Series(data[tree_name][branch_name])
    mean_value = branch_data.mean()

    plt.figure(figsize=(10, 6))
    #plt.hist(branch_data, label=branch_name)
    plt.hist(branch_data, bins=bins_count, range=(0, 10), label=branch_name)
    plt.title(f'Histogram of {branch_name}')
    plt.xlabel(f'{branch_name}')
    plt.xticks(range(0, 10))
    plt.ylabel('Number of collisions')
    plt.tight_layout()
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
