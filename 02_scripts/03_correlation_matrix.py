import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

TEXT_FONT_SIZE = 7
DIGIT_FONT_SIZE = 7
TILE_FONT_SIZE = 14
SAVE_PATH = '../03_results/correlation_matrix_tzbq.png'

def main():
    tHbq_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tHbq_SM_300K_(aTTreethbqSM;1).json')
    tt_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tt_SM_3M_(aTTreett;1).json')
    ttbb_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_ttbb_SM_300K_(aTTreett;1).json')
    ttH_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_ttH_SM_100K_(aTTreetth;1).json')
    tzbq_events = pd.read_json('../01_src/01_data/02_json/MiniNtuple_tzbq_SM_100K_(aTTreethbq;1).json')

    total_events = pd.concat([tzbq_events])

    correlation_matrix = total_events.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, mask=mask, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1,
                cbar=True, square=True, cbar_kws={"shrink": .75}, annot_kws={'fontsize': DIGIT_FONT_SIZE})
    plt.title('Correlation Matrix | tzbq events', fontsize=TILE_FONT_SIZE)
    plt.xticks(rotation=45, fontsize=TEXT_FONT_SIZE, ha='right')
    plt.yticks(fontsize=TEXT_FONT_SIZE)
    plt.savefig(SAVE_PATH, dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
