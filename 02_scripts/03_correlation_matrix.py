import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from modules import Config

TEXT_FONT_SIZE = 7
DIGIT_FONT_SIZE = 3
TILE_FONT_SIZE = 14
SAVE_PATH = '../03_results/02_correlation_matrices'

def main():
    tHbq_events = pd.read_json('../01_src/01_data/02_json/Ntuple_tHbq_22M.json')
    tt_events = pd.read_json('../01_src/01_data/02_json/Ntuple_tt_22M.json')
    ttbb_events = pd.read_json('../01_src/01_data/02_json/Ntuple_ttbb_65979470.json')
    ttH_events = pd.read_json('../01_src/01_data/02_json/Ntuple_ttH_22M.json')
    tZbq_events = pd.read_json('../01_src/01_data/02_json/Ntuple_tZbq_22M.json')
    ttZ_events = pd.read_json('../01_src/01_data/02_json/Ntuple_ttZ_22M.json')
    ttW_events = pd.read_json('../01_src/01_data/02_json/Ntuple_ttW_22M.json')

    total_events = pd.concat([tHbq_events, tt_events, ttbb_events, ttH_events, tZbq_events, ttZ_events, ttW_events])

    # Select variable
    #selected_variables = list(Config.VARIABLES_DESCRIPTION.keys())
    #total_events = total_events[selected_variables].copy()
    #

    correlation_matrix = total_events.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, mask=mask, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1,
                cbar=True, square=True, cbar_kws={"shrink": .75}, annot_kws={'fontsize': DIGIT_FONT_SIZE})
    plt.title('Correlation Matrix | Total Events', fontsize=TILE_FONT_SIZE)
    plt.xticks(rotation=45, fontsize=TEXT_FONT_SIZE, ha='right')
    plt.yticks(fontsize=TEXT_FONT_SIZE)
    plt.savefig(f'{SAVE_PATH}/total.png', dpi=300)
    plt.savefig(f'{SAVE_PATH}/total.pdf')
    plt.close()
    ########################
    threshold = 0.8

    # Получаем пары переменных с корреляцией выше порога (без дублирования и диагонали)
    high_corr = (
        correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    high_corr.columns = ['Variable 1', 'Variable 2', 'Pearson Correlation']
    high_corr_filtered = high_corr[high_corr['Pearson Correlation'].abs() >= threshold]

    # Печать результатов
    print("Highly correlated variable pairs (|correlation| >= 0.8):")
    print(high_corr_filtered.sort_values(by='Pearson Correlation', ascending=False).to_string(index=False))



if __name__ == '__main__':
    main()
