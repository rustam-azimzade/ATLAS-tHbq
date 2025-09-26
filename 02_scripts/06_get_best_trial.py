import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
import matplotlib.pyplot as plt
from Tools.scripts.var_access_benchmark import trials


def main():
    study = optuna.load_study(
        study_name='Hyperparameter_optimization',
        storage='sqlite:///../03_results/03_neural_network/optimization.db'
    )
    total_trials = study.trials
    pruned_trials = [t for t in total_trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in total_trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in total_trials if t.state == optuna.trial.TrialState.FAIL]

    print(f'Total number of trials: {len(total_trials)}')
    print(f'Pruned trials: {len(pruned_trials)}')
    print(f'Completed trials: {len(complete_trials)}')
    print(f'Failed trials: {len(failed_trials)}')

    best_trial = study.best_trial
    print('Best Neural Network:')
    print(f'\tID:  {best_trial.number}')
    print(f'\tAUC: {best_trial.value}')
    print('\tHyperparameters:')
    for key, value in best_trial.params.items():
        print(f'\t\t{key}: {value}')

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_parallel_coordinate(study) #plot_parallel_coordinate(study, params=["lr", "n_layers"])
    #fig4 = plot_contour(study, params=["lr", "n_layers"])

    #fig2.show()

if __name__ == '__main__':
    main()