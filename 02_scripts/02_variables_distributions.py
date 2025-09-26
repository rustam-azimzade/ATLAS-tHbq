from modules import Events, Plots

SAVE_PATH = '../03_results/01_variables_distributions'

def main():
    events_handler = Events()
    events_handler.load_data().label_data()

    signal_events = events_handler.get_signal_events()
    background_events = events_handler.get_background_events()

    plotter = Plots(save_path=SAVE_PATH)

    for variable_name in events_handler.get_variable_names():
        signal_variable = signal_events[[variable_name, 'channel']].copy()
        background_variable = background_events[[variable_name, 'channel']].copy()

        signal_weights = events_handler.get_weights(data=signal_variable, equalize_classes=False)
        background_weights = events_handler.get_weights(data=background_variable, equalize_classes=False)

        plotter.plot_histogram(
            signal_events=signal_variable, background_events=background_variable,
            signal_weights=signal_weights, background_weights=background_weights,
            normalize_to_background=True
        )
        plotter.save_plot()


if __name__ == '__main__':
    main()
