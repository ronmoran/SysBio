import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize


def question1():
    experiment_time = 180  # minutes
    decay_rate = 0.04  # 1/min
    create_rate = 8  # molecules/minute
    initial_quantity = 50  # 2/0.04 molecules = old_create_rate/decay_rate

    # ------- 1a + 1b ---------
    time_points_30min = np.array(range(0, experiment_time + 1, 30))
    gaussian_repeats_30min, poisson_repeats_30min = simulate_experiment_repeats(time_points_30min, create_rate,
                                                                                decay_rate, initial_quantity)
    # plot_simulations(gaussian_repeats_30min, poisson_repeats_30min)
    # --- estimate parameters and plot results
    gaussian_30min_estimators = estimate_parameters_from_experiments(gaussian_repeats_30min, time_points_30min)
    plot_parameters_estimations(gaussian_30min_estimators,
                                "Single Repeats of Population (Gaussian Noise)", 30, create_rate, decay_rate,
                                initial_quantity, "SG30")
    plot_parameters_estimations(triple_estimators_estimations(gaussian_30min_estimators),
                                "Triple Repeats of Population (Gaussian Noise)", 30, create_rate, decay_rate,
                                initial_quantity, "TG30")
    # create_params_estimations_triple_exps(gaussian_repeats_30min, time_points_30min) TODO
    poisson_30min_estimators = estimate_parameters_from_experiments(poisson_repeats_30min, time_points_30min)
    plot_parameters_estimations(poisson_30min_estimators,
                                "Single Repeats of Single Cell (Poisson Noise)", 30, create_rate, decay_rate,
                                initial_quantity, "SP30")
    plot_parameters_estimations(triple_estimators_estimations(poisson_30min_estimators),
                                "Single Repeats of Single Cell (Poisson Noise)", 30, create_rate, decay_rate,
                                initial_quantity, "TP30")

    # ------- 1c ---------
    time_points_10min = np.array(range(0, experiment_time + 1, 10))
    gaussian_repeats_10min, poisson_repeats_10min = simulate_experiment_repeats(time_points_10min, create_rate,
                                                                                decay_rate, initial_quantity)
    # plot_simulations(gaussian_repeats_10min, poisson_repeats_10min)
    # estimate parameters and plot results
    plot_parameters_estimations(estimate_parameters_from_experiments(gaussian_repeats_10min, time_points_10min),
                                "Single Repeats of Population (Gaussian Noise)", 10, create_rate, decay_rate,
                                initial_quantity, "SG10")
    plot_parameters_estimations(estimate_parameters_from_experiments(poisson_repeats_10min, time_points_10min),
                                "Single Repeats of Single Cell (Poisson Noise)", 10, create_rate, decay_rate,
                                initial_quantity, "SP10")


def production_base_function(time: np.ndarray, production_rate: float, removal_rate: float,
                             initial_quantity: float) -> np.ndarray:
    return initial_quantity * np.exp(-removal_rate * time) + \
           (production_rate / removal_rate) * (1 - np.exp(-removal_rate * time))


def simulate_experiment_repeats(time_points: np.ndarray, create_rate: float, decay_rate: float, start_amount: float):
    gaussian_repeats = pd.DataFrame(index=time_points, columns=[f"G-{i}" for i in range(1, 11)])
    poisson_repeats = pd.DataFrame(index=time_points, columns=[f"P-{i}" for i in range(1, 11)])
    for i in range(gaussian_repeats.shape[0]):
        real_amount = production_base_function(time_points[i], create_rate, decay_rate, start_amount)
        gaussian_repeats.iloc[i] = real_amount + np.random.normal(scale=10, size=10)
        poisson_repeats.iloc[i] = real_amount + np.random.poisson(lam=2, size=10)
    return gaussian_repeats, poisson_repeats


def plot_simulations(gaussian_repeats: pd.DataFrame, poisson_repeats: pd.DataFrame):
    gaussian_repeats.plot(title='mRNA Levels of Population (Simulated With Gaussian Noise)',
                          xlabel='Time (minutes)', ylabel='mRNA (# molecules)')
    plt.savefig("plots/gaussian_simulation.png")
    plt.show()
    poisson_repeats.plot(title='mRNA Levels of Single Cells (Simulated With Poisson Noise)',
                         xlabel='Time (minutes)', ylabel='mRNA (# molecules)')
    plt.savefig("plots/poisson_simulation.png")
    plt.show()


def estimate_parameters_from_experiments(experiment_repeats: pd.DataFrame, time_points: np.ndarray):
    params_estimations = pd.DataFrame(columns=["Create Rate", "Decay Rate", "Initial Quantity"])
    for repeat in experiment_repeats:
        params_opt, params_cov = optimize.curve_fit(f=production_base_function, xdata=time_points,
                                                    ydata=experiment_repeats[repeat], bounds=(0, np.inf))
        params_estimations.loc[repeat] = params_opt
    return params_estimations


def triple_estimators_estimations(params_estimations: pd.DataFrame):
    triplet_params_estimations = pd.DataFrame(columns=["Create Rate", "Decay Rate", "Initial Quantity"])
    for i in range(0, params_estimations.shape[0], 1):  # TODO - 1 or 3
        if i + 3 < params_estimations.shape[0]:
            triplet_params_estimations.loc[f'{i}'] = params_estimations.iloc[i:i+3, :].mean(axis=0)
    return triplet_params_estimations


# TODO - how
def triple_experiments_parameter_estimation(experiment_repeats: pd.DataFrame, time_points: np.array):
    params_estimations = pd.DataFrame(columns=["Create Rate", "Decay Rate", "Initial Quantity"])
    params_opt, params_cov = optimize.curve_fit(f=production_base_function,
                                                xdata=np.array([time_points, time_points, time_points]).T,
                                                ydata=experiment_repeats.iloc[:, 0:3], bounds=(0, np.inf))


def plot_parameters_estimations(params_estimations: pd.DataFrame, experiment_kind: str, sample_times: float,
                                real_create_rate: float, real_decay_rate: float, real_start_amount: float,
                                plot_save_name: str):
    # plot_single_param_estimation(params_estimations["Create Rate"], new_create_rate, 'Creation Rate', 'box')
    # plot_single_param_estimation(params_estimations["Decay Rate"], decay_rate, 'Decay Rate', 'box')
    # plot_single_param_estimation(params_estimations["Initial Quantity"], start_amount, 'Initial Quantity', 'box')
    params_estimations.plot(title=f'Estimations of Parameters Fitted By {experiment_kind} Experiments '
                                  f'Sampled Every {sample_times} Minutes\n '
                                  f'{mse_title_helper(params_estimations["Create Rate"], real_create_rate)}                        '
                                  f'{mse_title_helper(params_estimations["Decay Rate"], real_decay_rate)}                        '
                                  f'{mse_title_helper(params_estimations["Initial Quantity"], real_start_amount)}',
                            subplots=True, kind='box', figsize=(15, 5))
    plt.savefig(f"plots/{plot_save_name}.png")
    plt.show()


def mse_title_helper(estimators: pd.DataFrame, real_val: float):
    return f'Real Value Used: {real_val}, MSE: {mse(estimators, real_val):.5f}'


def mse(x: pd.DataFrame, val: float):
    return ((np.array(x) - val)**2).mean()


# todo - delete
def plot_single_param_estimation(estimators: pd.DataFrame, real_val: float, param_name: str, plot_kind: str):
    estimators.plot(kind=plot_kind, title=f'{param_name} Fit From Single Repeats,\n'
                                          f'Real Value Used: {real_val}, MSE: {mse(estimators, real_val):.2f}')
    plt.show()


if __name__ == '__main__':
    question1()
