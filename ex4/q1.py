import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

SAVE = False


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
    estimations_sg30 = estimate_parameters_from_single_experiments(gaussian_repeats_30min, time_points_30min)
    estimations_tg30 = estimate_parameters_from_triple_experiments(gaussian_repeats_30min, time_points_30min)
    estimations_sp30 = estimate_parameters_from_single_experiments(poisson_repeats_30min, time_points_30min)
    estimations_tp30 = estimate_parameters_from_triple_experiments(poisson_repeats_30min, time_points_30min)
    print("--- Gaussian 30min Single\n", estimations_sg30.mean(axis=0))
    print("--- Gaussian 30min Triple\n", estimations_tg30.mean(axis=0))
    print("--- Poisson 30min Single\n", estimations_sp30.mean(axis=0))
    print("--- Poisson 30min Triple\n", estimations_tp30.mean(axis=0))
    plot_parameters_estimations(estimations_sg30, "Single Repeats of Population (Gaussian Noise)",
                                30, create_rate, decay_rate, initial_quantity, "SG30")
    plot_parameters_estimations(estimations_tg30, "Triple Repeats of Population (Gaussian Noise)",
                                30, create_rate, decay_rate, initial_quantity, "TG30")
    plot_parameters_estimations(estimations_sp30, "Single Repeats of Single Cell (Poisson Noise)",
                                30, create_rate, decay_rate, initial_quantity, "SP30")
    plot_parameters_estimations(estimations_tp30, "Triple Repeats of Single Cell (Poisson Noise)",
                                30, create_rate, decay_rate, initial_quantity, "TP30")
    # ------- 1c ---------
    time_points_10min = np.array(range(0, experiment_time + 1, 10))
    gaussian_repeats_10min, poisson_repeats_10min = simulate_experiment_repeats(time_points_10min, create_rate,
                                                                                decay_rate, initial_quantity)
    # plot_simulations(gaussian_repeats_10min, poisson_repeats_10min)
    # estimate parameters and plot results
    estimations_sg10 = estimate_parameters_from_single_experiments(gaussian_repeats_10min, time_points_10min)
    estimations_sp10 = estimate_parameters_from_single_experiments(poisson_repeats_10min, time_points_10min)
    print("--- Gaussian 10min Single\n", estimations_sg10.mean(axis=0))
    print("--- Poisson 10min Single\n", estimations_sp10.mean(axis=0))
    plot_parameters_estimations(estimations_sg10, "Single Repeats of Population (Gaussian Noise)",
                                10, create_rate, decay_rate, initial_quantity, "SG10")
    plot_parameters_estimations(estimations_sp10, "Single Repeats of Single Cell (Poisson Noise)",
                                10, create_rate, decay_rate, initial_quantity, "SP10")


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
    if SAVE:
        plt.savefig("plots/gaussian_simulation.png")
    plt.show()
    poisson_repeats.plot(title='mRNA Levels of Single Cells (Simulated With Poisson Noise)',
                         xlabel='Time (minutes)', ylabel='mRNA (# molecules)')
    if SAVE:
        plt.savefig("plots/poisson_simulation.png")
    plt.show()


def estimate_parameters_from_triple_experiments(experiment_repeats: pd.DataFrame, time_points: np.ndarray):
    mean_triple_repeats = pd.DataFrame()
    for i in range(0, experiment_repeats.shape[0], 1): 
        if i + 3 < experiment_repeats.shape[1]:
            mean_triple_repeats[f'{i}'] = experiment_repeats.iloc[:, i:i + 3].mean(axis=1)
    return estimate_parameters_from_single_experiments(mean_triple_repeats, time_points)


def estimate_parameters_from_single_experiments(experiment_repeats: pd.DataFrame, time_points: np.ndarray):
    params_estimations = pd.DataFrame(columns=["Create Rate", "Decay Rate", "Initial Quantity"])
    for repeat in experiment_repeats:
        params_opt, params_cov = optimize.curve_fit(f=production_base_function, xdata=time_points,
                                                    ydata=experiment_repeats[repeat], bounds=(0, np.inf))
        params_estimations.loc[repeat] = params_opt
    return params_estimations


def plot_parameters_estimations(params_estimations: pd.DataFrame, experiment_kind: str, sample_times: float,
                                real_create_rate: float, real_decay_rate: float, real_start_amount: float,
                                plot_save_name: str):
    titles = [f'{nrmse_title_helper(params_estimations["Create Rate"], real_create_rate)}',
              f'{nrmse_title_helper(params_estimations["Decay Rate"], real_decay_rate)}']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    plt.suptitle(f"Estimation of Creation and Decay Rate Fitted By {experiment_kind}\nExperiments Sampled Every "
                 f"{sample_times} Minutes")
    for i, col in enumerate(["Create Rate", "Decay Rate"]):
        val = params_estimations[col].values
        axes[i].boxplot(val, labels=[col], showmeans=True, showfliers=False, meanline=True)
        axes[i].scatter(np.random.normal(1, 0.04, params_estimations[col].values.shape[0]), val, alpha=0.3)
        axes[i].set_title(titles[i])

    if SAVE:
        plt.savefig(f"plots/{plot_save_name}.png")
    plt.show()


def nrmse_title_helper(estimators: pd.DataFrame, real_val: float):
    return f'Real Value Used: {real_val}, NRMSE: {nrmse(estimators, real_val):.5f}'


def nrmse(x: pd.DataFrame, val: float):
    return np.sqrt(((np.array(x) - val) ** 2).mean()) / x.std()


if __name__ == '__main__':
    question1()
