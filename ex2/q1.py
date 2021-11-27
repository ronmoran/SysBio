import numpy as np
import plotly.express as px
import pandas as pd
import os

DT = 1e-3
TOTAL_TIME = 1.5
PLOT_PATH = r"C:\Uni\drive\Studies\yearC\SysBio\ex2"


def simulate_negative_autoregulation(max_rate: float, degradation_rate: float, kd: float, hill_coef: int,
                                     total_time: float, dt: float):
    steps = int(total_time / dt)
    protein_concentration = np.empty(steps, dtype=float)
    protein_concentration[0] = 0.0
    kd_exp_coef = kd ** hill_coef
    for i in range(1, steps):
        protein_concentration[i] = protein_concentration[i - 1] + \
                                   calc_nar_change(degradation_rate, hill_coef, kd_exp_coef, max_rate,
                                                   protein_concentration[i - 1]) * dt
    return np.array([np.linspace(0, steps / dt, num=steps), protein_concentration])


def calc_nar_change(degradation_rate, hill_coef, kd_exp_coef, max_rate, current_protein_concentration):
    return max_rate * (kd_exp_coef / (kd_exp_coef + (current_protein_concentration ** hill_coef)))\
            - degradation_rate * current_protein_concentration


def simulate_regular_regulation(max_rate: float, degradation_rate: float, total_time: float, dt: float) -> \
        (np.ndarray, np.ndarray):
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    return time_vals, (max_rate / degradation_rate) * (1 - np.exp(-degradation_rate * time_vals))
    # concentration = np.empty(time_vals.shape)
    # concentration[0] = 0
    # for i in range(1, len(time_vals)):
    #     concentration[i] = concentration[i - 1] + calc_reg_change(max_rate, degradation_rate, concentration[i - 1]) * dt
    # return time_vals, concentration


def calc_reg_change(max_rate, degradation_rate, current_concentration):
    return max_rate - degradation_rate * current_concentration


def plot_rate_balance(degradation_rate: float, max_rate: float, kd: float, hill_coefs: np.ndarray,
                      max_concentration: float, num_samples: int, fig_name: str):
    concentration = np.linspace(0, max_concentration, num=num_samples)
    productions_rates = np.empty((hill_coefs.shape[0], concentration.shape[0]))
    col_names = []
    for i, hill_coef in enumerate(hill_coefs):
        productions_rates[i] = max_rate * (kd ** hill_coef / (kd**hill_coef + concentration**hill_coef))
        col_names.append(f"prod hill coef = {hill_coef}")
    degradation = degradation_rate * concentration
    col_names.append('degradation')
    plt = px.line(
        pd.DataFrame(np.vstack([productions_rates, degradation]).T, index=concentration, columns=col_names),
                  title='Rate Balance Plot',
                  labels={
                      "index": "Concentration",
                      "value": "Rate",
                      "variable": "Regulation"
                  })
    plt.write_image(os.path.join(PLOT_PATH, fig_name))
    plt.show()


def plot_concentrations(data, variable, fig_name):
    plt = px.line(data,
                  title='Concentration In Time',
                  labels={
                      "index": "Time",
                      "value": "Concentration",
                      "variable": variable
                  }
                  )
    plt.add_hline(y=data.iloc[-1][0] / 2, line_dash="dot", annotation_text="steady-state / 2")
    plt.update_xaxes(visible=False)
    plt.show()
    plt.write_image(os.path.join(PLOT_PATH, fig_name))


def q1a(kd, hill_coef, max_rate, degradation_rate):
    nar_data = simulate_negative_autoregulation(max_rate, degradation_rate, kd, hill_coef, TOTAL_TIME, DT)[1]
    reg_data = simulate_regular_regulation(2.45, degradation_rate, TOTAL_TIME, DT)[1]
    both_regulation_data = pd.DataFrame(np.array([nar_data, reg_data]).T, columns=['NAR', 'Regular regulation'])
    plot_concentrations(both_regulation_data, "Regulation Type", "q1a1.png")
    plot_rate_balance(degradation_rate, max_rate, kd, hill_coef, 1, 1000, "q1a2.png")


def q1b(kd, max_rate, degradation_rate):
    hill_coefs = np.array([1, 4])
    hill_one = simulate_negative_autoregulation(max_rate, degradation_rate, kd, hill_coefs[0], TOTAL_TIME, DT)[1]
    hill_four = simulate_negative_autoregulation(max_rate, degradation_rate, kd, hill_coefs[1], TOTAL_TIME, DT)[1]
    both_regulation_data = pd.DataFrame(np.array([hill_one, hill_four]).T, columns=hill_coefs.astype(str))
    plot_concentrations(both_regulation_data, "Hill Coefficient", "q1b1.png")
    plot_rate_balance(degradation_rate, max_rate, kd, hill_coefs, 1.5, 1000, "q1b2.png")


if __name__ == "__main__":
    kd = 0.8
    hill_coef = 3
    max_rate = 5
    degradation_rate = 3
    q1a(kd, hill_coef, max_rate, degradation_rate)
    q1b(kd, max_rate, degradation_rate)
