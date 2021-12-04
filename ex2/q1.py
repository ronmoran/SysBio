import numpy as np
import plotly.express as px
import pandas as pd
import os

import plotly

DT = 1e-3
TOTAL_TIME = 1.5
PLOT_PATH = r"lyx_files\plots"


def simulate_negative_autoregulation(max_rate: float, degradation_rate: float, kd: float, hill_coef: int,
                                     total_time: float, dt: float):
    steps = int(total_time / dt)
    protein_concentration = np.empty(steps, dtype=float)
    protein_concentration[0] = 0.0
    for i in range(1, steps):
        protein_concentration[i] = protein_concentration[i - 1] + \
                                   calc_nar_change(degradation_rate, hill_coef, kd, max_rate,
                                                   protein_concentration[i - 1]) * dt
    return np.array([np.linspace(0, steps / dt, num=steps), protein_concentration])


def calc_nar_change(degradation_rate, hill_coef, kd, max_rate, current_protein_concentration):
    return max_rate * (kd ** hill_coef / (kd ** hill_coef + (current_protein_concentration ** hill_coef)))\
            - degradation_rate * current_protein_concentration


def simulate_regular_regulation(max_rate: float, degradation_rate: float, total_time: float, dt: float) -> \
        (np.ndarray, np.ndarray):
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    return time_vals, calc_regular_production(max_rate, degradation_rate, time_vals)
    # concentration = np.empty(time_vals.shape)
    # concentration[0] = 0
    # for i in range(1, len(time_vals)):
    #     concentration[i] = concentration[i - 1] + calc_reg_change(max_rate, degradation_rate, concentration[i - 1]) * dt
    # return time_vals, concentration


def calc_regular_production(max_rate, degradation_rate, time_vals):
    return (max_rate / degradation_rate) * (1 - np.exp(-degradation_rate * time_vals))

def calc_regular_degradation(initial_amount, degradation_rate, time_vals):
    return initial_amount * np.exp(-degradation_rate * time_vals)


def calc_reg_change(max_rate, degradation_rate, current_concentration):
    return max_rate - degradation_rate * current_concentration
def calc_reg_deg(degradation_rate, current_concentration):
    return -degradation_rate * current_concentration


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


def plot_concentrations(data, variable, fig_name, add_half_concentration=True, x_axis_visible=False):
    plt = px.line(data,
                  title='Concentration In Time',
                  labels={
                      "index": "Time",
                      "value": "Concentration",
                      "variable": variable
                  }
                  )
    if add_half_concentration:
        plt.add_hline(y=data.iloc[-1][0] / 2, line_dash="dot", annotation_text="steady-state / 2")
    plt.update_xaxes(visible=x_axis_visible)
    plt.show()
    plt.write_image(os.path.join(PLOT_PATH, fig_name))


def plot_concentrations_q4(data, variable, fig_name, add_half_concentration=True, x_axis_visible=False):
    fig = plotly.subplots.make_subplots(rows=3, cols=1)
    plt = px.line(data,
                  title='Concentration In Time',
                  labels={
                      "index": "Time",
                      "value": "Concentration",
                      "variable": variable
                  }
                  )
    if add_half_concentration:
        plt.add_hline(y=data.iloc[-1][0] / 2, line_dash="dot", annotation_text="steady-state / 2")
    plt.update_xaxes(visible=x_axis_visible)
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


def q4a(kyz, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time/dt + 2 < sig_stop_time/dt < total_time/dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    sig_start_step = int(sig_start_time/dt)
    sig_stop_step = int(sig_stop_time/dt)
    no_active_x_start = time_vals[0:sig_start_step]
    active_x = time_vals[sig_start_step:sig_stop_step] - time_vals[sig_start_step]
    no_active_x_end = time_vals[sig_stop_step:] - time_vals[sig_stop_step]
    y_prod = calc_regular_production(y_max_rate, y_deg_rate, active_x)
    y_deg = calc_regular_degradation(y_prod[-1], y_deg_rate, no_active_x_end)
    y = np.array([0.] * len(no_active_x_start) + list(y_prod) + list(y_deg))
    z_prod_time_vals = time_vals[(y > kyz) & (sig_start_step < np.arange(time_vals.size)) &
                                              (np.arange(time_vals.size) < sig_stop_step)]
    z_prod = calc_regular_production(z_max_rate, z_deg_rate, z_prod_time_vals - z_prod_time_vals[0])
    z_deg = calc_regular_degradation(z_prod[-1], z_deg_rate, no_active_x_end)
    no_z = np.zeros((time_vals[time_vals < z_prod_time_vals[0]].size,))
    z = np.hstack((no_z, z_prod, z_deg))
    signal = np.where((sig_start_step < np.arange(time_vals.size)) &  (np.arange(time_vals.size) < sig_stop_step),
                      np.max(np.hstack((y, z))) * 1.2, 0.)
    data = pd.DataFrame(np.vstack((y, z, signal)).T, columns=["Y", "Z", "signal"])
    plot_concentrations(data, "concentration", "q4a.png", False)


def q4b(kyz, kxy, kxz, x_max_rate, x_deg_rate, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time/dt + 2 < sig_stop_time/dt < total_time/dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    sig_start_step = int(sig_start_time/dt)
    sig_stop_step = int(sig_stop_time/dt)
    no_signal_start = time_vals[0:sig_start_step]
    x_prod_time = time_vals[sig_start_step:sig_stop_step] - time_vals[sig_start_step]
    x_deg_time = time_vals[sig_stop_step:] - time_vals[sig_stop_step]
    x_prod = calc_regular_production(x_max_rate, y_deg_rate, x_prod_time)
    x_deg = calc_regular_degradation(x_prod[-1], x_deg_rate, x_deg_time)
    x = np.array([0.] * len(no_signal_start) + list(x_prod) + list(x_deg))
    y_prod_time_vals = time_vals[x > kxy]
    y_prod_time = y_prod_time_vals - y_prod_time_vals[0]
    y_prod = calc_regular_production(y_max_rate, y_deg_rate, y_prod_time)
    y_deg_time = time_vals[(x < kxy) & (time_vals > y_prod_time_vals[-1])]
    y_deg = calc_regular_degradation(y_prod[-1], y_deg_rate, y_deg_time - y_deg_time[0])
    y = np.array([0.] * np.count_nonzero(time_vals < y_prod_time_vals[0]) + list(y_prod) + list(y_deg))
    z_prod_time_vals = time_vals[(y > kyz) & (x > kxz)]
    z_prod = calc_regular_production(z_max_rate, z_deg_rate, z_prod_time_vals - z_prod_time_vals[0])
    z_deg_time = time_vals[((x < kxz) | (y < kyz)) & (time_vals > z_prod_time_vals[-1])]
    z_deg = calc_regular_degradation(z_prod[-1], z_deg_rate, z_deg_time - z_deg_time[0])
    no_z = np.zeros((np.count_nonzero(time_vals < z_prod_time_vals[0]),))
    z = np.hstack((no_z, z_prod, z_deg))
    signal = np.where((sig_start_step < np.arange(time_vals.size)) & (np.arange(time_vals.size) < sig_stop_step),
                      np.max(np.hstack((x, y, z))) * 1.2, 0.)
    data = pd.DataFrame(np.vstack((x, y, z, signal)).T, columns=["X", "Y", "Z", "signal"])
    plot_concentrations(data, "concentration", "q4a.png", False)


def q4c(kyz, kxy, kxz, x_max_rate, x_deg_rate, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time/dt + 2 < sig_stop_time/dt < total_time/dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    sig_start_step = int(sig_start_time/dt)
    sig_stop_step = int(sig_stop_time/dt)
    no_signal_start = time_vals[0:sig_start_step]
    x_prod_time = time_vals[sig_start_step:sig_stop_step] - time_vals[sig_start_step]
    x_deg_time = time_vals[sig_stop_step:] - time_vals[sig_stop_step]
    x_prod = calc_regular_production(x_max_rate, y_deg_rate, x_prod_time)
    x_deg = calc_regular_degradation(x_prod[-1], x_deg_rate, x_deg_time)
    x = np.array([0.] * len(no_signal_start) + list(x_prod) + list(x_deg))
    y_prod_time_vals = time_vals[x > kxy]
    y_prod_time = y_prod_time_vals - y_prod_time_vals[0]
    y_prod = calc_regular_production(y_max_rate, y_deg_rate, y_prod_time)
    y_deg_time = time_vals[(x < kxy) & (time_vals > x_prod_time[-1])]
    y_deg = calc_regular_degradation(y_prod[-1], y_deg_rate, y_deg_time - y_deg_time[0])
    y = np.array([0.] * np.count_nonzero(time_vals < y_prod_time_vals[0]) + list(y_prod) + list(y_deg))
    z_prod_time_vals = time_vals[(y > kyz) | (x > kxz)]
    z_prod = calc_regular_production(z_max_rate, z_deg_rate, z_prod_time_vals - z_prod_time_vals[0])
    z_deg_time = time_vals[((x < kxz) & (y < kyz)) & (time_vals > z_prod_time_vals[-1])]
    z_deg = calc_regular_degradation(z_prod[-1], z_deg_rate, z_deg_time - z_deg_time[0])
    no_z = np.zeros((np.count_nonzero(time_vals < z_prod_time_vals[0]),))
    z = np.hstack((no_z, z_prod, z_deg))
    signal = np.where((sig_start_step < np.arange(time_vals.size)) & (np.arange(time_vals.size) < sig_stop_step),
                      np.max(np.hstack((x, y, z))) * 1.2, 0.)
    data = pd.DataFrame(np.vstack((x, y, z, signal)).T, columns=["X", "Y", "Z", "signal"])
    plot_concentrations(data, "concentration", "q4a.png", False)

def q4d(kyz, kxy, kxz, x_max_rate, x_deg_rate, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time/dt + 2 < sig_stop_time/dt < total_time/dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    sig_start_step = int(sig_start_time/dt)
    sig_stop_step = int(sig_stop_time/dt)
    no_signal_start = time_vals[0:sig_start_step]
    x_prod_time = time_vals[sig_start_step:sig_stop_step] - time_vals[sig_start_step]
    x_deg_time = time_vals[sig_stop_step:] - time_vals[sig_stop_step]
    x_prod = calc_regular_production(x_max_rate, y_deg_rate, x_prod_time)
    x_deg = calc_regular_degradation(x_prod[-1], x_deg_rate, x_deg_time)
    x = np.array([0.] * len(no_signal_start) + list(x_prod) + list(x_deg))
    y_prod_time_vals = time_vals[x > kxy]
    y_prod_time = y_prod_time_vals - y_prod_time_vals[0]
    y_prod = calc_regular_production(y_max_rate, y_deg_rate, y_prod_time)
    y_deg_time = time_vals[(x < kxy) & (time_vals > x_prod_time[-1])]
    y_deg = calc_regular_degradation(y_prod[-1], y_deg_rate, y_deg_time - y_deg_time[0])
    y = np.array([0.] * np.count_nonzero(time_vals < y_prod_time_vals[0]) + list(y_prod) + list(y_deg))
    z_prod_time_vals_bool = (y > kyz) & (x < kxz)
    z = np.zeros(time_vals.shape, dtype=float)
    for i in range(1, z.size):
        if z_prod_time_vals_bool[i]:
            change = calc_reg_change(z_max_rate, z_deg_rate, z[i - 1])
        else:
            change = calc_reg_deg(z_deg_rate, z[i - 1])
        z[i] += z[i - 1] + change * dt
    # z_prod = calc_regular_production(z_max_rate, z_deg_rate, z_prod_time_vals - z_prod_time_vals[0])
    # z_deg_time = time_vals[((x > kxz) | (y < kyz)) & (time_vals > z_prod_time_vals[-1])]
    # z_deg = calc_regular_degradation(z_prod[-1], z_deg_rate, z_deg_time - z_deg_time[0])
    # no_z = np.zeros((np.count_nonzero(time_vals < z_prod_time_vals[0]),))
    # z = np.hstack((no_z, z_prod, z_deg))
    signal = np.where((sig_start_step < np.arange(time_vals.size)) & (np.arange(time_vals.size) < sig_stop_step),
                      np.max(np.hstack((x, y, z))) * 1.2, 0.)
    data = pd.DataFrame(np.vstack((x, y, z, signal)).T, columns=["X", "Y", "Z", "signal"])
    plot_concentrations(data, "concentration", "q4a.png", False)


def q4e(kyz, kxy, kxz, x_max_rate, x_deg_rate, x_hill_coef, x_kd, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time/dt + 2 < sig_stop_time/dt < total_time/dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    sig_start_step = int(sig_start_time/dt)
    sig_stop_step = int(sig_stop_time/dt)
    no_signal_start = time_vals[0:sig_start_step]
    x_prod_time = time_vals[sig_start_step:sig_stop_step] - time_vals[sig_start_step]
    x_prod_time_bool = (sig_start_time < time_vals) & (time_vals < sig_stop_time)
    x = np.zeros(time_vals.shape, dtype=float)
    for i in range(1, x.size):
        if x_prod_time_bool[i]:
            change = calc_nar_change(x_deg_rate, x_hill_coef, x_kd, x_max_rate, x[i - 1])
        else:
            change = calc_reg_deg(x_deg_rate, x[i - 1])
        x[i] = x[i - 1] + change * dt
    # x_deg_time = time_vals[sig_stop_step:] - time_vals[sig_stop_step]
    # x_prod = calc_regular_production(x_max_rate, y_deg_rate, x_prod_time)
    # x_deg = calc_regular_degradation(x_prod[-1], x_deg_rate, x_deg_time)
    # x = np.array([0.] * len(no_signal_start) + list(x_prod) + list(x_deg))
    y_prod_time_vals = time_vals[x > kxy]
    y_prod_time = y_prod_time_vals - y_prod_time_vals[0]
    y_prod = calc_regular_production(y_max_rate, y_deg_rate, y_prod_time)
    y_deg_time = time_vals[(x < kxy) & (time_vals > x_prod_time[-1])]
    y_deg = calc_regular_degradation(y_prod[-1], y_deg_rate, y_deg_time - y_deg_time[0])
    y = np.array([0.] * np.count_nonzero(time_vals < y_prod_time_vals[0]) + list(y_prod) + list(y_deg))
    z_prod_time_vals_bool = (y > kyz) & (x < kxz)
    z = np.zeros(time_vals.shape, dtype=float)
    for i in range(1, z.size):
        if z_prod_time_vals_bool[i]:
            change = calc_reg_change(z_max_rate, z_deg_rate, z[i - 1])
        else:
            change = calc_reg_deg(z_deg_rate, z[i - 1])
        z[i] += z[i - 1] + change * dt
    # z_prod = calc_regular_production(z_max_rate, z_deg_rate, z_prod_time_vals - z_prod_time_vals[0])
    # z_deg_time = time_vals[((x > kxz) | (y < kyz)) & (time_vals > z_prod_time_vals[-1])]
    # z_deg = calc_regular_degradation(z_prod[-1], z_deg_rate, z_deg_time - z_deg_time[0])
    # no_z = np.zeros((np.count_nonzero(time_vals < z_prod_time_vals[0]),))
    # z = np.hstack((no_z, z_prod, z_deg))
    signal = np.where((sig_start_step < np.arange(time_vals.size)) & (np.arange(time_vals.size) < sig_stop_step),
                      np.max(np.hstack((x, y, z))) * 1.2, 0.)
    data = pd.DataFrame(np.vstack((x, y, z, signal)).T, columns=["X", "Y", "Z", "signal"])
    plot_concentrations(data, "concentration", "q4a.png", False)





if __name__ == "__main__":
    kd = 0.8
    hill_coef = 3
    max_rate = 5
    degradation_rate = 3
    kyz_4a = 0.5
    kxy_4b = 0.4
    kxz_4b = 0.65
    # q1a(kd, hill_coef, max_rate, degradation_rate)
    # q1b(kd, max_rate, degradation_rate)
    q4a(kyz_4a, max_rate, degradation_rate, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4b(kyz_4a, kxy_4b, kxz_4b, max_rate / 1.5, degradation_rate * 2, max_rate, degradation_rate * 1.5, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4c(kyz_4a, kxy_4b, kxz_4b, max_rate / 1.5, degradation_rate * 2, max_rate, degradation_rate * 1.5, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4d(kyz_4a, kxy_4b, kxz_4b, max_rate / 1.5, degradation_rate * 2, max_rate, degradation_rate * 1.5, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4e(kyz_4a, kxy_4b, kxz_4b, max_rate*1.5, degradation_rate * 2, hill_coef, kd, max_rate, degradation_rate * 1.5, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
