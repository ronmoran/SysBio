import numpy as np
import plotly.express as px
import pandas as pd
import os
from functools import partial
from plotly.subplots import make_subplots
import plotly.graph_objects as go

DT = 1e-3
TOTAL_TIME = 2.5
PLOT_PATH = r"lyx_files\plots"
SAVE_PLOTS = True


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
    return max_rate * (kd ** hill_coef / (kd ** hill_coef + (current_protein_concentration ** hill_coef))) \
           - degradation_rate * current_protein_concentration


def simulate_regular_regulation(max_rate: float, degradation_rate: float, total_time: float, dt: float) -> \
        (np.ndarray, np.ndarray):
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    return time_vals, calc_regular_production(max_rate, degradation_rate, time_vals)


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
        productions_rates[i] = max_rate * (kd ** hill_coef / (kd ** hill_coef + concentration ** hill_coef))
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
    plt.show()
    if SAVE_PLOTS:
        plt.write_image(os.path.join(PLOT_PATH, fig_name), width=1280, height=720)


def plot_concentrations(data, variable, fig_name, add_half_concentration=True):
    plt = px.line(data,
                  title='Concentration In Time',
                  labels={
                      "index": "Time",
                      "value": "Concentration",
                      "variable": variable
                  }
                  )
    if add_half_concentration:
        plt.add_hline(y=data.iloc[-1][0] / 2, line_dash="dot", annotation_text="steady-state / 2", opacity=0.5)
        half_val = data.iloc[-1] / 2
        half_times = np.abs(data - half_val).idxmin(axis=0)
        for half_time in half_times:
            plt.add_vline(x=half_time, line_dash="dot", opacity=0.2, annotation_text="t 1/2",
                          annotation_position='bottom right')
    plt.update_xaxes(showticklabels=False)
    plt.show()
    if SAVE_PLOTS:
        plt.write_image(os.path.join(PLOT_PATH, fig_name), width=1280, height=720)


def plot_stacked_q4(data, fig_name, title, y_axis_titles, **kwargs):
    fig = make_subplots(rows=data.shape[0], cols=1, shared_xaxes=True, vertical_spacing=0.01)
    for i, dat in enumerate(data):
        fig.add_trace(go.Line(y=data[i]), row=i+1, col=1)
        fig.update_yaxes(title_text=y_axis_titles[i] + " con.", row=i+1, col=1)
    if "hlines" in kwargs:
        for attrs in kwargs["hlines"]:
            fig.add_hline(**attrs, annotation_position="top left", line_width=0.7, line_dash="dot")
    if "vlines" in kwargs:
        for attrs in kwargs["vlines"]:
            fig.add_vline(**attrs, line_width=0.3, line_dash="dot")
    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text='Time', row=data.shape[0])
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False)
    fig.show()
    if SAVE_PLOTS:
        fig.write_image(os.path.join(PLOT_PATH, fig_name), width=1280, height=720)


def q1a(kd, hill_coef, max_rate, degradation_rate):
    nar_data = simulate_negative_autoregulation(max_rate, degradation_rate, kd, hill_coef, TOTAL_TIME, DT)[1]
    reg_data = simulate_regular_regulation(2.45, degradation_rate, TOTAL_TIME, DT)[1]
    both_regulation_data = pd.DataFrame(np.array([nar_data, reg_data]).T, columns=['NAR', 'Regular regulation'])
    plot_concentrations(both_regulation_data, "Regulation Type", "q1a1.png")
    plot_rate_balance(degradation_rate, max_rate, kd, np.array([hill_coef]), 1, 1000, "q1a2.png")


def q1b(kd, max_rate, degradation_rate):
    hill_coefs = np.array([1, 4])
    hill_one = simulate_negative_autoregulation(max_rate, degradation_rate, kd, hill_coefs[0], TOTAL_TIME, DT)[1]
    hill_four = simulate_negative_autoregulation(max_rate, degradation_rate, kd, hill_coefs[1], TOTAL_TIME, DT)[1]
    both_regulation_data = pd.DataFrame(np.array([hill_one, hill_four]).T, columns=hill_coefs.astype(str))
    plot_concentrations(both_regulation_data, "Hill Coefficient", "q1b1.png")
    plot_rate_balance(degradation_rate, max_rate, kd, hill_coefs, 1.5, 1000, "q1b2.png")


def _q4_calc_iteratively(prod_bool_vec, bound_prod_change_func, bound_deg_change_func, dt, init_val=0.):
    vals = np.empty(prod_bool_vec.shape, dtype=float)
    vals[0] = init_val
    for i in range(1, len(vals)):
        prev = vals[i - 1]
        if prod_bool_vec[i]:
            change = bound_prod_change_func(prev)
        else:
            change = bound_deg_change_func(prev)
        vals[i] = prev + change * dt
    return vals


def _prep_reg_paritals(max_prod_rate, deg_rate):
    prod = partial(calc_reg_change, max_prod_rate, deg_rate)
    deg = partial(calc_reg_deg, deg_rate)
    return prod, deg


def _prep_nar_partials(max_prod_rate, deg_rate, hill_coef, kd):
    prod = partial(calc_nar_change, deg_rate, hill_coef, kd, max_prod_rate)
    deg = partial(calc_reg_deg, deg_rate)
    return prod, deg


def _create_reg_data_series(prod_bool, max_rate, deg_rate, dt, init_val=0.):
    prod_func, deg_func = _prep_reg_paritals(max_rate, deg_rate)
    return _q4_calc_iteratively(prod_bool, prod_func, deg_func, dt, init_val)


def _create_nar_data_series(prod_bool, max_rate, deg_rate, hill_coef, kd, dt):
    prod_func, deg_func = _prep_nar_partials(max_rate, deg_rate, hill_coef, kd)
    return _q4_calc_iteratively(prod_bool, prod_func, deg_func, dt)


def q4a(kyz, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time / dt + 2 < sig_stop_time / dt < total_time / dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    signal_time_vals = (time_vals > sig_start_time) & (time_vals < sig_stop_time)
    y = _create_reg_data_series(signal_time_vals, y_max_rate, y_deg_rate, dt)
    z_prod_time_vals = ((y > kyz) & signal_time_vals)
    z = _create_reg_data_series(z_prod_time_vals, z_max_rate, z_deg_rate, dt)
    hlines = [dict(y=kyz, row=2, annotation_text="$K_{YZ}$")]
    vlines = [dict(x=z_prod_time_vals.argmax())]
    signal = np.where(signal_time_vals, np.max(np.hstack((y, z))) * 1.1, 0.)
    data = np.array([signal, y, z])
    plot_stacked_q4(data, "q4a.png", "X Saturated, AND Gate for Y Level and Signal",
                    y_axis_titles=["signal = X", "Y", "Z"], hlines=hlines, vlines=vlines)


def q4b(kyz, kxy, kxz, x_max_rate, x_deg_rate, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time,
        sig_stop_time, total_time, dt):
    assert sig_start_time / dt + 2 < sig_stop_time / dt < total_time / dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    signal_time_vals = (time_vals > sig_start_time) & (time_vals < sig_stop_time)
    x = _create_reg_data_series(signal_time_vals, x_max_rate, x_deg_rate, dt)
    y_prd_time = x > kxy
    y = _create_reg_data_series(y_prd_time, y_max_rate, y_deg_rate, dt)
    z_prod_time_vals = ((y > kyz) & (x > kxz))
    z = _create_reg_data_series(z_prod_time_vals, z_max_rate, z_deg_rate, dt)
    hlines = [dict(y=kyz, row=3, annotation_text="$K_{YZ}$", ),
              dict(y=kxy, row=2,  annotation_text="$K_{XY}$"),
              dict(y=kxz, row=2, annotation_text="$K_{XZ}$")]
    vlines = [dict(x=np.where(y == 0., np.inf, y).argmin()),
              dict(x=np.where(z == 0., np.inf, z).argmin()),
              dict(x=y.argmax()),
              dict(x=z.argmax())]
    signal = np.where(signal_time_vals, np.max(np.hstack((x, y, z))) * 1.1, 0.)
    data = np.array([signal, x, y, z])
    plot_stacked_q4(data, "q4b.png", "X Buildup Upon Signal, AND Gate", y_axis_titles=["signal", "X", "Y", "Z"],
                    hlines=hlines, vlines=vlines)


def q4c(kyz, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time / dt + 2 < sig_stop_time / dt < total_time / dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    signal_time_vals = (time_vals > sig_start_time) & (time_vals < sig_stop_time)
    y = _create_reg_data_series(signal_time_vals, y_max_rate, y_deg_rate, dt)
    z_prod_time_vals = ((y > kyz) | signal_time_vals)
    z = _create_reg_data_series(z_prod_time_vals, z_max_rate, z_deg_rate, dt)
    hlines = [dict(y=kyz, row=2, annotation_text="$K_{YZ}$")]
    vlines = [dict(x=z.argmax())]
    signal = np.where(signal_time_vals, np.max(np.hstack((y, z))) * 1.1, 0.)
    data = np.array([signal, y, z])
    plot_stacked_q4(data, "q4c.png", "Saturated X Activated Upon signal, OR gate",
                    y_axis_titles=["signal = X", "Y", "Z"], hlines=hlines, vlines=vlines)


def q4d(kyz, kxy, kxz, x_max_rate, x_deg_rate, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate, sig_start_time,
        sig_stop_time, total_time, dt):
    assert sig_start_time / dt + 2 < sig_stop_time / dt < total_time / dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    signal_time_vals = (time_vals > sig_start_time) & (time_vals < sig_stop_time)
    x = _create_reg_data_series(signal_time_vals, x_max_rate, x_deg_rate, dt)
    y_prd_time = x > kxy
    y = _create_reg_data_series(y_prd_time, y_max_rate, y_deg_rate, dt)
    z_prod_time_vals = ((y > kyz) & (x < kxz))
    z = _create_reg_data_series(z_prod_time_vals, z_max_rate, z_deg_rate, dt)
    z_changes = ((z_prod_time_vals ^ np.roll(z_prod_time_vals, 1))[1:-1]).nonzero()[0]
    y_changes = ((y_prd_time ^ np.roll(y_prd_time, 1))[1:-1]).nonzero()[0]
    hlines = [dict(y=kyz, row=3, annotation_text="$K_{YZ}$"),
              dict(y=kxy, row=2, annotation_text="$K_{XY}$"),
              dict(y=kxz, row=2, annotation_text="$K_{XZ}$")]
    vlines = [dict(x=change) for change in np.hstack((z_changes, y_changes))]
    signal = np.where(signal_time_vals, np.max(np.hstack((x, y, z))) * 1.1, 0.)
    data = np.array([signal, x, y, z])
    plot_stacked_q4(data, "q4d.png", "Incoherent Type 3, X Buildup Upon Signal",
                    y_axis_titles=["signal", "X", "Y", "Z"], hlines=hlines, vlines=vlines)


def q4e(kyz, kxy, kxz, x_max_rate, x_deg_rate, hill_coef, kd, y_max_rate, y_deg_rate, z_max_rate, z_deg_rate,
        sig_start_time, sig_stop_time, total_time, dt):
    assert sig_start_time / dt + 2 < sig_stop_time / dt < total_time / dt
    time_vals = np.linspace(0, total_time, num=int(total_time / dt))
    signal_time_vals = (time_vals > sig_start_time) & (time_vals < sig_stop_time)
    x = _create_nar_data_series(signal_time_vals, x_max_rate, x_deg_rate, hill_coef, kd, dt)
    y_prd_time = x > kxy
    y = _create_reg_data_series(y_prd_time, y_max_rate, y_deg_rate, dt)
    z_prod_time_vals = ((y < kyz) & (x < kxz))
    z = _create_reg_data_series(z_prod_time_vals, z_max_rate, z_deg_rate, dt, z_max_rate / z_deg_rate)
    z_changes = ((z_prod_time_vals ^ np.roll(z_prod_time_vals, 1))[1:-1]).nonzero()[0]
    y_changes = ((y_prd_time ^ np.roll(y_prd_time, 1))[1:-1]).nonzero()[0]
    hlines = [dict(y=kyz, row=3, annotation_text="$K_{YZ}$"),
              dict(y=kxy, row=2, annotation_text="$K_{XY}$"),
              dict(y=kxz, row=2, annotation_text="$K_{XZ}$")]
    vlines = [dict(x=change) for change in np.hstack((z_changes, y_changes))]
    signal = np.where(signal_time_vals, np.max(np.hstack((x, y, z))) * 1.1, 0.)
    data = np.array([signal, x, y, z])
    plot_stacked_q4(data, "q4e.png", "Coherent Type 3 With Negative Auto-Regulation on X",
                    y_axis_titles=["signal", "X", "Y", "Z"], hlines=hlines, vlines=vlines)


if __name__ == "__main__":
    kd = 0.8
    hill_coef = 3
    max_rate = 5
    degradation_rate = 3
    kyz_4a = 0.5
    kxy_4b = 0.4
    kxz_4b = 0.65
    q1a(kd, hill_coef, max_rate, degradation_rate)
    q1b(kd, max_rate, degradation_rate)
    q4a(kyz_4a, max_rate, degradation_rate, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4b(kyz_4a * 1.2, kxy_4b, kxz_4b / 3, max_rate / 1.5, degradation_rate * 2, max_rate / 2, degradation_rate * 0.8, max_rate/ 1.3, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4c(kyz_4a, max_rate, degradation_rate * 1.5, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4d(kyz_4a / 2, kxy_4b / 4, kxz_4b - 0.2, max_rate / 1.5, degradation_rate * 2, max_rate, degradation_rate * 1.5, max_rate, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
    q4e(kyz_4a, kxy_4b, kxz_4b, max_rate * 1.5, degradation_rate * 2, hill_coef, kd, max_rate / 1.5, degradation_rate,
         max_rate / 3, degradation_rate, 0.2, 1.2, TOTAL_TIME, DT)
