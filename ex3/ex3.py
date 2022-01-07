import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots

DX = 0.05
DY = 0.05
MIN_X = -1.0
MAX_X = 1.0
MIN_Y = -0.5
MAX_Y = 0.6

DT = 0.01

SHOW_FIG = True

def make_meshgrid(min_x, min_y, max_x, max_y, step_x, step_y):
    range_x = np.arange(min_x, max_x, step_x)
    range_y = np.arange(min_y, max_y, step_y)
    return np.meshgrid(range_x, range_y)


def func1(x, y, name) -> (np.ndarray, go.Scatter):
    return 3 * y - 2 * x, go.Scatter(x=x[0], y=2 * x[0] / 3, name=name)


def func2(x, y, name) -> (np.ndarray, go.Scatter):
    return x ** 2 - y, go.Scatter(x=x[0], y=x[0] ** 2, name=name)


def make_plots(min_x, min_y, max_x, max_y, step_x, step_y, x_func, y_func, graph_path,
               norm=True):
    x, y = make_meshgrid(min_x, min_y, max_x, max_y, step_x, step_y)
    x_func_vals, null_cline1 = x_func(x, y, "f(x)")
    y_func_vals, null_cline2 = y_func(x, y, "g(x)")
    if norm:
        vec_len = (x_func_vals ** 2 + y_func_vals ** 2) ** 0.5
        x_func_vals /= vec_len ** 0.5
        y_func_vals /= vec_len ** 0.5
    fig = ff.create_quiver(x, y, x_func_vals, y_func_vals, scale=0.025,
                           )
    fig.data[0].showlegend = False
    fig.add_traces((null_cline1, null_cline2))
    fig.update_layout(yaxis_range=[min_y, max_y], xaxis_range=[min_x, max_x], title="Quiver plot and nullclines",
                      width=1080, height=720, xaxis_title="X", yaxis_title="Y", legend={"font": {"size": 12}})
    intersect1 = go.Scatter(x=[0], y=[0], mode="markers", marker={"size": 10, "color": "blue"},
                            name="Stable\nequilibrium")
    intersect2 = go.Scatter(x=[2 / 3], y=[4 / 9], mode="markers", marker={"size": 10, "color": "gray"},
                            name="Unstable\nequilibrium")
    fig.add_vline(x=0)
    fig.add_hline(y=0)
    fig.update_annotations()
    fig.add_traces((intersect1, intersect2))
    fig.write_image(graph_path)


def calc_x_change(x, a1, b1, y):
    return DT * (x * (1 - x) - a1 * x / (1 + b1 * x) * y)


def calc_y_change(x, a1, b1, y, d1, a2, b2, z):
    return DT * (a1 * x / (1 + b1 * x) * y - d1 * y - a2 * y / (1 + b2 * y) * z)


def calc_z_change(y, a2, b2, d2, z):
    return DT * (a2 * y / (1 + b2 * y) * z - d2 * z)


def iterate_foodchain_model(total_time, a1, b1, a2, b2, d1, d2, x0, y0, z0):
    num_iterations = int(total_time / DT)
    x_vec = np.empty(num_iterations)
    x_vec[0] = x0
    y_vec = np.empty(num_iterations)
    y_vec[0] = y0
    z_vec = np.empty(num_iterations)
    z_vec[0] = z0
    for i in range(1, num_iterations):
        x_vec[i] = x_vec[i - 1] + calc_x_change(x_vec[i - 1], a1, b1, y_vec[i - 1])
        y_vec[i] = y_vec[i - 1] + calc_y_change(x_vec[i - 1], a1, b1, y_vec[i - 1], d1, a2, b2, z_vec[i - 1])
        z_vec[i] = z_vec[i - 1] + calc_z_change(y_vec[i - 1], a2, b2, d2, z_vec[i - 1])
    return x_vec, y_vec, z_vec


def draw_foodchain(x_vec, y_vec, z_vec, output_path=None, show_fig=SHOW_FIG):
    fig = px.line_3d(x=x_vec, y=y_vec, z=z_vec, height=1440, width=2160)
    if SHOW_FIG:
        fig.show()
    if output_path is not None:
        fig.write_image(output_path)


def plot_time_series_stacked(x_series, y_series, z_series, output_path=None, show_fig=SHOW_FIG,  **consts):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01,
                        x_title="Time")
    fig.update_xaxes(showticklabels=False)
    fig.add_trace(row=1, col=1, trace=go.Scatter(y=x_series))
    fig.add_trace(row=2, col=1, trace=go.Scatter(y=y_series))
    fig.add_trace(row=3, col=1, trace=go.Scatter(y=z_series))
    fig.update_yaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=2, col=1)
    fig.update_yaxes(title_text="Z", row=3, col=1)
    fig.update_layout(showlegend=False, title_text=f"{', '.join([k + '=' + str(v) for k, v in consts.items()])}")
    if show_fig:
        fig.show()
    if output_path is not None:
        fig.write_image(output_path)


def q3(total_time, upto_fac=4, out_path=None, **consts):
    x_vals, y_vals, z_vals = iterate_foodchain_model(total_time, **consts)
    draw_foodchain(x_vals, y_vals, z_vals, out_path + "_3d.jpg")
    upto = int(total_time / DT / upto_fac)
    plot_time_series_stacked(x_vals[:upto], y_vals[:upto], z_vals[:upto], out_path + "_time_plot.jpg", **consts)


if __name__ == "__main__":
    # make_plots(MIN_X, MIN_Y, MAX_X, MAX_Y, DX, DY, func1, func2, r"graphs\q1.png", True)
    total_time = 1e4
    q3(total_time, a1=5, b1=3, a2=0.1, b2=2, d1=0.4, d2=0.01, x0=1, y0=0.2, z0=8, out_path=r"graphs\default")  # keep regular (chaos)
    q3(total_time, a1=5, b1=3, a2=0.1, b2=2, d1=0.4, d2=0.01, x0=1, y0=1, z0=2, out_path=r"graphs\different_init_values") # keep initial values dont matter (chaos)
    q3(total_time, a1=5, b1=30, a2=0.1, b2=20, d1=0.4, d2=0.01, x0=1, y0=0.2, z0=8, out_path=r"graphs\quick_stead_state")  # quick steady state, big b values
    q3(total_time, a1=5, b1=1.3, a2=0.1, b2=1.5, d1=0.4, d2=0.01, x0=1, y0=0.2, z0=8, out_path=r"graphs\stable_after_oscillations")  # stablizes not before oscillating, small b values
    q3(total_time, a1=11.5, b1=3, a2=0.08, b2=1.5, d1=0.4, d2=0.01, x0=1, y0=0.2, z0=8, out_path=r"graphs\chaos_z_decrease")  # chaos, but cyclic
    q3(total_time, upto_fac=2, a1=9, b1=3, a2=0.8, b2=2, d1=2, d2=0.01, x0=1, y0=0.2, z0=8, out_path=r"graphs\late_equilibrium")  # Large a values also lead to equil, albeit interesting
