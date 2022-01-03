import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff

DX = 0.05
DY = 0.05
MIN_X = -1.0
MAX_X = 1.0
MIN_Y = -0.5
MAX_Y = 0.6


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


if __name__ == "__main__":
    make_plots(MIN_X, MIN_Y, MAX_X, MAX_Y, DX, DY, func1, func2, r"graphs\q1.png", True)
