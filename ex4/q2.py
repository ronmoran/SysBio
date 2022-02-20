import pandas as pd
import numpy as np
import re
from typing import Iterable
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import plotly.express as px

RNA_LEVELS_FILE_PATH = r"Exercise 4 files\rna_levels.xlsx"
COL_RE = re.compile("([0-9.]+) hr")


def read_rna_levels_to_pd_df(path: str) -> pd.DataFrame:
    data = pd.read_excel(path, index_col=0)
    data.columns = transform_columns(data.columns)
    return data


def transform_columns(columns: Iterable[str]):
    return [float(COL_RE.match(col_name.strip()).group(1)) * 60 for col_name in columns]


def production_base_function(time: np.ndarray, production_rate: float, removal_rate: float,
                             initial_quantity: float) -> np.ndarray:
    return initial_quantity * np.exp(-removal_rate * time) + \
           (production_rate / removal_rate) * (1 - np.exp(-removal_rate * time))


# solved by https://www.wolframalpha.com/input?i2d=true&i=x%27%5C%2840%29t%5C%2841%29%3DDivide%5Ba%2Ca%2Bb%5D*Divide%5B1%2CV%5D*c+%2B+Divide%5Bb%2Ca%2Bb%5D*Divide%5B1%2CV%5D*d-l*x%5C%2840%29t%5C%2841%29
def production_two_step_function(t, k_on, k_off, prod_on, prod_off, binding_speed, removal_rate, initial_amount):
    return (k_on * prod_on + k_off * prod_off) / ((k_on + k_off) * removal_rate * binding_speed) * (
                1 - np.exp(-removal_rate * t)) + initial_amount * np.exp(-removal_rate * t)


def find_parameters(time_values: np.ndarray, rna_measurements: np.ndarray, production_function,
                    initial_guess: np.ndarray = None) -> (
        np.ndarray, np.ndarray):
    bounds = (0, np.inf)
    if initial_guess is None:
        popt, pcov = curve_fit(production_function, time_values, rna_measurements, bounds=bounds)
    else:
        popt, pcov = curve_fit(production_function, time_values, rna_measurements, p0=initial_guess, bounds=bounds)
    return popt


def meanify_samples(data: pd.DataFrame):
    unique_indices = data.index.unique()
    meaned_samples = pd.DataFrame(columns=data.columns)
    for index in unique_indices:
        mean_index_vals: pd.DataFrame = data.loc[index].mean().to_frame(index).transpose()
        meaned_samples = meaned_samples.append(mean_index_vals)
    return meaned_samples


def round_pretty_print(data: pd.DataFrame, decimals=2):
    print(data.round(decimals).to_markdown())


def estimate_rna_parameters_base(meaned_samples: pd.DataFrame):
    params = pd.Series(["Production", "Removal", "Initial"])
    estimated_params = pd.DataFrame(columns=params, index=meaned_samples.index)
    for gene_name in meaned_samples.index:
        params = find_parameters(meaned_samples.columns.to_numpy(), meaned_samples.loc[gene_name],
                                 production_base_function)
        estimated_params.loc[gene_name] = params
    return estimated_params.astype(float)


def estimate_rna_parameters_two_step(meaned_samples: pd.DataFrame):
    params = pd.Series(["K_on", "K_off", "Prod_on", "Prod_off", "Binding Speed", "Removal", "Initial Amount"])
    estimated_params = pd.DataFrame(columns=params, index=meaned_samples.index)
    for gene_name in meaned_samples.index:
        params = find_parameters(meaned_samples.columns.to_numpy(), meaned_samples.loc[gene_name],
                                 production_two_step_function)
        estimated_params.loc[gene_name] = params
    return estimated_params.astype(float)


def estimate_rmse(labeled_mean_data: pd.DataFrame, estimated_params: pd.DataFrame, prod_func) -> pd.DataFrame:
    col_name = "RMSE"
    time_vals = labeled_mean_data.columns
    rmse_df = pd.DataFrame(data=np.empty((labeled_mean_data.shape[0], 1)), columns=[col_name],
                           index=labeled_mean_data.index)
    for index, row in labeled_mean_data.iterrows():
        rmse_df.loc[index] = mean_squared_error(row,
                                                prod_func(time_vals.to_numpy(),
                                                          *estimated_params.loc[index]), squared=False)
    return rmse_df.sort_values(col_name)


def draw_estimated(meaned_data: pd.DataFrame, params: pd.DataFrame, func):
    t_start, t_end = meaned_data.columns[[0, -1]]
    num_points = int(1e4)
    time_vals = np.linspace(t_start, t_end, num_points)
    columns = ["Time(min)", "Count(#)", "Gene", "is_estimated"]  # do not change without reading thoroughly
    res = pd.DataFrame(columns=columns)
    for gene, row in params.iterrows():
        tmp = pd.DataFrame(columns=columns, index=range(num_points))
        estimated = func(time_vals, *row)
        tmp.iloc[:, 0] = time_vals
        tmp.iloc[:, 1] = estimated
        tmp.iloc[:, 2] = gene
        tmp.iloc[:, 3] = True
        sampled_transformed = pd.DataFrame(columns=columns, index=range(len(meaned_data.columns)))
        sampled_transformed.iloc[:, 0] = meaned_data.columns
        sampled_transformed.iloc[:, 1] = meaned_data.loc[gene].to_numpy()
        sampled_transformed.iloc[:, 2] = gene
        sampled_transformed.iloc[:, 3] = False
        res = pd.concat((res, tmp, sampled_transformed))
    fig = px.line(res, x=columns[0], y=columns[1], color=columns[3], facet_col=columns[2],
                     facet_col_wrap=2, markers=True)
    fig.update_yaxes(matches=None)
    fig.show()


def main(path):
    meaned_samples = meanify_samples(read_rna_levels_to_pd_df(path))
    params = estimate_rna_parameters_base(meaned_samples)
    round_pretty_print(params)
    rmse = estimate_rmse(meaned_samples, params, production_base_function)
    round_pretty_print(rmse)
    two_step_params = estimate_rna_parameters_two_step(meaned_samples)
    round_pretty_print(two_step_params)
    rmse_two_step = estimate_rmse(meaned_samples, two_step_params, production_two_step_function)
    round_pretty_print(rmse_two_step)
    draw_estimated(meaned_samples, params, production_base_function)
    draw_estimated(meaned_samples, two_step_params, production_two_step_function)


if __name__ == "__main__":
    main(RNA_LEVELS_FILE_PATH)
