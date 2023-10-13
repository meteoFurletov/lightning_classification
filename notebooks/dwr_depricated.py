### Устаревшие функции, больше не использующеися.

from PIL import Image
from collections import Counter
import joblib
import gc
import time
from time import mktime
from datetime import datetime
import gzip
from functools import reduce
from IPython.display import display
from ipywidgets import interact, IntSlider
from tqdm.contrib.itertools import product
import seaborn as sns
import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
import re
import os
from pathlib import Path
import duckdb

from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use("seaborn")
mpl.rcParams["font.family"] = "serif"

phen_palette = {
    1: "#AAFFFF",
    2: "#000000",
    3: "#00FF2A",
    4: "#6060FF",
    6: "#FF7070",
    7: "#FF60FF",
    11: "#60FFFF",
    13: "#00D523",
    14: "#4040FF",
    16: "#FF4040",
    17: "#FF40FF",
    23: "#00B41E",
    24: "#0000FF",
    26: "#FF0000",
    27: "#FF00FF",
}


class Paths:
    main = "/home/meteofurletov/iram/lightning/"
    data = "/home/meteofurletov/iram/lightning/data/"
    binary = "/home/meteofurletov/iram/lightning/data/binary data/"
    lightning = "/home/meteofurletov/iram/lightning/data/data_BO/"
    test = "/home/meteofurletov/iram/lightning/data/test_data/"
    models = "/home/meteofurletov/iram/lightning/models/"
    dem = "/home/meteofurletov/iram/lightning/data/ASTER/tiff/"
    interim = "/home/meteofurletov/iram/lightning/data/interim/"
    numpy_data = "/home/meteofurletov/iram/lightning/data/numpy_data/"


def delete_unmatched_bo_data(data_bo, matched_list):
    """Not used"""
    matched_list_radar_over_bo = []
    for i in range(len(matched_list)):
        date = (
            matched_list[i][-21:-11] + " " + (matched_list[i][-9:-4]).replace("_", ":")
        ).replace("_", "-")
        matched_list_radar_over_bo.append(pd.to_datetime(date))
    data_bo.date = pd.to_datetime(data_bo.date)
    data_bo_cut = data_bo[data_bo.date.isin(matched_list_radar_over_bo)]
    final_arr_bo = bo_matrix(data_bo_cut, 200)
    return final_arr_bo


def filter_0(true_radar_data, threshold, param):
    """Computes mean reflectivity or zdr in central pixel of array
    and delete if it is lower than threshold

    Return: input array without deleted values,
            indices array of undeleted elements"""
    filter_0_indices_list = []

    if param == "reflectivity":
        true_radar_center_data = true_radar_data[
            :, true_radar_data.shape[1] // 2, true_radar_data.shape[2] // 2, 0:11
        ].compute()
        for i in range(true_radar_center_data.shape[0]):
            if np.max(true_radar_center_data[i]) >= threshold:
                filter_0_indices_list.append(i)
    # elif param == 'zdr':
    #   true_radar_center_data = true_radar_data[
    #                         :, true_radar_data.shape[1] // 2, true_radar_data.shape[2] // 2, 11:22].compute()

    elif param == "echotop":
        true_radar_center_data = (
            true_radar_data[
                :, true_radar_data.shape[1] // 2, true_radar_data.shape[2] // 2, 11
            ].compute()
            * 250
        )
        for i in range(true_radar_center_data.shape[0]):
            if true_radar_center_data[i] >= threshold:
                filter_0_indices_list.append(i)

    return true_radar_data[filter_0_indices_list], filter_0_indices_list


def vil(radar_array, batch):
    """Computes VIL but not correctly"""
    vil_list = []
    for i in tqdm(range(0, radar_array.shape[0], batch)):
        mini_batch = radar_array[i : batch + i, :, :, 0:11]
        computed_batch = mini_batch.compute()
        computed_batch[computed_batch == -100] = 0
        iterable = (i for i in range(10))
        iter_arr = np.fromiter(iterable, dtype=np.intp)
        for x in range(batch):
            a = (
                computed_batch[x, :, :, iter_arr]
                + computed_batch[x, :, :, iter_arr + 1]
            ) / 2
            b = np.sum((np.sign(a) * (np.abs(a)) ** (4 / 7)), axis=0)
            vil_arr = 3.44e-6 * b * 1000
            vil_list.append(vil_arr)
    return np.array(vil_list)


def get_npy_file_from_images(matched_list):
    """Return an array from all npy files in given directory"""
    train_images = []
    for elem in tqdm(matched_list):
        data = np.array(Image.open(elem))
        train_images.append(data)
    return np.array(train_images)


def scorer_bold_attempt(y_true, y_pred) -> None:
    """Вычисляет следующие метрики:
    Acc, POD, FNR, FPR, F1, roc_auc, ETS, conf_matrix

    Args:
        y_true (np.array): вектор фактических значений для теста
        y_pred (np.array): вектор предсказаний

    Returns:
        plot of confusion matrix
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tn + fp + fn + tp)
    recall = tp / (tp + fn)  # POD
    precision = tp / (tp + fp)

    fpr = fp / (fp + tp)
    fnr = fn / (tp + fn)
    f_measure = 2 * ((precision * recall) / (precision + recall))

    num = (tp + fp) * (tp + fn)
    den = tp + fn + fp + tn
    dr = num / den
    ets = (tp - dr) / (tp + fn + fp - dr)

    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[1, 0]))
    disp.plot()

    print(f"Accuracy is {accuracy}")
    print(f"POD is {recall}")
    print(f"FNR is {fnr}")
    print(f"FPR is {fpr}")
    print(f"F-Measure is {f_measure}")
    print(f"roc_auc_score is {roc_auc_score(y_true, y_pred)}")
    print(f"ETS is {ets}")


def pct_method(data, level):
    # Upper and lower limits by percentiles
    upper = np.percentile(data, 100 - level)
    lower = np.percentile(data, level)
    # Returning the upper and lower limits
    return [lower, upper]


def iqr_method(data):
    # Calculating the IQR
    perc_75 = np.percentile(data, 75)
    perc_25 = np.percentile(data, 25)
    iqr_range = perc_75 - perc_25
    # Obtaining the lower and upper bound
    iqr_upper = perc_75 + 1.5 * iqr_range
    iqr_lower = perc_25 - 1.5 * iqr_range
    # Returning the upper and lower limits
    return [iqr_lower, iqr_upper]


def std_method(data, sigma=3):
    # Creating three standard deviations away boundaries
    std = np.std(data)
    upper_3std = np.mean(data) + sigma * std
    lower_3std = np.mean(data) - sigma * std
    # Returning the upper and lower limits
    return [lower_3std, upper_3std]


def threshold_function(probas, threshold):
    classes = probas > threshold
    return classes


def plot_case(index, df, reflect_18, echotop_plot) -> None:
    """Plot one case"""
    plt.rcParams["axes.grid"] = False
    coor_x = df.loc[index, "X(2,2)"]
    coor_y = df.loc[index, "Y(2,2)"]
    time = df.loc[index, "datetime"]
    print(coor_y, coor_x, time)

    if len(str(time.minute)) == 1:
        minutes = "0" + str(time.minute)
    else:
        minutes = time.minute

    if len(str(time.hour)) == 1:
        hours = "0" + str(time.hour)
    else:
        hours = time.hour

    if len(str(time.day)) == 1:
        days = "0" + str(time.day)
    else:
        days = time.day

    if len(str(time.month)) == 1:
        months = "0" + str(time.month)
    else:
        months = time.month

    full = (
        np.load(
            Path.data
            + f"\echo_top\26061_{time.year}_{months}_{days}__{hours}_{minutes}.npy"
        )
        * 250
    )

    data_plot_0 = np.round(np.max(reflect_18[index, :, :, :], axis=2), 2)
    data_plot_1 = np.round(echotop_plot[index, :, :], 2)
    data_plot_2 = full[coor_y - 5 : coor_y + 5, coor_x - 5 : coor_x + 5] / 1000
    data_plot_3 = full

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(data_plot_0)
    for (j, i), label in np.ndenumerate(data_plot_0):
        axes[0, 0].text(i, j, label, ha="center", va="center")

    axes[0, 1].imshow(data_plot_1)
    for (j, i), label in np.ndenumerate(data_plot_1):
        axes[0, 1].text(i, j, label, ha="center", va="center")

    axes[1, 0].imshow(data_plot_2)
    for (j, i), label in np.ndenumerate(data_plot_2):
        axes[1, 0].text(i, j, label, ha="center", va="center")

    ticks = np.arange(0, 200, 50)

    im3 = axes[1, 1].imshow(data_plot_3)
    fig.colorbar(im3, orientation="vertical")
    rect = patches.Rectangle(
        (coor_x - 5, coor_y - 5), 10, 10, linewidth=1, edgecolor="r", facecolor="none"
    )
    axes[1, 1].add_patch(rect)
    axes[1, 1].set_yticks(ticks)
    axes[1, 1].set_xticks(ticks)
    axes[1, 1].grid()


def plot_6(data_to_plot, title) -> None:
    """Plot 6 graphs with info about: x_coor, y_coor, H max, Rmax, month"""
    plt.rcParams["axes.grid"] = True
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=20)

    axes[0, 0].hist(data_to_plot["x_coor"], bins=np.unique(data_to_plot["x_coor"]))
    axes[0, 0].set_title("Center X")

    axes[1, 0].hist(
        data_to_plot["y_coor"], bins=np.unique(data_to_plot["y_coor"]).shape[0]
    )
    axes[1, 0].set_title("Center Y")

    axes[0, 1].hist(data_to_plot["hmax"], bins=np.unique(data_to_plot["hmax"]))
    axes[0, 1].set_title("H max")

    axes[1, 1].hist(data_to_plot["rmax"], bins=np.unique(data_to_plot["rmax"]))
    axes[1, 1].set_title("R max")

    axes[1, 2].hist(
        data_to_plot["delta_iso22"], bins=np.unique(data_to_plot["hmax"]).shape[0]
    )
    axes[1, 2].set_title("H max - isotherm 22")

    axes[0, 2].hist(
        pd.DatetimeIndex(np.array(data_to_plot["date"], dtype="datetime64[m]")).month
    )
    axes[0, 2].set_xticks(
        range(
            pd.DatetimeIndex(
                np.array(data_to_plot["date"], dtype="datetime64[m]")
            ).month.min(),
            pd.DatetimeIndex(
                np.array(data_to_plot["date"], dtype="datetime64[m]")
            ).month.max()
            + 1,
        )
    )
    axes[0, 2].set_title("Month")
    plt.rcParams["axes.grid"] = False


def plot_6_sns(data_to_plot, title) -> None:
    """Plot 6 graphs with info about: x_coor, y_coor, H max, Rmax, month
    In seaborn style with difference between true and false"""
    sns.set()
    fig, axes = plt.subplots(2, 3, figsize=(24, 9))
    fig.suptitle(title, fontsize=20)
    for ax in axes.flatten():
        ax.set_yscale("log")

    sns.histplot(
        ax=axes[0, 0],
        data=data_to_plot,
        x="x_coor",
        hue="target",
        bins=np.unique(data_to_plot["x_coor"]),
        legend=False,
    )
    axes[0, 0].set_title("Center X")

    sns.histplot(
        ax=axes[1, 0],
        data=data_to_plot,
        x="y_coor",
        hue="target",
        bins=np.unique(data_to_plot["y_coor"]).shape[0],
        legend=False,
    )
    axes[1, 0].set_title("Center Y")

    sns.histplot(
        ax=axes[0, 1],
        data=data_to_plot,
        x="hmax",
        hue="target",
        bins=np.unique(data_to_plot["hmax"]),
        legend=False,
    )
    axes[0, 1].set_title("H max")

    sns.histplot(
        ax=axes[1, 1],
        data=data_to_plot,
        x="rmax",
        hue="target",
        bins=np.unique(data_to_plot["rmax"]),
        legend=False,
    )
    axes[1, 1].set_title("R max")

    sns.histplot(
        ax=axes[1, 2],
        data=data_to_plot,
        x="delta_iso22",
        hue="target",
        bins=np.unique(data_to_plot["hmax"]).shape[0],
        legend=False,
    )
    axes[1, 2].set_title("H max - isotherm 22")

    sns.histplot(
        ax=axes[0, 2],
        data=data_to_plot,
        x=pd.DatetimeIndex(np.array(data_to_plot["date"], dtype="datetime64[m]")).month,
        hue="target",
        shrink=20,
        legend=False,
    )
    axes[0, 2].set_title("Month")


def freeze_header(df, num_rows=30, num_columns=10, step_rows=1, step_columns=1) -> None:
    """
    Freeze the headers (column and index names) of a Pandas DataFrame. A widget
    enables to slide through the rows and columns.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to display
    num_rows : int, optional
        Number of rows to display
    num_columns : int, optional
        Number of columns to display
    step_rows : int, optional
        Step in the rows
    step_columns : int, optional
        Step in the columns

    Returns
    -------
    Displays the DataFrame with the widget
    """

    @interact(
        last_row=IntSlider(
            min=min(num_rows, df.shape[0]),
            max=df.shape[0],
            step=step_rows,
            description="rows",
            readout=False,
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            slider_color="purple",
        ),
        last_column=IntSlider(
            min=min(num_columns, df.shape[1]),
            max=df.shape[1],
            step=step_columns,
            description="columns",
            readout=False,
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            slider_color="purple",
        ),
    )
    def _freeze_header(last_row, last_column) -> None:
        display(
            df.iloc[
                max(0, last_row - num_rows) : last_row,
                max(0, last_column - num_columns) : last_column,
            ]
        )
