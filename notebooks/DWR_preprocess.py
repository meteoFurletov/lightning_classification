from collections import Counter
import joblib
import gc
import time
from time import mktime
from datetime import datetime
import gzip
from functools import reduce
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


def onlyfiles(path):
    """Возвращает список всех файлов в указанной директории

    Args:
        path (str): Путь к директории

    Returns:
        list: Все файлы в указанной директории
    """
    onlyfiles_list = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            onlyfiles_list.append(os.path.join(path, name))
    return onlyfiles_list


def deg2dec(deg):
    """Перевод из градусов с минутами, секундами в градусы с десятичными значениями

    Args:
        deg (str): фомрат к примеру '''59°55'18"N'''

    Returns:
        float: градусы с десятичными значениями
    """
    lat = deg
    deg, minutes, seconds, direction = re.split("[°'\"]", lat)
    return np.round(
        (float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60))
        * (-1 if direction in ["W", "S"] else 1),
        5,
    )


def length_of_a_degree_of_longitude(latitude):
    """Расчитывает длину одного градуса долготы
    на заданной широте для земли в километрах

    Args:
        latitude (float): Значение широты с десятичной частью

    Returns:
        float: Возвращает длину одного градуса долготы в км
    """
    a = 6378137.0
    b = 6356752.3142
    e_sq = (a**2 - b**2) / a**2

    return (np.pi * 6378.1370 * np.cos(np.radians(latitude))) / (
        180 * np.sqrt(1 - e_sq * np.sin(np.radians(latitude)) ** 2)
    )


def length_of_a_degree_of_latitude(longitude):
    """Расчитывает длину одного градуса широты
    на заданной долготе для земли в километрах

    Args:
        longitude (float): Значение долготы с десятичной частью

    Returns:
        float: Возвращает длину одного градуса широты в км
    """
    return (
        111132.954
        - 559.822 * np.cos(2 * np.radians(longitude))
        + +1.175 * np.cos(4 * np.radians(longitude))
    ) / 1000


def coords_matrix(latitude, longitude, shape):
    """Составляет матрицу по заданным координатам и размерам

    Args:
        latitude (float): долгота цетральной точки (расположения радара)
        longitude (float): широта цетральной точки (расположения радара)
        shape (int): размер матрицы радара

    Returns:
        tuple of np.arrays: Возвращает две матрицы (np.array) со значениями координат в узлах сетки
    """
    n_border = latitude + shape / length_of_a_degree_of_latitude(longitude)
    s_border = latitude - shape / length_of_a_degree_of_latitude(longitude)
    y_coor = np.atleast_2d(np.linspace(n_border, s_border, shape + 1)).T
    longitude_matrix = (
        (
            np.linspace(
                longitude - shape / length_of_a_degree_of_longitude(y_coor),
                longitude + shape / length_of_a_degree_of_longitude(y_coor),
                shape + 1,
            )
        ).reshape(shape + 1, -1)
    ).T

    latitude_matrix = np.tile(
        np.linspace(n_border, s_border, shape + 1).reshape((shape + 1, 1)),
        (1, shape + 1),
    )
    return latitude_matrix, longitude_matrix


def convert_json_to_df_blitz(path, lat_max, lat_min, lon_max, lon_min, station_number):
    """Переводит данные из blitzortung json файлов в pd.DataFrame

    Args:
        path (str): Путь к данным
        lat_max (float): Максимум широты
        lat_min (float): Минимум широты
        lon_max (float): Максимум долготы
        lon_min (float): Минимум долготы
        station_number (int): Номер радара

    Returns:
        pd.DataFrame: таблица с временем и координатами разрядов, также сохраняется в "data/interim/df_blitz_{station_number}.parquet"
    """
    if os.path.exists(Paths.interim + f"df_blitz_{station_number}.parquet"):
        print("Blitzortung data is already converted")
        return pd.read_parquet(Paths.interim + f"df_blitz_{station_number}.parquet")
    else:
        blitz_files = onlyfiles(path)

        whatever = []

        for jsonfilename in tqdm(blitz_files):
            with gzip.open(jsonfilename, "r") as fin:
                json_bytes = fin.read()
            splitted = json_bytes.split(b"\n")

            for i in range(len(splitted)):
                try:
                    pime = datetime.fromtimestamp(
                        mktime(time.gmtime((float(splitted[i][8:27]) * 10**-9)))
                    )
                    pat = float(splitted[i][34:42])
                    pon = float(splitted[i][50:58])
                    whatever.append((pime, pat, pon))

                except ValueError as e:
                    if str(e) == "could not convert string to float: ''":
                        pass

        print(f"Number of blitzortung cases is: {len(whatever)}")
        df = pd.DataFrame(np.array(whatever), columns=["date", "lat", "lon"])
        df["lat"] = pd.to_numeric(df["lat"])
        df["lon"] = pd.to_numeric(df["lon"])
        df["date"] = df["date"].dt.ceil(freq="10min")
        df["Lightning"] = 1
        df_drop = df[
            (df.lat < lat_max)
            & (df.lat > lat_min)
            & (df.lon < lon_max)
            & (df.lon > lon_min)
        ]
        df_final = df_drop.sort_values("date").reset_index(drop=True)
        df_final.to_parquet(Paths.interim + f"df_blitz_{station_number}.parquet")
        return df_final


def centers_of_cells_lat(matrix):
    """Расчёт центров ячеек для долготной матрицы

    Args:
        matrix (np.array): матрица узлов долгот

    Returns:
        np.array: Матрица центров ячеек широтной сетки
    """
    matrix = matrix[:, 0]
    matrix_mean = []
    for i in range(matrix.shape[0] - 1):
        matrix_mean.append(np.mean((matrix[i], matrix[i + 1])))
    return np.array(matrix_mean)


def centers_of_cells_lon(matrix):
    """Расчёт центров ячеек для широтной матрицы

    Args:
        matrix (np.array): матрица узлов широт

    Returns:
        np.array: Матрица центров ячеек широтной сетки
    """
    matrix_mean = []
    for j in range(matrix.shape[1] - 1):
        for i in range(matrix.shape[0] - 1):
            matrix_mean.append(np.mean((matrix[j][i], matrix[j][i + 1])))
    return np.array(matrix_mean).reshape((200, 200))


def find_nearest(data, lat_matrix, lon_matrix):
    """Нахождение ближайших ячеек к молниевым разрядам

    Args:
        data (pd.DataFrame): таблица data/interim/df_blitz_{station_number}.parquet
        lat_matrix (np.array): Матрица центров ячеек широтной сетки
        lon_matrix (np.array): Матрица центров ячеек долготной сетки

    Returns:
        np.array: матрицу, формата таблицы, с координатами ячеек на радарной сетке (0,200)
    """
    lat_values = data.lat
    lon_values = data.lon
    idx_lat = []
    idx_lon = []
    for i in range(data.shape[0]):
        lat_idx = (np.abs(lat_matrix - lat_values[i])).argmin()
        idx_lat.append(lat_idx)
        idx_lon.append((np.abs(lon_matrix[lat_idx] - lon_values[i])).argmin())
    position = list(zip(idx_lat, idx_lon))
    return np.array(position)


def bo_matrix(data_bo, shape):
    """Составление матриц с отметками грозопленгаторов на радарной сетке

    Args:
        data_bo (list): Список совпавших дат с радарными данными
        shape (int): размер радарной матрицы

    Returns:
        np.array: Матрица формата (DxWxH), где D - сроки, W и H - размеры матрицы по ширине и высоте
    """
    matrix_bo = np.zeros((data_bo.date.nunique(), shape, shape))
    dfs = [y for x, y in data_bo.groupby("date", as_index=False)]
    for counter, value in enumerate(dfs):
        for index, row in value.iterrows():
            matrix_bo[counter, row["pos_lat"], row["pos_lon"]] += 1
    return matrix_bo


def date_intersection(path_to_binary, station_number, year_regex):
    """Вычисляет пересечение радарных данных и данных грозопеленгаторов
    Обрабатывает допплеровские характеристики

    Args:
        path_to_binary (str): Путь к бинарным файлам (Paths.binary)
        station_number (int): Номер радара/станции
        year_regex (str): год данных в формате регулярных выражений

    Returns:
        list: список совпадающих сроков в формате файлов по отражаемости
    """

    # Найти пересечение радарных файлов
    Z = onlyfiles(path_to_binary + f"S{station_number}")
    W = [
        elem.replace("W", "S", 1).replace("W", "A", 1)
        for elem in onlyfiles(path_to_binary + f"W{station_number}")
    ]
    R = [
        elem.replace("R", "S", 1).replace("R", "A", 1)
        for elem in onlyfiles(path_to_binary + f"R{station_number}")
    ]
    D = [
        elem.replace("D", "S", 1)[:-1] + "A"
        for elem in onlyfiles(path_to_binary + f"D{station_number}")
    ]
    F = [
        elem.replace("F", "S", 1).replace("F", "A", 1)
        for elem in onlyfiles(path_to_binary + f"F{station_number}")
    ]
    radar_matched = list(set(Z) & set(W) & set(R) & set(D) & set(F))

    # get blitz dates

    blitz_data = pd.read_parquet(Paths.interim + f"df_blitz_{station_number}.parquet")
    blitz_test = blitz_data[
        blitz_data["date"].astype(str).str.contains(year_regex, regex=True)
    ]
    blitz_dates = np.unique(blitz_test.date)

    matched_list = get_same_dates(radar_matched, blitz_dates)
    match_str = [str(elem) for elem in matched_list]

    binary_list = get_binary_list_from_matched_list(match_str, station_number, "S", "A")
    return binary_list


def get_same_dates(radar_list, blitz_dates):
    """Вычисляет пересечение радарных данных и данных грозопеленгаторов

    Args:
        radar_list (list): список радарных файлов
        blitz_dates (_type_): список уникальных дат грозопеленгаторов

    Returns:
        list: список совпадающих дат
    """
    radar_date = []
    for i in range(len(radar_list)):
        fileName = radar_list[i]
        date = (
            f"{re.search('S.....', fileName).group(0)[1:]}_"
            f"{re.search('G....', fileName).group(0)[1:]}_{re.search('M..', fileName).group(0)[1:]}_"
            f"{re.search('D..', fileName).group(0)[1:]}__{re.findall('/....', fileName)[-1][1:3]}_"
            f"{re.findall('/....', fileName)[-1][3:5]}"
        )
        radar_date.append(date)

    radar_dt = pd.to_datetime(
        [elem.replace("_", "-")[6:] for elem in radar_date], format="%Y-%m-%d--%H-%M"
    )
    inter = list(set(pd.to_datetime(blitz_dates)) & set(radar_dt))
    inter.sort()
    return inter


def get_npy_file(matched_list):
    """Создаёт единый np.array со всеми данными из указанных файлов

    Args:
        matched_list (list): список с файлами для преобразования

    Returns:
        np.array: Матрица со всеми файлами
    """
    train_images = []
    for i in tqdm(matched_list):
        data = np.load(i, allow_pickle=True)
        train_images.append(data)
    return train_images


def window_slider(data_radar, window_shape_radar=5):
    """Проводит операцию скользящего окна по имеющимся данным

    Args:
        data_radar (np.array): Матрицы со всем данными (stack.npy)
        window_shape_radar (int): размер окна

    Returns:
        da.array: dask матрица ('Ленивая' обработка)
    """

    radar_da = da.from_array(data_radar)
    window_radar = da.lib.stride_tricks.sliding_window_view(
        radar_da, window_shape=window_shape_radar
    )
    reshape_param_radar = (
        window_radar.shape[0]
        * window_radar.shape[1]
        * window_radar.shape[2]
        * window_radar.shape[3]
    )

    reshaped_window_radar = window_radar.reshape(
        reshape_param_radar,
        window_shape_radar[1],
        window_shape_radar[2],
        window_shape_radar[3],
    )
    return reshaped_window_radar


def scorer(y_true, y_pred, verbose=1):
    """Вычисляет следующие метрики:
    HSS, ETS, CSI, Acc, F1, precision, recall, roc_auc, conf_matrix

    Args:
        y_true (np.array): вектор фактических значений для теста
        y_pred (np.array): вектор предсказаний
        verbose (int, optional): Выводит предсказания. Defaults to 1.

    Returns:
        pd.Series: таблица с результатами
        sns.heatmap: plot of confusion matrix
    """
    correctnegatives, falsealarms, misses, hits = confusion_matrix(
        y_true, y_pred
    ).ravel()

    hss_num = 2 * (hits * correctnegatives - misses * falsealarms)
    hss_den = (
        misses**2
        + falsealarms**2
        + 2 * hits * correctnegatives
        + (misses + falsealarms) * (hits + correctnegatives)
    )

    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    dr = num / den
    ets = (hits - dr) / (hits + misses + falsealarms - dr)
    pre = hits / (hits + misses)
    rec = hits / (hits + falsealarms)

    h = hits / (hits + falsealarms)
    f = misses / (misses + correctnegatives)

    sedi = (np.log10(f) - np.log10(h) - np.log10(1 - f) + np.log10(1 - h)) / (
        np.log10(f) + np.log10(h) + np.log10(1 - f) + np.log10(1 - h)
    )

    hss = hss_num / hss_den
    csi = hits / (hits + misses + falsealarms)
    acc = (hits + correctnegatives) / (correctnegatives + falsealarms + misses + hits)
    f1 = 2 * ((pre * rec) / (pre + rec))
    roc_auc = roc_auc_score(y_true, y_pred)

    if verbose == 1:
        print(f"Accuracy is {np.round(acc,4)}")
        print(f"f1_score is {np.round(f1,4)}")
        print(f"precision_score is {np.round(pre,4)}")
        print(f"recall_score is {np.round(rec,4)}")
        print(f"roc_auc_score is {np.round(roc_auc,4)}")
        print(f"HSS is {np.round(hss,4)}")
        print(f"ETS is {np.round(ets,4)}")
        print(f"CSI is {np.round(csi,4)}")
        print(f"sedi is {np.round(sedi,4)}")
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="flag")

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(["True", "False"])
        ax.yaxis.set_ticklabels(["True", "False"])

    return pd.Series((acc, f1, pre, rec, roc_auc, hss, ets, csi, sedi))


def convert_binary_to_array_phen(list_of_files, station_number, path_to_save) -> None:
    """Преобразует бинарные файлы отражаемости в матрицы явлений

    Args:
        list_of_files (list): Список радарных файлов по отражаемости
        station_number (int): Номер радара/станции
        path_to_save (str): Путь для сохранения данных
    """
    if os.path.isdir(path_to_save + f"phen_{station_number}"):
        print("phen data is already converted")
        return
    else:
        print("Converting phen")
        Path(path_to_save + f"phen_{station_number}").mkdir(parents=True, exist_ok=True)
        for fileName in tqdm(list_of_files):
            array_one_time = np.fromfile(fileName, dtype="byte")
            data = np.fromfile(fileName, dtype="ubyte")
            shaped_array = np.zeros((200, 200))

            for i in range(514, array_one_time.shape[0] - 20, 16):
                if data[i] == 255:
                    y = data[i + 1]
                    x = data[i + 2]
                    shaped_array[y][x] = array_one_time[i + 4]

            shaped_array = shaped_array.astype("int32")
            np.save(
                path_to_save
                + f"phen_{station_number}/{re.search('S.....', fileName).group(0)[1:]}_"
                f"{re.search('G....', fileName).group(0)[1:]}_{re.search('M..', fileName).group(0)[1:]}_"
                f"{re.search('D..', fileName).group(0)[1:]}__{re.findall('/....', fileName)[-1][1:3]}_"
                f"{re.findall('/....', fileName)[-1][3:5]}.npy",
                shaped_array,
            )


def convert_binary_to_array_echo_top(
    list_of_files, station_number, path_to_save
) -> None:
    """Преобразует бинарные файлы отражаемости в матрицы, содержащие информацию о верхней высоте облачности

    Args:
        list_of_files (list): Список радарных файлов по отражаемости
        station_number (int): Номер радара/станции
        path_to_save (str): Путь для сохранения данных
    """
    if os.path.isdir(path_to_save + f"echo_top_{station_number}"):
        print("echo_top data is already converted")
        return
    else:
        print("Converting echo_top")
        Path(path_to_save + f"echo_top_{station_number}").mkdir(
            parents=True, exist_ok=True
        )
        for fileName in tqdm(list_of_files):
            array_one_time = np.fromfile(fileName, dtype="byte")
            data = np.fromfile(fileName, dtype="ubyte")
            shaped_array = np.zeros((200, 200))

            for i in range(514, array_one_time.shape[0] - 20, 16):
                if data[i] == 255:
                    y = data[i + 1]
                    x = data[i + 2]
                    shaped_array[y][x] = array_one_time[i + 3]

            shaped_array = shaped_array.astype("int32")
            np.save(
                path_to_save
                + f"echo_top_{station_number}/{re.search('S.....', fileName).group(0)[1:]}_"
                f"{re.search('G....', fileName).group(0)[1:]}_{re.search('M..', fileName).group(0)[1:]}_"
                f"{re.search('D..', fileName).group(0)[1:]}__{re.findall('/....', fileName)[-1][1:3]}_"
                f"{re.findall('/....', fileName)[-1][3:5]}.npy",
                shaped_array,
            )


def convert_binary_to_array_properties(
    list_of_files, station_number, path_to_save
) -> None:
    """Преобразует бинарные файлы отражаемости в матрицы, содержащие информацию:
          - высоте изотермы -22
          - высоте изотермы 0
          - высоте тропопаузы
          - дате
          - x координате
          - y координате

    Args:
        list_of_files (list): Список радарных файлов по отражаемости
        station_number (int): Номер радара/станции
        path_to_save (str): Путь для сохранения данных
    """
    if os.path.isdir(path_to_save + f"prop_{station_number}"):
        print("prop data is already converted")
        return
    else:
        print("Converting properties")
        Path(path_to_save + f"prop_{station_number}").mkdir(parents=True, exist_ok=True)

        for fileName in tqdm(list_of_files):
            data_us = np.fromfile(fileName, dtype="ushort")
            iso_22 = np.full((200, 200), data_us[14])
            iso_0 = np.full((200, 200), data_us[15])
            tropopause = np.full((200, 200), data_us[92])

            date_a = (
                f"{re.search('G....', fileName).group(0)[1:]}-{re.search('M..', fileName).group(0)[1:]}-"
                f"{re.search('D..', fileName).group(0)[1:]}T{re.findall('/....', fileName)[-1][1:3]}:"
                f"{re.findall('/....', fileName)[-1][3:5]}"
            )

            date = np.full((200, 200), np.datetime64(date_a).astype("int32"))

            x_coordinate = np.array([np.arange(0, 200) for _ in range(200)])
            y_coordinate = np.stack(x_coordinate, axis=1)

            stack = np.stack(
                (iso_22, iso_0, date, x_coordinate, y_coordinate, tropopause), axis=-1
            )
            stack = stack.astype("int32")

            np.save(
                path_to_save
                + f"prop_{station_number}/{re.search('S.....', fileName).group(0)[1:]}_"
                f"{re.search('G....', fileName).group(0)[1:]}_{re.search('M..', fileName).group(0)[1:]}_"
                f"{re.search('D..', fileName).group(0)[1:]}__{re.findall('/....', fileName)[-1][1:3]}_"
                f"{re.findall('/....', fileName)[-1][3:5]}.npy",
                stack,
            )


def convert_binary_to_array_reflectivity(
    list_of_files, station_number, path_to_save
) -> None:
    """Преобразует бинарные файлы отражаемости в матрицы, содержащие информацию об отражаемости

    Args:
        list_of_files (list): Список радарных файлов по отражаемости
        station_number (int): Номер радара/станции
        path_to_save (str): Путь для сохранения данных
    """
    if os.path.isdir(path_to_save + f"reflectivity_{station_number}"):
        print("reflectivity data is already converted")
        return

    else:
        print("Converting reflectivity")
        Path(path_to_save + f"reflectivity_{station_number}").mkdir(
            parents=True, exist_ok=True
        )
        for fileName in tqdm(list_of_files):
            array_one_time = np.fromfile(fileName, dtype="byte")
            data = np.fromfile(fileName, dtype="ubyte")
            shaped_array = np.full((200, 200, 11), -100)

            for i in range(514, array_one_time.shape[0] - 20, 16):
                if data[i] == 255:
                    y = data[i + 1]
                    x = data[i + 2]
                    shaped_array[y][x][0] = array_one_time[i + 5]
                    shaped_array[y][x][1] = array_one_time[i + 6]
                    shaped_array[y][x][2] = array_one_time[i + 7]
                    shaped_array[y][x][3] = array_one_time[i + 8]
                    shaped_array[y][x][4] = array_one_time[i + 9]
                    shaped_array[y][x][5] = array_one_time[i + 10]
                    shaped_array[y][x][6] = array_one_time[i + 11]
                    shaped_array[y][x][7] = array_one_time[i + 12]
                    shaped_array[y][x][8] = array_one_time[i + 13]
                    shaped_array[y][x][9] = array_one_time[i + 14]
                    shaped_array[y][x][10] = array_one_time[i + 15]

            shaped_array = shaped_array.astype("int32")
            np.save(
                path_to_save
                + f"reflectivity_{station_number}/{re.search('S.....', fileName).group(0)[1:]}_"
                f"{re.search('G....', fileName).group(0)[1:]}_{re.search('M..', fileName).group(0)[1:]}_"
                f"{re.search('D..', fileName).group(0)[1:]}__{re.findall('/....', fileName)[-1][1:3]}_"
                f"{re.findall('/....', fileName)[-1][3:5]}.npy",
                shaped_array,
            )


def convert_binary_to_array_doppler(
    list_of_files, station_number, path_to_save, doppler_param
) -> None:
    """Преобразует бинарные файлы отражаемости в матрицы,
    содержащие информацию о допплеровских характеристиках:
          - Ширина спектра
          - Радиальная скорость
          - Дифференциальная отражаемость
          - Дифференциальаня фаза

    Args:
        list_of_files (list): Список радарных файлов по отражаемости
        station_number (int): Номер радара/станции
        path_to_save (str): Путь для сохранения данных
    """

    shift_for_day = 0

    if doppler_param == "W":
        param = "Spectrum_Width"

    if doppler_param == "R":
        param = "Radial_Velocity"

    if doppler_param == "D":
        param = "Differential_Reflectivity"
        shift_for_day = 1

    if doppler_param == "F":
        param = "Differential_Phase_Shift"

    param_files = [re.sub("S|A", doppler_param, elem) for elem in list_of_files]

    if os.path.isdir(path_to_save + f"{param}_{station_number}"):
        print(f"{param} data is already converted")
        return
    else:
        print(f"Converting {param}")
        Path(path_to_save + f"{param}_{station_number}").mkdir(
            parents=True, exist_ok=True
        )

        for fileName in tqdm(param_files):
            array_one_time = np.fromfile(fileName, dtype="byte")
            data = np.fromfile(fileName, dtype="ubyte")
            shaped_array = np.full((200, 200, 11), -100)
            iterator = np.array(())
            for i in range(1, len(data) // 514):
                iterator = np.append(
                    iterator,
                    np.arange(514 * i - 2 * (i - 1), 514 * i + 504 - 2 * (i - 1), 14),
                ).astype(int)
            for i in iterator:
                if data[i] == 255:
                    y = data[i + 1]
                    x = data[i + 2]
                    if x > 199 or y > 199:
                        continue
                    shaped_array[y][x][0] = array_one_time[i + 3]
                    shaped_array[y][x][1] = array_one_time[i + 4]
                    shaped_array[y][x][2] = array_one_time[i + 5]
                    shaped_array[y][x][3] = array_one_time[i + 6]
                    shaped_array[y][x][4] = array_one_time[i + 7]
                    shaped_array[y][x][5] = array_one_time[i + 8]
                    shaped_array[y][x][6] = array_one_time[i + 9]
                    shaped_array[y][x][7] = array_one_time[i + 10]
                    shaped_array[y][x][8] = array_one_time[i + 11]
                    shaped_array[y][x][9] = array_one_time[i + 12]
                    shaped_array[y][x][10] = array_one_time[i + 13]

            shaped_array = shaped_array.astype("int32")
            np.save(
                path_to_save
                + f"{param}_{station_number}/{re.search(f'{doppler_param}.....', fileName).group(0)[1:]}_"
                f"{re.search('G....', fileName).group(0)[1:]}_{re.search('M..', fileName).group(0)[1:]}_"
                f"{re.findall('D..', fileName)[shift_for_day][1:]}__{re.findall('/....', fileName)[-1][1:3]}_"
                f"{re.findall('/....', fileName)[-1][3:5]}.npy",
                shaped_array,
            )


def get_stacked_array(matched_list, doppler=False):
    """Собирает все радарные данные в np.array

    Args:
        matched_list (list): список совпавших данных в формате данных отражаемости
        doppler (bool, optional): Обработка допплеровской информации. Defaults to False.

    Returns:
        np.array: обработанные радарные данные
    """

    matched_list_phen = [elem.replace("reflectivity", "phen") for elem in matched_list]
    matched_list_echotop = [
        elem.replace("reflectivity", "echo_top") for elem in matched_list
    ]
    matched_list_prop = [elem.replace("reflectivity", "prop") for elem in matched_list]

    phen = get_npy_file(matched_list_phen)
    echotop = get_npy_file(matched_list_echotop)
    reflectivity = get_npy_file(matched_list)
    prop_array = get_npy_file(matched_list_prop)

    if doppler:
        matched_list_R = [
            re.sub("reflectivity", "Radial_Velocity", elem) for elem in matched_list
        ]
        matched_list_W = [
            re.sub("reflectivity", "Spectrum_Width", elem) for elem in matched_list
        ]
        matched_list_D = [
            re.sub("reflectivity", "Differential_Reflectivity", elem)
            for elem in matched_list
        ]
        matched_list_F = [
            re.sub("reflectivity", "Differential_Phase_Shift", elem)
            for elem in matched_list
        ]

        R = get_npy_file(matched_list_R)
        W = get_npy_file(matched_list_W)
        D = get_npy_file(matched_list_D)
        F = get_npy_file(matched_list_F)

        final_radar = np.concatenate(
            (reflectivity, np.stack((echotop, phen), axis=3), prop_array, R, W, D, F),
            axis=3,
        )

    else:
        quater_array = np.stack((echotop, phen), axis=3)
        semi_array = np.concatenate((quater_array, prop_array), axis=3)
        final_radar = np.concatenate((reflectivity, semi_array), axis=3)

    return final_radar


def get_binary_list_from_matched_list(matched_list, station_number, param, file_param):
    """Возвращает лист с бинарными файлами для выбранного парамтера из совпавших дат

    Args:
        matched_list (list): Лист совпавших дат
        station_number (int): Номер станции/радара
        param (str): Код параметра (папки)
        file_param (str): Код параметра (файл)
                к примеру: /S26850/G2021/M06/D06/005006.06A, где S - param, A - file_param.
    Returns:
        list: список бинарных файлов
    """
    matched_list_new = []
    for elem in matched_list:
        elem_new = (
            Paths.binary + f"{param}{station_number}/G{elem[:4]}/M{elem[5:7]}/"
            f"D{elem[8:10]}/{elem[11:13]}{elem[14:16]}{elem[8:10]}.{elem[5:7]}{file_param}"
        )
        matched_list_new.append(elem_new)
    return matched_list_new


def decision_func(prob_array):
    """Сравнивает вероятность из prob_array с рандомно сгенерированным значением

    Args:
        prob_array (np.array): Вектор вероятностей

    Returns:
        np.array: Вектор булевых занчений, true - значение prob_array >= random_array
    """
    random_array = np.random.random(size=prob_array.shape[0])
    return np.less(random_array, prob_array)


def define_hmax_greater_h22_false(data):
    """Расчёт разницы (дельты) между верхней высотой облачности и высоты изотермы -22

    Args:
        data (dask.array): Данные после скользящего окна для фильтрации

    Returns:
        np.array: Вектор дельт
    """
    results = np.array(())
    for inds in product(*map(range, data.blocks.shape)):
        chunk = data.blocks[inds]

        chunk_hmax = chunk[:, :, :, 11]
        chunk_h22 = chunk[:, :, :, 13]

        computed_hmax = chunk_hmax.compute()
        computed_h22 = chunk_h22.compute()

        echotop_max = np.max(computed_hmax, axis=(1, 2))
        h22_max = np.max(computed_h22, axis=(1, 2))

        delta = echotop_max * 250 - h22_max

        results = np.concatenate((results, delta))

    return results


def drop_zeros_matrix(data, zero_value):
    """Фильтрует блоки без эха при заданном значении

    Args:
        data (dask.array): Данные после скользящего окна для фильтрации
        zero_value (int): Значение, которым помечается данные без эха

    Returns:
        np.array: Индексы блоков, имеющих эхо.
    """
    final_indices = np.array(())
    for inds in product(*map(range, data.blocks.shape)):
        chunk = data.blocks[inds]
        computed_batch = chunk[:, :, :, :11].compute()
        final_indices = np.concatenate(
            (
                final_indices,
                np.not_equal(np.max(computed_batch, axis=(1, 2, 3)), zero_value),
            )
        )

    return final_indices


def exponential_filter(delta_data, station_number):
    """Применение фильтра равномерного распределения.
    (от экспоненциального отказались, но название осталось)
    Используется для уменьшения количества данных (downscaling)

    Args:
        delta_data (np.array): Данные о дельте верхней границы облачности и изотермы -22
        station_number (int): Номер станции/радара

    Returns:
        np.array: вектор значений 1 и 0.
    """
    decision_table = pd.read_parquet(
        Paths.interim + f"decision_table_{station_number}.parquet"
    )

    delta_array = np.array(decision_table["delta"])
    expon_array = np.array(decision_table["expon"])
    probas = np.searchsorted(delta_array, delta_data)
    probabilities = expon_array[probas]
    filter_bool = decision_func(probabilities)
    return filter_bool


def factors(n) -> set[int]:
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )


def blitzortung_preprocessing(
    station_number, matrix_shape, year_regex, path_to_binary, doppler=False
):
    """Мета функция. Обрабатывает данные о грозопеленгации.

    Args:
        station_number (int): Номер станции/радара
        matrix_shape (int): Размер матрицы радара
        year_regex (str): Год данных, в формате регулярных выражений
        path_to_binary (str): Путь к бинарным данным
        doppler (bool, optional): Обработка допплеровской информации. Defaults to False.

    Returns:
        np.array: Матрица формата (DxWxH), где D - сроки, W и H - размеры матрицы по ширине и высоте
    """

    Z = onlyfiles(path_to_binary + f"S{station_number}")

    if doppler:
        W = [
            elem.replace("W", "S", 1).replace("W", "A", 1)
            for elem in onlyfiles(path_to_binary + f"W{station_number}")
        ]
        R = [
            elem.replace("R", "S", 1).replace("R", "A", 1)
            for elem in onlyfiles(path_to_binary + f"R{station_number}")
        ]
        D = [
            elem.replace("D", "S", 1)[:-1] + "A"
            for elem in onlyfiles(path_to_binary + f"D{station_number}")
        ]
        F = [
            elem.replace("F", "S", 1).replace("F", "A", 1)
            for elem in onlyfiles(path_to_binary + f"F{station_number}")
        ]
        radar_files = list(set(Z) & set(W) & set(R) & set(D) & set(F))
    else:
        radar_files = Z
    # get central latitude and longitude from radar files
    fileName = radar_files[0]
    data = np.fromfile(fileName, dtype="ubyte")
    latitude = deg2dec(f"""{data[74]}°{data[76]}'{data[78]}"N""")
    longitude = deg2dec(f"""{data[82]}°{data[84]}'{data[86]}"N""")
    print(f"Latitude is {latitude}, Longitude is {longitude}")

    # getting border latitude and longitude of matricies
    #  from given lat and long of radar location
    latitude_matrix = coords_matrix(latitude, longitude, matrix_shape)[0]
    longitude_matrix = coords_matrix(latitude, longitude, matrix_shape)[1]

    # define max and mins of matricies
    lat_max = latitude_matrix.max()
    lat_min = latitude_matrix.min()
    lon_max = longitude_matrix.max()
    lon_min = longitude_matrix.min()

    if os.path.exists(Paths.interim + f"df_blitz_{station_number}.parquet"):
        print("Blitzortung data is already converted")
        blitz_data = pd.read_parquet(
            Paths.interim + f"df_blitz_{station_number}.parquet"
        )
    else:
        print("Converting data from .json to pd.DataFrame")
        blitz_data = convert_json_to_df_blitz(
            Paths.lightning, lat_max, lat_min, lon_max, lon_min, station_number
        )

    # calculate center of matrices
    lat_centers = centers_of_cells_lat(latitude_matrix)
    lon_centers = centers_of_cells_lon(longitude_matrix)

    # find nearest strikes to centers of cells, add get cell coordiantes
    position = find_nearest(blitz_data, lat_centers, lon_centers)

    # add to dataframe cell coordinates for each strike
    blitz_data["pos_lat"], blitz_data["pos_lon"] = position[:, 0], position[:, 1]

    # getting matched dates in year_regex
    blitz_test = blitz_data[
        blitz_data["date"].astype(str).str.contains(year_regex, regex=True)
    ]
    blitz_dates = np.unique(blitz_test.date)

    matched_list = get_same_dates(radar_files, blitz_dates)
    matched_date = blitz_data.loc[
        blitz_data["date"].isin(np.array(matched_list).astype("datetime64[ns]"))
    ]
    # convert blitzortung data to arrays
    final_matrix = bo_matrix(matched_date, matrix_shape)
    print(f"Array shape is: {final_matrix.shape}")
    return final_matrix


def radar_preprocessing(
    station_number,
    year_regex,
    path_to_binary,
    path_to_save,
    n_extra_files,
    doppler=False,
):
    """Мета функция. Обрабатывает радарные данные.

    Args:
        station_number (int): Номер станции/радара
        year_regex (str): Год данных, в формате регулярных выражений
        path_to_binary (str): Путь к бинарным данным
        path_to_save (str): Путь к директории для сохранения
        n_extra_files (int): Количество файлов, наибольших по объёму,
                             которые будут дополнительно добавлены к срокам с молниевыми разрядами.
                             Сроки по объёму и разрядам могут пересекаться.
        doppler (bool, optional): Обработка допплеровских характеристик. Defaults to False.

    Returns:
        np.array: Обработанный массив с радарными данными.
    """

    binary_matched = date_intersection(path_to_binary, station_number, year_regex)

    if doppler:
        Z = binary_matched
        W = [
            elem.replace("W", "S", 1).replace("W", "A", 1)
            for elem in onlyfiles(Paths.binary + f"W{station_number}")
        ]
        R = [
            elem.replace("R", "S", 1).replace("R", "A", 1)
            for elem in onlyfiles(Paths.binary + f"R{station_number}")
        ]
        D = [
            elem.replace("D", "S", 1)[:-1] + "A"
            for elem in onlyfiles(Paths.binary + f"D{station_number}")
        ]
        F = [
            elem.replace("F", "S", 1).replace("F", "A", 1)
            for elem in onlyfiles(Paths.binary + f"F{station_number}")
        ]

        binary_matched = list(set(Z) & set(W) & set(R) & set(D) & set(F))

        if n_extra_files > 0:
            path_to_binary = Paths.binary
            Z = onlyfiles(path_to_binary + f"S{station_number}")
            W = [
                elem.replace("W", "S", 1).replace("W", "A", 1)
                for elem in onlyfiles(path_to_binary + f"W{station_number}")
            ]
            R = [
                elem.replace("R", "S", 1).replace("R", "A", 1)
                for elem in onlyfiles(path_to_binary + f"R{station_number}")
            ]
            D = [
                elem.replace("D", "S", 1)[:-1] + "A"
                for elem in onlyfiles(path_to_binary + f"D{station_number}")
            ]
            F = [
                elem.replace("F", "S", 1).replace("F", "A", 1)
                for elem in onlyfiles(path_to_binary + f"F{station_number}")
            ]
            radar_matched = list(set(Z) & set(W) & set(R) & set(D) & set(F))

            size_list = [os.path.getsize(elem) for elem in radar_matched]
            sizes = pd.DataFrame((radar_matched, size_list)).T
            sizes.columns = ["name", "size"]
            extra_df = sizes.sort_values(by="size", ascending=False).iloc[
                :n_extra_files
            ]
            binary_matched = list(set(binary_matched) | set(extra_df["name"]))

        convert_binary_to_array_doppler(
            binary_matched, station_number, path_to_save, "W"
        )
        convert_binary_to_array_doppler(
            binary_matched, station_number, path_to_save, "R"
        )
        convert_binary_to_array_doppler(
            binary_matched, station_number, path_to_save, "D"
        )
        convert_binary_to_array_doppler(
            binary_matched, station_number, path_to_save, "F"
        )

    if n_extra_files > 0 and doppler == False:
        reflectivity_files = onlyfiles(path_to_binary + f"S{station_number}")
        size_list = [os.path.getsize(elem) for elem in reflectivity_files]
        sizes = pd.DataFrame((reflectivity_files, size_list)).T
        sizes.columns = ["name", "size"]
        extra_df = sizes.sort_values(by="size", ascending=False).iloc[:n_extra_files]
        binary_matched = list(set(binary_matched) | set(extra_df["name"]))

    convert_binary_to_array_phen(binary_matched, station_number, path_to_save)
    convert_binary_to_array_echo_top(binary_matched, station_number, path_to_save)
    convert_binary_to_array_reflectivity(binary_matched, station_number, path_to_save)
    convert_binary_to_array_properties(binary_matched, station_number, path_to_save)

    print("Converting to np.array")
    data_to_stack = onlyfiles(path_to_save + f"reflectivity_{station_number}/")
    data_to_stack.sort()
    final_radar = get_stacked_array(data_to_stack, doppler=doppler)
    print(f"Shape of radar array: {final_radar.shape}")
    return final_radar


def concatenate_data(radar_array, blitzortung_array, station_number, year_regex):
    """Мета функция для конкатенации радарных и грозопеленгационных данных.
        Создаёт и сохраняет stack_{station_number}.npy - основной файл для работы

    Args:
        radar_array (np.array): радарные данные
        blitzortung_array (np.array): грозопеленгационные данные
        station_number (int): Номер радара/станции
        year_regex (str): год данных в формате регулярных выражений

    Returns:
        np.array: Основной массив, выровненный в пространстве и времени.
    """
    blitz_data = pd.read_parquet(Paths.interim + f"df_blitz_{station_number}.parquet")
    blitz_test = blitz_data[
        blitz_data["date"].astype(str).str.contains(year_regex, regex=True)
    ]
    blitz_dates = np.unique(blitz_test.date)

    radar_dates = np.array(radar_array[:, 1, 1, 15], dtype="datetime64[m]").astype(
        "datetime64[ns]"
    )

    intersection = list(set(radar_dates) & set(blitz_dates))
    intersection.sort()

    # get indices of matched dates
    index_list = []
    for elem in intersection:
        index_list.append(np.argwhere(radar_dates == elem))

    # create blitzortung array with unmached and matched cases
    zeros_blitz = np.zeros(
        (radar_array.shape[0], radar_array.shape[1], radar_array.shape[2])
    )
    for i in range(len(index_list)):
        idx = index_list[i]
        zeros_blitz[idx] = blitzortung_array[i]
    zeros_blitz = np.expand_dims(zeros_blitz, axis=-1).astype("int32")
    stack = np.concatenate((radar_array, zeros_blitz), axis=3)
    np.save(Paths.interim + f"stack_{station_number}.npy", stack)
    print("All data converted and saved!")
    return stack


def precomputing(
    stack, station_number, window, distribution=np.random.uniform, value=0.05
) -> None:
    """Перед расчётом скользящего окна с фильтрацией пустых значений и уменьшением размерности,
       Необходимо предрасчитать дельту между верхней границей облачности и высотой изотермы -22.
       Также фильтруется большее количество блоков из-за их количества и объёма.

    Args:
        stack (np.array): stack.npy
        station_number (int): Номер станции/радара
        window (int): Размер скользящего окна
        distribution (numpy_distribution, optional): Распределение для фильтрации по дельте. Defaults to np.random.uniform.
        value (float, optional): Доля оставленных блоков. Defaults to 0.05.

    Save:
        decision_table_{station_number}.parquet: Таблица для отброса значений, основанный на дельте.
        target_{station_number}.npy: вектор с данными о метках грозопеленгатора в центре блока
    """
    if os.path.isfile(Paths.interim + f"target_{station_number}.npy"):
        print("target is already computed")
    else:
        print('precomputing "target"')
        window_shape = 1, window, window, stack.shape[-1]
        resh_window_radar = window_slider(stack, window_shape)

        target = resh_window_radar[:, 2, 2, -1].compute()

        np.save(Paths.interim + f"target_{station_number}.npy", target)

    if os.path.isfile(Paths.interim + f"decision_table_{station_number}.parquet"):
        print("delta is already computed")
    else:
        window_shape = 1, window, window, stack.shape[-1]
        resh_window_radar = window_slider(stack, window_shape)
        print('precomputing "delta"')
        delta = define_hmax_greater_h22_false(resh_window_radar)
        unique = np.unique(delta)
        sorted_expon = np.sort(distribution(size=unique.shape[0]))
        scaler = MinMaxScaler().fit(sorted_expon.reshape(-1, 1))
        scaled_probability = scaler.transform(sorted_expon.reshape(-1, 1)).reshape(-1)

        # decreasing prob to get less data from filter. without it 10%, with 3.9
        decision_table = pd.DataFrame((unique, scaled_probability)).T
        decision_table.columns = ["delta", "expon"]
        if distribution == np.random.uniform:
            decision_table["expon"] = value
        decision_table.to_parquet(
            Paths.interim + f"decision_table_{station_number}.parquet"
        )
        print("success")


def window_data_filter(stack, split_value, window, station_number) -> None:
    """Проводит операцию скользящего окна над stack.
        Фильтрует данные по предрасчётанным данным.

    Args:
        stack (np.array): stack.npy
        split_value (int): Размер блоков для расчёта.
                           Рекомендуется значение, нацело делящееся на кол-во сроков около 500 при 16гб RAM, до 1500 при 64гб RAM
        window (int): Размер скользящего окна
        station_number (int): Номер станции/радара

    Save:
        lightning_stack_{station_number}/.../..npy: Сохраняет данные в указанную директорию
    """
    for iteration in range(0, int(stack.shape[0] / split_value)):
        print(f"Iteration: {iteration}")

        # use window slider to get array of given shape
        part_stack = stack[iteration * split_value : (iteration + 1) * split_value]
        window_shape = 1, window, window, part_stack.shape[-1]
        resh_window_radar = window_slider(part_stack, window_shape)
        print(f"Iteration data is \n {resh_window_radar}")

        # load target to define true-false for filter
        target_np = np.load(Paths.interim + f"target_{station_number}.npy")[
            resh_window_radar.shape[0]
            * iteration : resh_window_radar.shape[0]
            * (iteration + 1)
        ]
        (true_index,) = np.where(target_np != 0)
        (false_index,) = np.where(target_np == 0)
        print(f"TF: false-{false_index.shape}, true-{true_index.shape}")

        # False data
        # filtering data based on delta analyzis via expon filter
        delta_data = define_hmax_greater_h22_false(resh_window_radar[false_index])
        delta_drop = exponential_filter(delta_data, station_number)
        print(f"Remaining percent  = {np.round(delta_drop.mean() * 100,2)}")

        false_data = resh_window_radar[false_index]
        false_delta = false_data[delta_drop]
        false_drop_ind = np.where(drop_zeros_matrix(false_delta, -100) == 1)[0]
        false_drop = false_delta[false_drop_ind]

        # True data
        true_data = resh_window_radar[true_index]
        true_drop_ind = np.where(drop_zeros_matrix(true_data, -100) == 1)[0]
        true_drop = true_data[true_drop_ind]

        finale = da.concatenate((false_drop, true_drop))
        print(
            f"TF processed \n : false-{false_drop_ind.shape}, true-{true_drop_ind.shape}"
        )

        print(f"Iteration result is \n {finale}")

        newpath = Paths.data + f"lightning_stack_{station_number}/{iteration}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        for inds in product(*map(range, finale.blocks.shape)):
            chunk = finale.blocks[inds]
            computed = chunk.compute()
            np.save(
                Paths.data
                + f"lightning_stack_{station_number}/{iteration}/{inds[0]}.npy",
                computed,
            )


def from_windows_to_df(path, window_shape, station_number) -> pd.DataFrame:
    """Создаёт pd.DataFrame из lightning_stack.
    Вычисляет и добавляет следующие данные:
    Максимальная отражаемость в блоке
    Максимальная высота облачности в блоке
    Максимальная отражаемость в блоке по вертикальным слоям
    Высоту слоя максимальной отражаемости
    Параметр Y -  Максимальная высота облачности * на Отражаемость на уровне y. Hmax * Z_level_y (level_y = iso_0 + 2km)


    Args:
        path (str): путь к lightning_stack
        window_shape (int): Размер скользящего окна
        station_number (int): Номер станции/радара

    Returns:
        pd.DataFrame: Данные для анализа блоков.
    Save:
        {station_number}_Convective_cloud_windows_data.parquet
    """
    all_data = np.concatenate(get_npy_file(onlyfiles(path)), axis=0)
    # compute reflectivity and echotop
    reflect_18 = all_data[:, :, :, :11] + 18
    reflect_18[reflect_18 == -82] = -100
    echotop = (all_data[:, :, :, 11] * 250).reshape(all_data.shape[0], -1)

    iso_22 = np.max(all_data[:, :, :, 13], axis=(1, 2)).reshape(all_data.shape[0], -1)
    iso_0 = np.max(all_data[:, :, :, 14], axis=(1, 2)).reshape(all_data.shape[0], -1)
    dates = np.max(all_data[:, :, :, 15], axis=(1, 2)).reshape(all_data.shape[0], -1)
    x_coordinate = np.mean(all_data[:, :, :, 16], axis=(1, 2)).reshape(
        all_data.shape[0], -1
    )
    y_coordinate = np.mean(all_data[:, :, :, 17], axis=(1, 2)).reshape(
        all_data.shape[0], -1
    )

    target = all_data[:, all_data.shape[1] // 2, all_data.shape[1] // 2, -1].reshape(
        all_data.shape[0], -1
    )
    finale = np.concatenate(
        (
            reflect_18.reshape(all_data.shape[0], -1),
            echotop,
            iso_22,
            iso_0,
            dates,
            x_coordinate,
            y_coordinate,
            target,
        ),
        axis=1,
    )
    del all_data, echotop, iso_22, iso_0, x_coordinate, y_coordinate, target
    df = pd.DataFrame(finale)
    del finale
    gc.collect()

    # column rename
    col_list = []
    for i in range(window_shape):
        for j in range(window_shape):
            for k in range(11):
                col_list.append(f"R{i, j, k+1}".replace(" ", ""))
    for i in range(window_shape):
        for j in range(window_shape):
            col_list.append(f"H{i, j}".replace(" ", ""))

    col_list.append(f"iso_22".replace(" ", ""))
    col_list.append(f"iso_0".replace(" ", ""))
    col_list.append(f"date".replace(" ", ""))
    col_list.append(f"x_coor".replace(" ", ""))
    col_list.append(f"y_coor".replace(" ", ""))
    col_list.append(f"target".replace(" ", ""))

    # feature engineering
    df.columns = col_list
    df["Hmax"] = df.filter(regex="H").max(axis=1)
    df["Zmax"] = df.filter(regex="R").max(axis=1)
    df["delta_iso22"] = df["Hmax"] - df["iso_22"]

    for i in range(11):
        df[f"Zmax_layer_{i+1}"] = np.max(reflect_18[:, :, :, i], axis=(1, 2))

    # Y param
    level_y = np.round(df["iso_0"] / 1000 + 2).astype("int8")

    log_list = []
    for i in range(df.shape[0]):
        log_list.append(np.log(df[f"Zmax_layer_{level_y[i]}"][i]))

    df["y_param"] = (
        df["Hmax"]
        / 1000
        * np.nan_to_num(np.array(log_list), nan=0.0, posinf=0.0, neginf=0.0)
    )

    df["H_Zmax"] = (
        df.filter(regex="Zmax_layer_.").idxmax(axis=1).str[-2:].str.replace("_", "")
    )
    df["H_Zmax"].where(np.not_equal(df.Zmax, -100), 0, inplace=True)

    df = df.astype("int32")
    df.to_parquet(
        Paths.interim + f"{station_number}_Convective_cloud_windows_data.parquet"
    )
    return df


def true_data_filter(path, station_number):
    """Фильтрует блоки с молниевыми разрядами по:
    - Максимальной отражаемости
    - Максимальной высоте облачности
    - Дельта_изотерма22
    - Удаляет граничные блоки (0-10, 190-200)

    Args:
        path (str): путь к "{station_number}_Convective_cloud_windows_data.parquet"
        station_number (int): Номер станции/радара

    Returns:
        pd.DataFrame: Отфильтрованные данные.
    Save:
        df_filter_{station_number}.parquet
    """
    df = pd.read_parquet(path)

    (true_index,) = np.where(df.target != 0)
    (false_index,) = np.where(df.target == 0)
    print(f"Before: 0 - {false_index.shape}, 1 - {true_index.shape}")
    df_true = df.iloc[true_index]
    df_false = df.iloc[false_index]

    df_filter_rmax = df_true[(df_true["Zmax"] < 3.3)]
    df_filter_hmax = df_true[(df_true["Hmax"] < 4000)]
    df_filter_delta = df_true[(df_true["delta_iso22"] < -2000)]

    union = np.array(
        list(
            set(df_filter_rmax.index)
            | set(df_filter_hmax.index)
            | set(df_filter_delta.index)
        )
    )
    df_filtered_1 = df_true.loc[union]
    df_filter_1 = df_true.loc[~df_true.index.isin(union)]
    df_filtered_1["target"] = df_filtered_1["target"].replace(1, 0)
    df_false.shape, df_filter_1.shape, df_filtered_1.shape

    df = pd.concat((df_false, df_filter_1, df_filtered_1))

    df_drop = df.drop(
        df[
            (df.x_coor > 190) | (df.y_coor > 190) | (df.x_coor < 10) | (df.y_coor < 10)
        ].index
    )

    (true_index,) = np.where(df_drop.target != 0)
    (false_index,) = np.where(df_drop.target == 0)
    print(f"After:  0 - {false_index.shape}, 1 - {true_index.shape}")

    # filter 16 features with regex
    # df_filter = df.filter(regex='max|target|param|delta')

    df.to_parquet(Paths.interim + f"df_filter_{station_number}.parquet")
    return df_drop


def recursive_feature_elimination(path_to_df_filter, station_number):
    """Применяет RFE c кросс-валидацией

    Return  with data about selected features and list of them
    Save resulting data into selected_df_{station_number}.parquet
    Args:
        path_to_df_filter (str): Путь к df_filter_{station_number}.parquet
        station_number (int): Номер станции/радара

    Returns:
        pd.DataFrame: Данные о признаках и их ранке, чем меньше, тем важнее признак.
    Save:
        selected_df_{station_number}.parquet
    """
    # load and trasform data
    df_filter = pd.read_parquet(path_to_df_filter)
    X = df_filter.loc[:, df_filter.columns != "target"]
    y = df_filter.loc[:, "target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=7575
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7575)
    rfecv = RFECV(
        estimator=RandomForestClassifier(n_jobs=-1, random_state=7575),
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    model = RandomForestClassifier(n_jobs=-1, random_state=7575)

    pipeline = Pipeline([("feature selection", rfecv), ("model", model)])

    print("Fitting pipeline")
    pipeline.fit(X_train, y_train)
    print("Optimal number of features : %d" % rfecv.n_features_)

    # make df with ranks and plot
    df_ranks = pd.DataFrame(rfecv.ranking_, index=X.columns, columns=["Rank"])

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test f1-score")
    plt.errorbar(
        range(1, n_scores + 1),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.grid()

    features = df_ranks[df_ranks.Rank == 1].index.tolist()

    selected_df = pd.concat((X[features], y), axis=1)
    selected_df.to_parquet(Paths.interim + f"selected_df_{station_number}.parquet")
    return df_ranks, features


def oversampler_and_scaler(path_to_selected_df, station_number):
    """Осуществляет дублирование блоков с отметками
        грозопеленгатора для сбалансирования данных.
        Осуществляет нормализацию данных.

    Args:
        path_to_selected_df (str): Путь к selected_df_{station_number}.parquet
        station_number (int): Номер станции/радара
    Save:
        Данные для обучения и теста.
    """
    # Load data
    df = pd.read_parquet(path_to_selected_df)
    X = df.loc[:, df.columns != "target"]
    y = df.loc[:, "target"]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # save scaler
    filename = Paths.models + "scaler.joblib"
    joblib.dump(scaler, filename)
    means = scaler.mean_
    vars = scaler.var_
    np.save(Paths.models + "scaler_means.npy", means)
    np.save(Paths.models + "scaler_vars.npy", vars)
    print(f"Scaler params: means {means}, vars: {vars}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    true_index = np.where(y_train == 1)
    false_index = np.where(y_train != 1)
    print("Original dataset shape %s" % Counter(y_train.values.flatten().tolist()))

    # Search for best repeats of true class
    X_train_class1 = np.array(X_train)[true_index[0]]
    y_train_class1 = np.array(y_train)[true_index[0]]
    results = np.array(())
    for i in range(4):
        print(f"iteration: {i}")
        # Oversample the class 1 data points by repeating each element thrice
        X_train_class1_oversampled = X_train_class1.repeat(i, axis=0)
        y_train_class1_oversampled = y_train_class1.repeat(i, axis=0)

        # Concatenate the oversampled data tensors with the original data tensors
        X_train_oversampled = np.concatenate([X_train, X_train_class1_oversampled])
        y_train_oversampled = np.concatenate([y_train, y_train_class1_oversampled])

        # classificator
        sgd_clf = SGDClassifier(n_jobs=-1, early_stopping=True)
        sgd_clf.fit(X_train_oversampled, y_train_oversampled)
        sgd_pred = sgd_clf.predict(X_test)

        # get scores

        sgd_res = scorer(y_test, sgd_pred, verbose=0)
        results = np.hstack((results, sgd_res))
    print(
        f"Best oversampling at {np.argmax(results[1::9])} repeat with f1: {np.round(results[1::9][np.argmax(results[1::9])],2)}"
    )
    repeats = np.argmax(results[1::9])
    # Train model to test it
    true_index = np.where(y_train == 1)
    false_index = np.where(y_train != 1)
    print("Original dataset shape %s" % Counter(y_train.values.flatten().tolist()))

    X_train_class1 = np.array(X_train)[true_index[0]]
    y_train_class1 = np.array(y_train)[true_index[0]]

    # Oversample the class 1 data points by repeating each element thrice
    y_train_class1_oversampled = y_train_class1.repeat(repeats, axis=0)
    X_train_class1_oversampled = X_train_class1.repeat(repeats, axis=0)

    # Concatenate the oversampled data tensors with the original data tensors
    X_train_oversampled = np.concatenate([X_train, X_train_class1_oversampled])
    y_train_oversampled = np.concatenate([y_train, y_train_class1_oversampled])

    print(
        "Resampled dataset shape %s" % Counter(y_train_oversampled.flatten().tolist())
    )

    # classificator
    sgd_clf = LogisticRegression(solver="saga", n_jobs=-1)
    sgd_clf.fit(X_train_oversampled, y_train_oversampled)
    sgd_pred = sgd_clf.predict(X_test)

    # get scores
    sgd_res = scorer(y_test, sgd_pred)

    np.save(Paths.data + f"X_train_{station_number}.npy", X_train_oversampled)
    np.save(Paths.data + f"y_train_{station_number}.npy", y_train_oversampled)
    np.save(Paths.data + f"X_test_{station_number}.npy", X_test)
    np.save(Paths.data + f"y_test_{station_number}.npy", y_test)


def basic_dataset(station_number) -> pd.DataFrame:
    """Создание из stack.npy, таблицу для базовой классификации молний.
       Включает в себя фильтрацию ячеек и генерацию признаков.

    Args:
        station_number (int): Номер станции

    Returns:
        pd.DataFrame: готовый pd.DataFrame для анализа и классификации
    """

    if os.path.exists(Paths.interim + f"basic_classif_{station_number}.parquet"):
        print("Data is already converted!")
        return pd.read_parquet(
            Paths.interim + f"basic_classif_{station_number}.parquet"
        )

    else:
        stack = np.load(Paths.interim + f"stack_{station_number}.npy").reshape(-1, 64)
        print(f"Размерность данных для базовой классификации: {stack.shape}")

        # Создание pd.DataFrame и обозначение колонок
        df = pd.DataFrame(
            stack,
            columns=[
                "Z1",
                "Z2",
                "Z3",
                "Z4",
                "Z5",
                "Z6",
                "Z7",
                "Z8",
                "Z9",
                "Z10",
                "Z11",
                "Hmax",
                "phen",
                "iso_22",
                "iso_0",
                "date",
                "x",
                "y",
                "tropopause",
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "R9",
                "R10",
                "R11",
                "W1",
                "W2",
                "W3",
                "W4",
                "W5",
                "W6",
                "W7",
                "W8",
                "W9",
                "W10",
                "W11",
                "D1",
                "D2",
                "D3",
                "D4",
                "D5",
                "D6",
                "D7",
                "D8",
                "D9",
                "D10",
                "D11",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "F9",
                "F10",
                "F11",
                "target",
            ],
        )

        # Изменить типы данных для экономии памяти
        df = df.astype(
            {
                "Hmax": "int16",
                "iso_22": "int16",
                "iso_0": "int16",
                "date": "int32",
                "x": "int16",
                "y": "int16",
                "tropopause": "int16",
                "Z1": "int8",
                "Z2": "int8",
                "Z3": "int8",
                "Z4": "int8",
                "Z5": "int8",
                "Z6": "int8",
                "Z7": "int8",
                "Z8": "int8",
                "Z9": "int8",
                "Z10": "int8",
                "Z11": "int8",
                "R1": "int8",
                "R2": "int8",
                "R3": "int8",
                "R4": "int8",
                "R5": "int8",
                "R6": "int8",
                "R7": "int8",
                "R8": "int8",
                "R9": "int8",
                "R10": "int8",
                "R11": "int8",
                "W1": "int8",
                "W2": "int8",
                "W3": "int8",
                "W4": "int8",
                "W5": "int8",
                "W6": "int8",
                "W7": "int8",
                "W8": "int8",
                "W9": "int8",
                "W10": "int8",
                "W11": "int8",
                "D1": "int8",
                "D2": "int8",
                "D3": "int8",
                "D4": "int8",
                "D5": "int8",
                "D6": "int8",
                "D7": "int8",
                "D8": "int8",
                "D9": "int8",
                "D10": "int8",
                "D11": "int8",
                "F1": "int8",
                "F2": "int8",
                "F3": "int8",
                "F4": "int8",
                "F5": "int8",
                "F6": "int8",
                "F7": "int8",
                "F8": "int8",
                "F9": "int8",
                "F10": "int8",
                "F11": "int8",
                "phen": "int8",
                "target": "int16",
            }
        )

        df.reset_index(drop=True, inplace=True)
        # К отражаемости необходимо добавить +18, значение без эха = -100
        df.loc[:, df.filter(regex="Z").columns] = (
            df.loc[:, df.filter(regex="Z").columns] + 18
        )
        df = df.replace(-82, -100)
        # Привести Hmax в метры из стробов, +1 - исправление ошибки на радаре 26061
        if df["Hmax"].max() == 63:
            df["Hmax"] = (df["Hmax"] + 1) * 250
        else:
            df["Hmax"] = df["Hmax"] * 250

        # Максимальная отражаемость в ячейке
        df["Zmax"] = df.filter(regex="Z").max(axis=1)
        df["Rmax"] = df.filter(regex="R").max(axis=1)
        df["Wmax"] = df.filter(regex="W").max(axis=1)
        df["Dmax"] = df.filter(regex="D").max(axis=1)
        df["Fmax"] = df.filter(regex="F").max(axis=1)

        # Удалить данные без эха в ячейке и граничные ячейки
        df = df.drop(
            df[
                (df.Zmax == -100)
                | (df.x > 190)
                | (df.y > 190)
                | (df.x < 10)
                | (df.y < 10)
                | (df.phen == -110)
            ].index
        ).reset_index(drop=True)
        print(
            f"Размерность данных после фильтрации нулевых значений и граничных ячеек: {df.shape}"
        )

        # Разность максимальной высоты радиоэха и высоты изотермы -22
        df["delta_iso22"] = df["Hmax"] - df["iso_22"]

        # Разность максимальной высоты радиоэха и высоты тропопаузы
        df["delta_tropo"] = df["Hmax"] - df["tropopause"]

        # Уровень максимальной отражаемости
        df["H_Zmax"] = np.argmax(df.filter(regex="Z").values, axis=1, keepdims=True) + 1

        print("Расчёт параметров на уровне y = iso_0 + 2 km")
        # y param = Hmax * Z_level_y
        # level_y = iso_0 + 2km
        # Параметр y = Максимальная высота * Отражаемость на уровне на 2 км выше, чем нулевая изотерма
        level_y = np.round(df["iso_0"] / 1000 + 2).astype("int8")
        Z_level_y = []
        R_level_y = []
        W_level_y = []
        D_level_y = []
        F_level_y = []
        for i in tqdm(range(df.shape[0])):
            Z_level_y.append(df[f"Z{level_y[i]}"][i])
            R_level_y.append(df[f"R{level_y[i]}"][i])
            W_level_y.append(df[f"W{level_y[i]}"][i])
            D_level_y.append(df[f"D{level_y[i]}"][i])
            F_level_y.append(df[f"F{level_y[i]}"][i])

        df["Z_level_y"] = np.array(Z_level_y)
        df["R_level_y"] = np.array(R_level_y)
        df["W_level_y"] = np.array(W_level_y)
        df["D_level_y"] = np.array(D_level_y)
        df["F_level_y"] = np.array(F_level_y)

        df["y_param"] = df["Hmax"] / 1000 * df["Z_level_y"]

        # Бинарный таргет для классификации
        df["target_binary"] = np.where(df.target >= 1, 1, 0)

        print("Расчёт параметров на уровне изотермы -22")
        level_iso_22 = np.round(df["iso_22"] / 1000).astype("int8")
        Z_iso_22 = []
        R_iso_22 = []
        W_iso_22 = []
        D_iso_22 = []
        F_iso_22 = []

        for i in tqdm(range(df.shape[0])):
            Z_iso_22.append(df[f"Z{level_iso_22[i]}"][i])
            R_iso_22.append(df[f"R{level_iso_22[i]}"][i])
            W_iso_22.append(df[f"W{level_iso_22[i]}"][i])
            D_iso_22.append(df[f"D{level_iso_22[i]}"][i])
            F_iso_22.append(df[f"F{level_iso_22[i]}"][i])
        df["Z_iso_22"] = np.array(Z_iso_22)
        df["R_iso_22"] = np.array(R_iso_22)
        df["W_iso_22"] = np.array(W_iso_22)
        df["D_iso_22"] = np.array(D_iso_22)
        df["F_iso_22"] = np.array(F_iso_22)

        # g param - Тоже самое, что и y, только отражаемость уровня iso_22
        df["g_param"] = df["Hmax"] / 1000 * df["Z_iso_22"]

        # Стандартное отклонение отражаемости в ячейке
        df["Zstd"] = df.filter(regex="Z.$|Z..$").std(axis=1)

        # Сохранение pd.DataFrame в следующий файл
        df.to_parquet(Paths.interim + f"basic_classif_{station_number}.parquet")

    return df
