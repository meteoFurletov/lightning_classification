import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
import re
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm.contrib.itertools import product
from ipywidgets import interact, IntSlider
from IPython.display import display
from functools import reduce


def to_labels(pos_probs, threshold):
    """Make labels from given data with threshold"""
    return pos_probs >= threshold


def onlyfiles(path):
    """Get all filenames from directory
    Returns a list of filenames"""
    onlyfiles_list = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            onlyfiles_list.append(os.path.join(path, name))
    return onlyfiles_list


def deg2dec(deg):
    """Convert degrees to decimals"""
    # add degree in this format - '''59°55'18"N'''
    lat = deg
    deg, minutes, seconds, direction = re.split("[°'\"]", lat)
    return np.round(
        (float(deg) + float(minutes) / 60 + float(seconds) / (60 * 60))
        * (-1 if direction in ["W", "S"] else 1),
        5,
    )


def length_of_a_degree_of_longitude(latitude):
    """Returns the length of a degree of longitude
    (east–west distance) for earth as ellipsoid in km"""
    a = 6378137.0
    b = 6356752.3142
    e_sq = (a**2 - b**2) / a**2

    return (np.pi * 6378.1370 * np.cos(np.radians(latitude))) / (
        180 * np.sqrt(1 - e_sq * np.sin(np.radians(latitude)) ** 2)
    )


def length_of_a_degree_of_latitude(longitude):
    """Returns the length of a degree of latitude
    (north-south distance) for earth as ellipsoid in km"""
    return (
        111132.954
        - 559.822 * np.cos(2 * np.radians(longitude))
        + +1.175 * np.cos(4 * np.radians(longitude))
    ) / 1000


def coords_matrix(latitude, longitude, shape):
    """Returns a matrices of latitude and longitude from given
    center coords of radar and from given shape of image"""
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


def drop_data_bo(data, n, s, w, e):
    """Return DataFrame of blitzortung data
    deleting data beyond the given border"""
    df = data
    df_drop = df.drop(df[df.lat < s].index)
    df_drop_1 = df_drop.drop(df_drop[df_drop.lat > n].index)
    df_drop_2 = df_drop_1.drop(df_drop_1[(df_drop_1.lon < e)].index)
    df_drop_3 = df_drop_2.drop(df_drop_2[(df_drop_2.lon > w)].index)
    return df_drop_3.reset_index(drop=True)


def centers_of_cells_lat(matrix):
    """Return an array of coordinates of centers of matrices for latitude"""
    matrix = matrix[:, 0]
    matrix_mean = []
    for i in range(matrix.shape[0] - 1):
        matrix_mean.append(np.mean((matrix[i], matrix[i + 1])))
    return np.array(matrix_mean)


def centers_of_cells_lon(matrix):
    """Return an array of coordinates of centers of matrices for longitude"""
    matrix_mean = []
    for j in range(matrix.shape[1] - 1):
        for i in range(matrix.shape[0] - 1):
            matrix_mean.append(np.mean((matrix[j][i], matrix[j][i + 1])))
    return np.array(matrix_mean).reshape((200, 200))


def find_nearest(data, lat_matrix, lon_matrix):
    """Return an array of nearest position
    of lightning strike on given matrices"""
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
    """Return an array of blitzortung data matched with
    coordinates matrices"""
    matrix_bo = np.zeros((data_bo.date.nunique(), shape, shape))
    dfs = [y for x, y in data_bo.groupby("date", as_index=False)]
    for counter, value in enumerate(dfs):
        for index, row in value.iterrows():
            matrix_bo[counter, row["pos_lat"], row["pos_lon"]] = 1
    return matrix_bo


def get_same_dates(radar_list, bo_list):
    """Return a list of same dates for radar data over BO data
    matched list - list of matched filenames
    indicies_list - list of matched indicies of radar_list"""
    matched_list = []
    indices_list = np.array(())
    for i in tqdm(range(len(radar_list))):
        date = (
            radar_list[i][-21:-11] + " " + (radar_list[i][-9:-4]).replace("_", ":")
        ).replace("_", "-")
        date_dt = pd.to_datetime(date)
        if date_dt in pd.to_datetime(bo_list.date.unique()):
            matched_list.append(radar_list[i])
            indices_list = np.append(indices_list, i)
    return matched_list, indices_list


def get_npy_file(matched_list):
    """Return an array from all npy files in given directory"""
    train_images = []
    for i in tqdm(matched_list):
        data = np.load(i, allow_pickle=True)
        train_images.append(data)
    return train_images


def window_slider(data_radar, window_shape_radar):
    """Return an window slided arrays"""

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


def scorer(y_true, y_pred, clf, window_shape):
    """Return the following scores:
    HSS, ETS, CSI, Acc, F1, precision, recall, roc_auc, conf_matrix"""
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

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="flag")

    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["True", "False"])
    ax.yaxis.set_ticklabels(["True", "False"])
    plt.savefig(f"images/{clf}_conf_matrix_{window_shape}")

    hss = hss_num / hss_den
    csi = hits / (hits + misses + falsealarms)
    acc = (hits + correctnegatives) / (correctnegatives + falsealarms + misses + hits)
    f1 = 2 * ((pre * rec) / (pre + rec))
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"Accuracy is {np.round(acc,4)}")
    print(f"f1_score is {np.round(f1,4)}")
    print(f"precision_score is {np.round(pre,4)}")
    print(f"recall_score is {np.round(rec,4)}")
    print(f"roc_auc_score is {np.round(roc_auc,4)}")
    print(f"HSS is {np.round(hss,4)}")
    print(f"ETS is {np.round(ets,4)}")
    print(f"CSI is {np.round(csi,4)}")
    print(f"sedi is {np.round(sedi,4)}")

    return pd.Series((acc, f1, pre, rec, roc_auc, hss, ets, csi, sedi))


def scorer_bold_attempt(y_true, y_pred):
    """Return the following scores:
    Acc, POD, FNR, FPR, F1, roc_auc, ETS, conf_matrix"""
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


def convert_binary_to_array_phen(list_of_files, station_number):
    """Convert binary data of phenomena to npy arrays in folder separately"""
    Path(f"E://data//data_npy//{station_number}_data//phen").mkdir(
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
                shaped_array[y][x] = array_one_time[i + 4]

        np.save(
            f"E://data//data_npy//{station_number}_data//phen//{fileName[21:26]}_{fileName[28:32]}_{fileName[34:36]}_"
            f"{fileName[38:40]}__{fileName[41:43]}_{fileName[43:45]}.npy",
            shaped_array,
        )


def convert_binary_to_array_echo_top(list_of_files, station_number):
    """Convert binary data of echo top to npy arrays in folder separately"""
    Path(f"E://data//data_npy//{station_number}_data//echo_top").mkdir(
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

        np.save(
            f"E://data//data_npy//{station_number}_data//echo_top//{fileName[21:26]}_{fileName[28:32]}_"
            f"{fileName[34:36]}_{fileName[38:40]}__{fileName[41:43]}_{fileName[43:45]}.npy",
            shaped_array,
        )


def convert_binary_to_array_properties(list_of_files, station_number):
    """Convert binary data of isotherms and tropopause to npy arrays in folder separately"""
    Path(f"E://data//data_npy//{station_number}_data//prop").mkdir(
        parents=True, exist_ok=True
    )
    for fileName in tqdm(list_of_files):
        data_us = np.fromfile(fileName, dtype="ushort")
        iso_22 = np.full((200, 200), data_us[14])
        date = np.full(
            (200, 200),
            np.datetime64(
                f"{fileName[28:32]}-{fileName[34:36]}-{fileName[38:40]}"
                f"T{fileName[41:43]}:{fileName[43:45]}",
                "m",
            ).astype(np.int64),
        )

        x_coordinate = np.array([np.arange(0, 200) for _ in range(200)])
        y_coordinate = np.stack(x_coordinate, axis=1)

        stack = np.stack((iso_22, date, x_coordinate, y_coordinate), axis=-1)

        np.save(
            f"E://data//data_npy//{station_number}_data//prop//{fileName[21:26]}_{fileName[28:32]}_"
            f"{fileName[34:36]}_{fileName[38:40]}__{fileName[41:43]}_{fileName[43:45]}.npy",
            stack,
        )


def convert_binary_to_array_reflectivity(list_of_files, station_number):
    """Convert binary data of reflectivity on all levels to npy arrays in folder separately"""
    Path(f"E://data//data_npy//{station_number}_data//reflectivity").mkdir(
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

        np.save(
            f"E://data//data_npy//{station_number}_data//reflectivity//{fileName[21:26]}_{fileName[28:32]}_"
            f"{fileName[34:36]}_{fileName[38:40]}__{fileName[41:43]}_{fileName[43:45]}.npy",
            shaped_array,
        )


def convert_binary_to_array_zdr(list_of_files, station_number):
    """Convert binary data of zdr on all levels to npy arrays in folder separately"""
    Path(f"E://data//data_npy//{station_number}_data//zdr").mkdir(
        parents=True, exist_ok=True
    )

    indicies_to_delete = np.concatenate((np.arange(0, 26), np.arange(252 - 26, 252)))

    for fileName in tqdm(list_of_files):
        array_one_time = np.fromfile(fileName, dtype="byte")
        data = np.fromfile(fileName, dtype="ubyte")
        shaped_array = np.full((252, 252, 11), -100)

        iterator = np.array(())
        for i in range(1, len(data) // 514):
            iterator = np.append(
                iterator,
                np.arange(514 * i - 2 * (i - 1), 514 * i + 504 - 2 * (i - 1), 14),
            )

        for i in iterator.astype(int):
            if data[i] == 255:
                y = data[i + 1]
                x = data[i + 2]
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

        arr_deleted_0 = np.delete(shaped_array, indicies_to_delete, 0)
        arr_deleted_1 = np.delete(arr_deleted_0, indicies_to_delete, 1)

        np.save(
            f"E://data//data_npy//{station_number}_data//Zdr//{fileName[21:26]}_{fileName[28:32]}_"
            f"{fileName[34:36]}_{fileName[38:40]}__{fileName[41:43]}_{fileName[43:45]}.npy",
            arr_deleted_1,
        )


def get_stacked_array(matched_list, prop=False, zdr=False):
    """Return stacked array about reflectivity, echo top, phen"""
    matched_list_phen = []
    matched_list_echotop = []
    for elem in matched_list:
        matched_list_phen.append(elem.replace("reflectivity", "phen"))
        matched_list_echotop.append(elem.replace("reflectivity", "echo_top"))

    phen = get_npy_file(matched_list_phen)
    echotop = get_npy_file(matched_list_echotop)
    reflectivity = get_npy_file(matched_list)

    if zdr:
        matched_list_zdr = []
        for elem in matched_list:
            matched_list_zdr.append(elem.replace("reflectivity", "zdr"))
        zdr_arr = get_npy_file(matched_list_zdr)
        quater_array = np.stack((echotop, phen), axis=3)
        semi_array = np.concatenate((zdr_arr, quater_array), axis=3)
        final_radar = np.concatenate((reflectivity, semi_array), axis=3)

    elif prop:
        matched_list_prop = []
        for elem in matched_list:
            matched_list_prop.append(elem.replace("reflectivity", "prop"))
        prop_array = get_npy_file(matched_list_prop)
        quater_array = np.stack((echotop, phen), axis=3)
        semi_array = np.concatenate((quater_array, prop_array), axis=3)
        final_radar = np.concatenate((reflectivity, semi_array), axis=3)

    else:
        semi_array = np.stack((echotop, phen), axis=3)
        final_radar = np.concatenate((reflectivity, semi_array), axis=3)

    return final_radar


def delete_unmatched_bo_data(data_bo, matched_list):
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
    """useless piece of garbage and also doesn't work"""
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


def get_binary_list_from_matched_reflectivity(
    matched_list, station_number, param, file_param
):
    matched_list_new = []
    for elem in matched_list:
        elem_new = (
            f"E:\\data\\binary data\\{param}{station_number}\\G{elem[43 + 9:47 + 9]}\\M{elem[48 + 9:50 + 9]}\\"
            f"D{elem[51 + 9:53 + 9]}\\{elem[55 + 9:57 + 9]}{elem[58 + 9:60 + 9]}{elem[51 + 9:53 + 9]}."
            f"{elem[48 + 9:50 + 9]}{file_param}"
        )
        matched_list_new.append(elem_new)
    return matched_list_new


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


def decision(prob_array):
    random_array = np.random.random(size=prob_array.shape[0])
    return np.less(random_array, prob_array)


def define_hmax_greater_h22_false(data):
    """Return an arrays without zero values"""
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
    """Return an arrays without zero values"""
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


def exponential_filter(delta_data):
    decision_table = pd.read_excel("decision_table.xlsx", index_col=0)
    decision_table.columns = ["delta", "expon"]

    delta_array = np.array(decision_table["delta"])
    expon_array = np.array(decision_table["expon"])

    probabilities = []
    for elem in tqdm(delta_data):
        prob = expon_array[np.argmax(delta_array == elem)]
        probabilities.append(prob)

    probabilities = np.array(probabilities)
    filter_bool = decision(probabilities)
    return filter_bool


def freeze_header(df, num_rows=30, num_columns=10, step_rows=1, step_columns=1):
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
    def _freeze_header(last_row, last_column):
        display(
            df.iloc[
                max(0, last_row - num_rows) : last_row,
                max(0, last_column - num_columns) : last_column,
            ]
        )


def factors(n):
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )


def plot_case(index, df, reflect_18, echotop_plot):
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
            f"E:\\data\\data_npy\\26061_data\\echo_top\\26061_{time.year}_{months}_{days}__{hours}_{minutes}.npy"
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


def plot_6(data_to_plot, title):
    plt.rcParams["axes.grid"] = True
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=20)

    axes[0, 0].hist(data_to_plot["X(2,2)"], bins=np.unique(data_to_plot["X(2,2)"]))
    axes[0, 0].set_title("Center X")

    axes[1, 0].hist(
        data_to_plot["Y(2,2)"], bins=np.unique(data_to_plot["Y(2,2)"]).shape[0]
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
        data_to_plot["datetime"].dt.month,
        bins=np.unique(data_to_plot["datetime"].dt.month),
    )
    axes[0, 2].set_title("Month")
    plt.rcParams["axes.grid"] = False
