import numpy as np
import pandas as pd
import warnings
import json
import os


def merge_dataset(folder_path, sensor_type='Respeck'):
    '''
    sensor_type: type of sensor data to choose ['Respeck', 'Thingy']
    folder_path: path of folder named like 's1862671'
    '''
    df = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if sensor_type in file_name:
            file_path = os.path.join(folder_path, file_name)
            new_df = pd.read_csv(file_path)
            df = pd.concat([df, new_df])
    df.reset_index(drop=True, inplace=True)
    return df


def select_dataset_and_fusion_label(df):
    '''change dataset to 4 activity dataset setting'''
    label_fusion = {
        "Lying down left": "Lying down",
        "Lying down on back": "Lying down",
        "Lying down on stomach": "Lying down",
        "Lying down right": "Lying down",
        "Running": "Running",
        "Walking at normal speed": "Walking",
        "Climbing stairs": "Walking",
        "Descending stairs": "Walking",
        "Desk work": "Sitting/Standing",
        "Sitting": "Sitting/Standing",
        "Sitting bent backward": "Sitting/Standing",
        "Sitting bent forward": "Sitting/Standing",
        "Standing": "Sitting/Standing"
    }
    select_labels = [label for label in label_fusion.keys()]
    df = df[df["activity_type"].isin(select_labels)]
    df = df.replace({"activity_type": label_fusion})
    return df


def slice_window_for_each_activity(respeck_data, thingy_data, activity_type, class_labels, window_size, step_size):
    '''
    respeck_data: dataframe for one person sensor data
    thingy_data: dataframe for one person sensor data
    activity type: type for distinguishing activity
    class_labels: label of classes
    window_size: window size for framing
    step_size: stride size for framing
    '''
    respeck_data = respeck_data[respeck_data['activity_type'] == activity_type]
    thingy_data = thingy_data[thingy_data['activity_type'] == activity_type]
    row_num = min(len(respeck_data), len(thingy_data))
    respeck_data = respeck_data[:row_num]
    thingy_data = thingy_data[:row_num]
    respeck_data.reset_index(drop=True, inplace=True)
    thingy_data.reset_index(drop=True, inplace=True)
    data = pd.concat([respeck_data, thingy_data], axis=1)

    columns_of_interest = ['accel_x', 'accel_y', 'accel_z',
                           'gyro_x', 'gyro_y', 'gyro_z']

    large_enough_windows = [window for window in data.rolling(
        window=window_size, min_periods=window_size) if len(window) == window_size]
    overlapping_windows = large_enough_windows[::step_size]
    window_number = 0
    for window in overlapping_windows:
        window.loc[:, 'window_id'] = window_number
        window_number += 1
    final_sliding_windows = pd.concat(
        overlapping_windows).reset_index(drop=True)
    data, label = [], []
    for _, group in final_sliding_windows.groupby('window_id'):
        data.append(group[columns_of_interest].values)
        label.append(class_labels[group["activity_type"].values[0][0]])
    return data, label


def make_dataset_for_each_folder(folder_path, window_size, step_size, class_labels, label_fusion):
    '''
    folder_path: path of folder named like 's1862671'
    window_size: window size for framing
    step_size: stride size for framing
    class_labels: label of classes
    label_fusion: whether fusion the label
    '''
    respeck_data = merge_dataset(folder_path, 'Respeck')
    thingy_data = merge_dataset(folder_path, 'Thingy')
    if label_fusion:
        respeck_data = select_dataset_and_fusion_label(respeck_data)
        thingy_data = select_dataset_and_fusion_label(thingy_data)
    respeck_act = respeck_data['activity_type'].unique()
    thingy_act = thingy_data['activity_type'].unique()
    common_act = list(set(respeck_act) & set(thingy_act))
    dataset, labels = [], []
    for activity_type in common_act:
        data, label = slice_window_for_each_activity(
            respeck_data, thingy_data, activity_type, class_labels, window_size, step_size)
        dataset.append(data)
        labels.append(label)
    dataset = np.concatenate(dataset)
    labels = np.concatenate(labels)
    return dataset, labels


def make_dataset_for_all_folders(whole_data_folder, window_size, step_size, class_labels, label_fusion, save_npz_path):
    '''
    whole_folder_path: path of folder named like 'Data' which contains folders named like 's1862671'
    window_size: window size for framing
    step_size: stride size for framing
    class_labels: label of classes
    label_fusion: whether fusion the label
    save_npz_path: path of npz data to be saved
    '''
    whole_dataset, whole_labels = [], []
    for folder_name in os.listdir(whole_data_folder):
        if folder_name[0] == 's':
            folder_path = os.path.join(whole_data_folder, folder_name)
            dataset, labels = make_dataset_for_each_folder(
                folder_path, window_size, step_size, class_labels, label_fusion)
            whole_dataset.append(dataset)
            whole_labels.append(labels)
    whole_dataset = np.concatenate(whole_dataset)
    whole_labels = np.concatenate(whole_labels)
    whole_dataset[np.isnan(whole_dataset)] = 0.0
    np.savez(save_npz_path, x=whole_dataset, y=whole_labels)
    npzfile = np.load(save_npz_path)
    whole_dataset = npzfile['x']
    whole_labels = npzfile['y']
    print(whole_dataset.shape)
    print(whole_labels.shape)
    return whole_dataset, whole_labels


def make_dataset_for_each_user(respeck_data, thingy_data, window_size, step_size, class_labels, label_fusion):
    '''
    respeck_data: dataframe for one person sensor data
    thingy_data: dataframe for one person sensor data
    window_size: window size for framing
    step_size: stride size for framing
    class_labels: label of classes
    label_fusion: whether fusion the label
    '''
    if label_fusion:
        respeck_data = select_dataset_and_fusion_label(respeck_data)
        thingy_data = select_dataset_and_fusion_label(thingy_data)
    respeck_act = respeck_data['activity_type'].unique()
    thingy_act = thingy_data['activity_type'].unique()
    common_act = list(set(respeck_act) & set(thingy_act))
    dataset, labels = [], []
    for activity_type in common_act:
        data, label = slice_window_for_each_activity(
            respeck_data, thingy_data, activity_type, class_labels, window_size, step_size)
        dataset.append(data)
        labels.append(label)
    dataset = np.concatenate(dataset)
    labels = np.concatenate(labels)
    return dataset, labels


def make_dataset_for_all_users(respeck_data_path, thingy_data_path, window_size, step_size, class_labels, label_fusion, save_npz_path):
    '''
    respeck_data_path: respeck data path for test dataset
    thingy_data_path: thingy data path for test dataset
    window_size: window size for framing
    step_size: stride size for framing
    class_labels: label of classes
    label_fusion: whether fusion the label
    save_npz_path: path of npz data to be saved
    '''
    whole_dataset, whole_labels = [], []
    respeck_test_data = pd.read_csv(respeck_data_path)
    thingy_test_data = pd.read_csv(thingy_data_path)
    respeck_users = respeck_test_data['subject_id'].unique()
    thingy_users = thingy_test_data['subject_id'].unique()
    common_users = list(set(respeck_users) & set(thingy_users))
    for subject_id in common_users:
        respeck_data = respeck_test_data[respeck_test_data['subject_id'] == subject_id]
        thingy_data = thingy_test_data[thingy_test_data['subject_id'] == subject_id]
        respeck_data.reset_index(drop=True, inplace=True)
        thingy_data.reset_index(drop=True, inplace=True)
        dataset, labels = make_dataset_for_each_user(
            respeck_data, thingy_data, window_size, step_size, class_labels, label_fusion)
        whole_dataset.append(dataset)
        whole_labels.append(labels)
    whole_dataset = np.concatenate(whole_dataset)
    whole_labels = np.concatenate(whole_labels)
    whole_dataset[np.isnan(whole_dataset)] = 0.0
    np.savez(save_npz_path, x=whole_dataset, y=whole_labels)
    npzfile = np.load(save_npz_path)
    whole_dataset = npzfile['x']
    whole_labels = npzfile['y']
    print(whole_dataset.shape)
    print(whole_labels.shape)
    return whole_dataset, whole_labels


if __name__ == '__main__':
    '''prepare respeck data for training and validation'''
    warnings.filterwarnings('ignore')

    whole_train_data_folder = './Data/train/'
    window_size, step_size = 25, 10
    label_fusion = False
    save_npz_path = './Data/train_data.npz'
    class_labels = json.load(open('./labels.json', 'r'))
    whole_dataset, whole_labels = make_dataset_for_all_folders(
        whole_train_data_folder, window_size, step_size,
        class_labels, label_fusion, save_npz_path
    )

    respeck_data_path = './Data/valid/Respeck_recordings_unseen_2022.csv'
    thingy_data_path = './Data/valid/Thingy_recordings_unseen_2022.csv'
    window_size, step_size = 25, 10
    label_fusion = False
    save_npz_path = './Data/valid_data.npz'
    class_labels = json.load(open('./labels.json', 'r'))
    whole_dataset, whole_labels = make_dataset_for_all_users(
        respeck_data_path, thingy_data_path, window_size, step_size,
        class_labels, label_fusion, save_npz_path
    )
