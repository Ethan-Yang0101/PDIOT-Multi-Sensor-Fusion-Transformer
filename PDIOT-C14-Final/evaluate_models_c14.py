from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import warnings


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


def make_dataset_for_all_users(respeck_data_path, thingy_data_path, window_size, step_size, class_labels, label_fusion):
    '''
    respeck_data_path: respeck data path for test dataset
    thingy_data_path: thingy data path for test dataset
    window_size: window size for framing
    step_size: stride size for framing
    class_labels: label of classes
    label_fusion: whether fusion the label
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
    return whole_dataset, whole_labels


def evaluate_model(model_path, respeck_data_path, thingy_data_path):
    '''evaluate model performance on test dataset'''
    # prepare dataset
    window_size, step_size = 25, 10
    label_fusion = False
    class_labels = {
        "Climbing stairs": 0,
        "Descending stairs": 1,
        "Desk work": 2,
        "Lying down left": 3,
        "Lying down on back": 4,
        "Lying down on stomach": 5,
        "Movement": 6,
        "Running": 7,
        "Lying down right": 8,
        "Sitting bent backward": 9,
        "Sitting bent forward": 10,
        "Sitting": 11,
        "Standing": 12,
        "Walking at normal speed": 13
    }
    whole_dataset, whole_labels = make_dataset_for_all_users(
        respeck_data_path, thingy_data_path, window_size, step_size,
        class_labels, label_fusion)
    whole_dataset = whole_dataset.astype(np.float32)
    whole_labels = whole_labels.astype(np.float32)
    # model inference
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    y_pred = []
    for sample in whole_dataset:
        interpreter.set_tensor(
            input_details[0]['index'], np.expand_dims(sample, axis=0))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.argmax(np.array(output_data[0]))
        y_pred.append(output_data)
    # classification report
    print("*" * 80)
    print("Classification report")
    print("*" * 80)
    target_names = list(class_labels.keys())
    print(classification_report(whole_labels, y_pred, target_names=target_names))
    # confusion matrix
    cm = confusion_matrix(whole_labels, y_pred, labels=list(
        class_labels.values()), normalize='true')
    fig, ax = plt.subplots(figsize=(14, 12))
    map = sns.heatmap(cm, annot=True, fmt=".2f",
                      xticklabels=target_names, yticklabels=target_names)
    map.set_xlabel('Predicted labels', fontsize=14)
    map.set_ylabel('True labels', fontsize=14)
    plt.yticks(rotation=360)
    plt.xticks(rotation=90)
    map.collections[0].colorbar.set_label("Accuracy", fontsize=14)
    plt.savefig('./confusion_matrix.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--respeck_data_path', type=str)
    parser.add_argument('--thingy_data_path', type=str)
    args = parser.parse_args()
    model_path = args.model_path
    respeck_data_path = args.respeck_data_path
    thingy_data_path = args.thingy_data_path
    evaluate_model(model_path, respeck_data_path, thingy_data_path)
