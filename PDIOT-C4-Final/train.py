from tensorflow import keras
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from config import TrainingConfig
from loss import smoothed_sparse_categorical_crossentropy
from scheduler import cosine_schedule
from HARTransformer import HARTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import json


def train_eval_model(config):

    # prepare dataset
    npzfile = np.load(config.save_train_npz_path)
    X_train = npzfile['x']
    y_train = npzfile['y']

    npzfile = np.load(config.save_valid_npz_path)
    X_test = npzfile['x']
    y_test = npzfile['y']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        train_size=0.8, test_size=0.2,
        shuffle=True, random_state=32,
        stratify=y_train
    )

    org_X_val, org_y_val = X_val, y_val

    # Generate new model
    model = HARTransformer(
        num_layers=config.num_layers,
        embed_dim=config.embed_layer_size,
        mlp_dim=config.fc_layer_size,
        num_heads=config.num_heads,
        num_classes=config.num_classes,
        dropout_rate=config.dropout,
        attention_dropout_rate=config.attention_dropout,
    )

    # Select optimizer
    if config.optimizer == "adam":
        optim = Adam(
            global_clipnorm=config.global_clipnorm,
            amsgrad=config.amsgrad,
        )

    # set model checkpoint
    checkpoint_filepath = './model/respeck_trans_model.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True)

    # compile model
    model.compile(
        loss=smoothed_sparse_categorical_crossentropy(
            label_smoothing=config.label_smoothing),
        optimizer=optim,
        metrics=["accuracy"],
    )

    # train model
    history = model.fit(
        X_train, y_train, batch_size=config.batch_size, epochs=config.epochs,
        validation_data=(X_val, y_val),
        callbacks=[
            LearningRateScheduler(cosine_schedule(
                base_lr=config.learning_rate,
                total_steps=config.epochs,
                warmup_steps=config.warmup_steps)),
            model_checkpoint_callback,
            EarlyStopping(monitor="val_accuracy", mode='max',
                          min_delta=0.001, patience=10),
        ],
        verbose=1
    )

    # save learning history
    history_data = {}
    for k, v in history.history.items():
        history_data[k] = [float(val) for val in v]
    with open('./model/history.json', 'w', encoding='utf-8') as fp:
        json.dump(history_data, fp, indent=4)

    # evaluate model
    print("*" * 80)
    print("Classification report (2022 Valid)")
    print("*" * 80)
    model.load_weights('./model/respeck_trans_model.h5')
    y_pred = model.predict(org_X_val)
    y_pred = np.argmax(y_pred, axis=-1)
    class_labels = json.load(open('./labels.json', 'r'))
    target_names = list(class_labels.keys())
    print(classification_report(org_y_val, y_pred, target_names=target_names))

    print("*" * 80)
    print("Classification report (2021 Test)")
    print("*" * 80)
    y_pred2 = model.predict(X_test)
    y_pred2 = np.argmax(y_pred2, axis=-1)
    print(classification_report(y_test, y_pred2, target_names=target_names))

    # plot confusion matrix
    cm = confusion_matrix(org_y_val, y_pred, labels=list(
        class_labels.values()), normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    map = sns.heatmap(cm, annot=True, fmt=".2f",
                      xticklabels=target_names, yticklabels=target_names)
    map.set_xlabel('Predicted labels', fontsize=14)
    map.set_ylabel('True labels', fontsize=14)
    plt.yticks(rotation=360)
    plt.xticks(rotation=30)
    map.collections[0].colorbar.set_label("Accuracy", fontsize=14)
    plt.savefig('./model/2022_valid_conf.png', dpi=300, bbox_inches='tight')

    cm = confusion_matrix(y_test, y_pred2, labels=list(
        class_labels.values()), normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    map = sns.heatmap(cm, annot=True, fmt=".2f",
                      xticklabels=target_names, yticklabels=target_names)
    map.set_xlabel('Predicted labels', fontsize=14)
    map.set_ylabel('True labels', fontsize=14)
    plt.yticks(rotation=360)
    plt.xticks(rotation=30)
    map.collections[0].colorbar.set_label("Accuracy", fontsize=14)
    plt.savefig('./model/2021_test_conf.png', dpi=300, bbox_inches='tight')

    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("./model/model_trans_respeck.tflite", "wb").write(tflite_model)

    return


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train_eval_model(TrainingConfig)
