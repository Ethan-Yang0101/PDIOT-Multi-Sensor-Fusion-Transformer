
class TrainingConfig(object):
    save_train_npz_path = './Data/train_data.npz'
    save_valid_npz_path = './Data/valid_data.npz'
    epochs = 100
    num_layers = 1
    embed_layer_size = 64
    fc_layer_size = 256
    num_heads = 8
    dropout = 0.1
    attention_dropout = 0.1
    optimizer = 'adam'
    amsgrad = False
    label_smoothing = 0.0
    learning_rate = 1e-3
    warmup_steps = 10
    batch_size = 128
    global_clipnorm = 3.0
    num_classes = 14
