from tensorflow import keras
from keras.layers import MultiHeadAttention, LayerNormalization, Layer, Add, ReLU, Dense, Dropout
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.initializers import TruncatedNormal
from keras.models import Sequential
from keras import Model
import tensorflow as tf


class FeatureExtractionEncoder(Layer):
    '''Feature Extraction Encoder with Convolutional Layers'''

    def __init__(self, emded_dim, **kwargs):
        super(FeatureExtractionEncoder, self).__init__(**kwargs)
        self.embed_dim = emded_dim

        self.encoder = Sequential()
        self.encoder.add(
            Conv1D(filters=emded_dim, kernel_size=3, activation='linear'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(ReLU())
        self.encoder.add(
            Conv1D(filters=emded_dim, kernel_size=3, activation='linear'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(ReLU())
        self.encoder.add(
            Conv1D(filters=emded_dim, kernel_size=3, activation='linear'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(ReLU())
        self.encoder.add(
            Conv1D(filters=emded_dim, kernel_size=3, activation='linear'))
        self.encoder.add(BatchNormalization())
        self.encoder.add(ReLU())

    def call(self, inputs):
        return self.encoder(inputs)

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
        }
        base_config = super(FeatureExtractionEncoder, self).get_config()
        base_config.update(config)
        return base_config


class TransformerEmbeddingLayer(Layer):
    '''Positional Embedding Layer'''

    def __init__(self, embed_dim, dropout_rate, **kwargs):
        super(TransformerEmbeddingLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(rate=dropout_rate)

    def build(self, input_shape):
        super(TransformerEmbeddingLayer, self).build(input_shape)
        self.position = self.add_weight(
            name='position',
            shape=(1, input_shape[1], self.embed_dim),
            initializer=TruncatedNormal(stddev=0.02),
            trainable=True
        )

    def call(self, inputs, training):
        x = inputs + self.position
        return self.dropout(x, training=training)

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(TransformerEmbeddingLayer, self).get_config()
        base_config.update(config)
        return base_config


class TransformerEncoderLayer(Layer):
    '''Transformer Encoder Block'''

    def __init__(self, embed_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.acc_mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // 4,
            value_dim=self.embed_dim // 4,
            output_shape=self.embed_dim // 4,
            dropout=self.attention_dropout_rate,
            kernel_initializer=TruncatedNormal(stddev=0.02)
        )
        self.gyro_mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // 4,
            value_dim=self.embed_dim // 4,
            output_shape=self.embed_dim // 4,
            dropout=self.attention_dropout_rate,
            kernel_initializer=TruncatedNormal(stddev=0.02)
        )
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // 2,
            value_dim=self.embed_dim // 2,
            output_shape=self.embed_dim // 2,
            dropout=self.attention_dropout_rate,
            kernel_initializer=TruncatedNormal(stddev=0.02)
        )
        self.dense_0 = Dense(
            units=self.mlp_dim, activation='gelu',
            kernel_initializer=TruncatedNormal(stddev=0.02)
        )
        self.dense_1 = Dense(
            units=self.embed_dim, activation=None,
            kernel_initializer=TruncatedNormal(stddev=0.02)
        )

        self.dropout_0 = Dropout(rate=self.dropout_rate)
        self.dropout_1 = Dropout(rate=self.dropout_rate)
        self.norm_0 = LayerNormalization(epsilon=1e-5)
        self.norm_1 = LayerNormalization(epsilon=1e-5)
        self.add_0 = Add()
        self.add_1 = Add()

    def call(self, inputs, training):
        # HART Attention block
        x = self.norm_0(inputs)
        acc = x[:, :, :self.embed_dim // 2]
        gyro = x[:, :, self.embed_dim // 2:]
        acc = self.acc_mha(
            query=acc, value=acc, key=acc,
            training=training)
        gyro = self.gyro_mha(
            query=gyro, value=gyro, key=gyro,
            training=training)
        x = self.mha(
            query=x, value=x, key=x,
            training=training)
        x = tf.concat([acc, x, gyro], axis=2)
        x = self.dropout_0(x, training=training)
        x = self.add_0([x, inputs])

        # MLP block
        y = self.norm_1(x)
        y = self.dense_0(y)
        y = self.dense_1(y)
        y = self.dropout_1(y, training=training)
        return self.add_1([x, y])

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'mlp_dim': self.mlp_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate
        }
        base_config = super(TransformerEncoderLayer, self).get_config()
        base_config.update(config)
        return base_config


class ClassificationHeader(Layer):
    '''Classification Header for HAR'''

    def __init__(self, embed_dim, num_classes, dropout_rate, **kwargs):
        super(ClassificationHeader, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.norm = LayerNormalization(epsilon=1e-5)
        self.hidden = Dense(
            units=embed_dim,
            activation='gelu',
            kernel_initializer=TruncatedNormal(stddev=0.02)
        )
        self.dropout = Dropout(rate=self.dropout_rate)
        self.out = Dense(
            units=num_classes,
            kernel_initializer='zeros',
            activation='softmax'
        )

    def call(self, inputs, training):
        x = self.norm(inputs)
        x = self.hidden(x)
        x = self.dropout(x, training=training)
        return self.out(x)

    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(ClassificationHeader, self).get_config()
        base_config.update(config)
        return base_config


class HARTransformer(Model):
    '''Human Activity Recognition Transformer Model for HAR'''

    def __init__(self, num_layers, embed_dim, mlp_dim, num_heads, num_classes,
                 dropout_rate, attention_dropout_rate, **kwargs):
        super(HARTransformer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.acc_encoder = FeatureExtractionEncoder(self.embed_dim // 2)
        self.gyro_encoder = FeatureExtractionEncoder(self.embed_dim // 2)
        self.embedding = TransformerEmbeddingLayer(
            self.embed_dim, self.dropout_rate)
        self.encoder_layers = [TransformerEncoderLayer(
            self.embed_dim, self.mlp_dim, self.num_heads,
            self.dropout_rate, self.attention_dropout_rate)
            for _ in range(self.num_layers)]
        self.global_pool = GlobalAveragePooling1D()
        self.header = ClassificationHeader(
            self.embed_dim, self.num_classes, self.dropout_rate)

    def call(self, inputs, training):
        x1 = self.acc_encoder(inputs[:, :, :6])
        x2 = self.gyro_encoder(inputs[:, :, 6:])
        x = self.embedding(tf.concat([x1, x2], axis=2), training=training)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        x = self.global_pool(x)
        x = self.header(x, training=training)
        return x

    def get_config(self):
        config = {
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'mlp_dim': self.mlp_dim,
            'num_heads': self.num_heads,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate
        }
        base_config = super(HARTransformer, self).get_config()
        base_config.update(config)
        return base_config
