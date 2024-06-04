from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Concatenate, Dropout, Bidirectional
from tensorflow.keras.layers import Flatten, Dense, Activation, BatchNormalization, LSTM, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, Reshape, Lambda
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import GlobalAveragePooling1D
import tensorflow as tf
from spektral.layers import GCNConv,GINConv
from mamba import RMSNorm,MambaBlock,ResidualBlock,Mamba


def MKey_Net_DiladCNNBiLSTM_Attention(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype=tf.int64, name='main_input')
    key_input = Input(shape=(166,), dtype=tf.int64, name='key_input')
    net_input = Input(shape=(60, 7, 1), name='net_input')
    feature_input = Input(shape=(100,78)) # 节点特征
    adj_input = Input((100, 100), sparse=True) # 邻接矩阵

    graph = GCNConv(128, 'relu')([feature_input, adj_input]) # 图卷积
    graph = GCNConv(64, 'relu')([graph, adj_input]) # 图卷
    graph = Dropout(0.5)(graph)
    graph = GlobalAveragePooling1D()(graph)
    print(graph.shape)

    # Process molecular fingerprint
    y = Embedding(output_dim=128, input_dim=3, input_length=166)(key_input)
    y = Conv1D(16, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    

    # Process topological data
    net = Conv2D(8, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net_input)
    net = BatchNormalization()(net)
    net = Conv2D(16, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net)
    net = BatchNormalization()(net)
    net = Conv2D(32, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net)
    net = BatchNormalization()(net)

    print(y.shape)  # (?, 166, 32)
    print(net.shape)  # (?, 60, 7, 128)

    # Process sequence data
    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=2)(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)

    b = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)

    c = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Bidirectional(LSTM(48, return_sequences=True))(merge)
    print(x.shape)  # (?, 517, 100)

    # Reshape y, net, x to y_att, net_att, x_att with shape (None, 166, 128)
    # net_att = GlobalAveragePooling2D()(net)
    # net_att = Reshape((1, 128))(net_att)
    # net_att = UpSampling1D(size=166)(net_att)

    # dense_layer_y = Dense(128)
    # y_att = dense_layer_y(y)
    # dense_layer_x = Dense(128)
    # x_att = dense_layer_x(x)
    # x_att = Lambda(lambda x: x[:, :166, :])(x_att)
    # x_att = Reshape((166, 128))(x_att)

    # fuse_data = MultiHeadAttention(num_heads=8, key_dim=16)(net_att, y_att, x_att)

    # Fusion stage
    x = Flatten()(x)
    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)

    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)

    net = GlobalAveragePooling2D(data_format='channels_last')(net)
    net = Dropout(0.3)(net)

    # fuse_data = GlobalAveragePooling1D()(fuse_data)
    # fuse_data = Dropout(0.3)(fuse_data)

    x = Concatenate(axis=1)([x, y, net, graph])
    x = Dropout(0.3)(x)

    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)

    model = Model([main_input, key_input, net_input, feature_input, adj_input], output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


import keras
from keras.layers import Layer
import keras.backend as K

class FRNLayer(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(FRNLayer, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.tau = self.add_weight(name='tau', shape=(1,),
                                   initializer='zeros', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],),
                                    initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],),
                                     initializer='ones', trainable=True)
        super(FRNLayer, self).build(input_shape)

    def call(self, x):
        nu2 = K.mean(K.square(x), axis=[1, 2], keepdims=True)
        x = x * (1.0 / (K.sqrt(nu2) + K.abs(self.epsilon)))
        return K.maximum(self.gamma * x + self.beta, self.tau)

    def compute_output_shape(self, input_shape):
        return input_shape

from keras.layers import Layer

class TLULayer(Layer):
    def __init__(self, tau=0.0, **kwargs):
        super(TLULayer, self).__init__(**kwargs)
        self.tau = tau

    def call(self, x):
        return K.maximum(x, self.tau)

    def compute_output_shape(self, input_shape):
        return input_shape




def MKey_Net_DiladCNNBiLSTM_GCN_Attention(length, out_length, para, feature_vis=False):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype=tf.int64, name='main_input')
    key_input = Input(shape=(2048,), dtype=tf.int64, name='key_input')
    net_input = Input(shape=(60, 7, 1), name='net_input')
    feature_input = Input(shape=(100,78)) # 节点特征
    adj_input = Input((100, 100), sparse=True) # 邻接矩阵
    esm_input = Input(shape=(1280,), name='esm_input')
  
    esm_feature = Dense(128, activation='relu')(esm_input)
    esm_feature = Dropout(0.5)(esm_feature)
    esm_feature = Dense(64, activation='relu')(esm_feature)

    graph = GCNConv(128, 'relu')([feature_input, adj_input]) # 图卷积
    graph = GCNConv(64, 'relu')([graph, adj_input]) # 图卷
    graph = Dropout(0.5)(graph)
    graph = GlobalAveragePooling1D()(graph)

    # Process molecular fingerprint
    y = Embedding(output_dim=128, input_dim=3, input_length=2048)(key_input)
    y = Conv1D(16, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = FRNLayer()(y)
    y = TLULayer()(y)
    y = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = FRNLayer()(y)
    y = TLULayer()(y)
    y = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = FRNLayer()(y)
    y = TLULayer()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)
    y = Mamba(vocab_size=64, d_model=64, n_layer=16,name='my_mamba')(y)
    y = GlobalAveragePooling1D()(y)

    # Process topological data
    net = Conv2D(8, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net_input)
    net = BatchNormalization()(net)
    net = Conv2D(16, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net)
    net = BatchNormalization()(net)
    net = Conv2D(32, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net)
    net = BatchNormalization()(net)
    net = GlobalAveragePooling2D(data_format='channels_last')(net)
    net = Dropout(0.3)(net)


    # Process sequence data
    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    a = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=2)(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
    b = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
    c = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)
    x = Bidirectional(LSTM(80, return_sequences=True))(merge)
    x = Flatten()(x)
    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)
 
    x = Concatenate(axis=1)([x, y,net,graph,esm_feature])
    x = Dropout(0.3)(x)
    # 定义主输出层
    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=tf.keras.regularizers.l2(l2value))(x)

    if feature_vis:
        # 如果需要可视化特征，则创建一个额外的输出层，但不对它计算损失
        vis_output = Dense(5, activation='relu', name='vis_output')(x)
        output = Dense(5, activation='sigmoid', name='output')(x)  # 假设有5个输出标签
        model = Model(inputs=[main_input, key_input, net_input, feature_input, adj_input, esm_input], outputs=[output, vis_output])
    # 只对'output'计算损失
        model.compile(optimizer=Adam(lr=lr), loss={'output': 'binary_crossentropy'}, metrics={'output': 'accuracy'})
    else:
        output = Dense(5, activation='sigmoid', name='output')(x)
        model = Model(inputs=[main_input, key_input, net_input, feature_input, adj_input, esm_input], outputs=output)
        model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def MKey_Net_DiladCNNBiLSTM_GCN_GCN_AttentionCat(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype=tf.int64, name='main_input')
    key_input = Input(shape=(166,), dtype=tf.int64, name='key_input')
    net_input = Input(shape=(60, 7, 1), name='net_input')
    feature_input = Input(shape=(100,78)) # 节点特征
    adj_input = Input((100, 100), sparse=True) # 邻接矩阵

    graph = GCNConv(128, 'relu')([feature_input, adj_input]) # 图卷积
    graph = GCNConv(64, 'relu')([graph, adj_input]) # 图卷
    graph = Dropout(0.5)(graph)
    graph = GlobalAveragePooling1D()(graph)

    # Process molecular fingerprint
    y = Embedding(output_dim=128, input_dim=3, input_length=166)(key_input)
    y = Conv1D(16, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = Conv1D(32, 5, activation='relu', padding='same', kernel_regularizer=l2(l2value))(y)
    y = BatchNormalization()(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.3)(y)
    
    # Process topological data
    net = Conv2D(8, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net_input)
    net = BatchNormalization()(net)
    net = Conv2D(16, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net)
    net = BatchNormalization()(net)
    net = Conv2D(32, kernel_size=3, padding='same', activation='relu', data_format='channels_last')(net)
    net = BatchNormalization()(net)
    net = GlobalAveragePooling2D(data_format='channels_last')(net)
    net = Dropout(0.3)(net)

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)
    a = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=2)(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
    b = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=4)(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
    c = Conv1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value), dilation_rate=8)(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)
    x = Bidirectional(LSTM(50, return_sequences=True))(merge)
    x = Flatten()(x)
    x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)

    x = Concatenate(axis=1)([x, y, net, graph])
    x = Dropout(0.3)(x)
    x_expand = tf.expand_dims(x, axis=1)  # 将特征向量增加一个时间维度
    x_att = MultiHeadAttention(num_heads=8, key_dim=128)(x_expand, x_expand, x_expand)
    x = Dropout(0.3)(x_att)
    x = GlobalAveragePooling1D()(x)
    
    output = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
    model = Model([main_input, key_input, net_input, feature_input, adj_input], output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model