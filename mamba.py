import math
import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Layer, Conv1D, Dense, Embedding


class RMSNorm(Layer):
    def __init__(self, d_model, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model  # 保存 d_model 参数
        self.eps = eps
        self.scale = self.add_weight(name='scale', shape=[d_model], initializer='ones', trainable=True)

    def call(self, inputs):
        variance = tf.math.reduce_mean(tf.math.square(inputs), axis=-1, keepdims=True)
        norm_x = inputs * tf.math.rsqrt(variance + self.eps)
        return self.scale * norm_x

    def get_config(self):
        config = super(RMSNorm, self).get_config()
        config.update({
            'd_model': self.d_model,  # 现在可以安全地访问 self.d_model
            'eps': self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class MambaBlock(Layer):
    def __init__(self, d_model, d_inner, d_state, dt_rank, d_conv, conv_bias=True, use_bias=False, name_suffix="1", **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # 为每个层提供唯一的名称，这里使用了name_suffix来增加唯一性
        self.in_proj = Dense(d_inner * 2, use_bias=use_bias, name=f'in_proj{name_suffix}')
        self.conv__1d = Conv1D(filters=d_inner, kernel_size=d_conv, groups=d_inner, padding='same', use_bias=conv_bias, name=f'conv_1d{name_suffix}')
        self.x_proj = Dense(dt_rank + d_state * 2, use_bias=False, name=f'x_proj{name_suffix}')
        self.dt_proj = Dense(d_inner, use_bias=True, name=f'dt_proj{name_suffix}')

        self.A_log = self.add_weight(name=f'A_log{name_suffix}', shape=[d_inner, d_state], initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True)
        self.D = self.add_weight(name=f'D{name_suffix}', shape=[d_inner], initializer='ones', trainable=True)

        self.out_proj = Dense(d_model, name=f'out_proj{name_suffix}')

    def call(self, inputs):
        # 投影输入到更高维度并分割
#         print("MambaBlock input shape:", inputs.shape)  # 打印输入形状5.（none,166,32,166）
        x_and_res = self.in_proj(inputs)
        x, res = tf.split(x_and_res, 2, axis=-1)

        # 应用1D卷积
        x = self.conv__1d(x)
        x = tf.nn.silu(x)

        # 应用SSM
        y = self.ssm(x, inputs)
        y = y * tf.nn.silu(res)

        # 输出投影
        output = self.out_proj(y)
#         print("MambaBlock output shape:", output.shape)  # 打印输出形状

        return output

    def ssm(self, x, inputs):
        A = -tf.exp(self.A_log)
        D = self.D

        x_dbl = self.x_proj(inputs)
        delta, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)
        delta = tf.nn.softplus(self.dt_proj(delta))
#         print("Before selective_scan_tf, delta shape:", delta.shape)  # 打印delta形状 6.（none,166,32,332）
        y = self.selective_scan_tf(x, delta, A, B, C, D)
        return y

    def selective_scan_tf(self, u, delta, A, B, C, D):
#         print("selective_scan_tf input u shape:", u.shape)  # 打印输入u的形状 7.（none,166,32,332）
        b, l, d_in = tf.shape(u)[0], tf.shape(u)[1], tf.shape(u)[2]
        n = tf.shape(A)[1]

        # Discretize continuous parameters (A, B)
        deltaA = tf.exp(tf.einsum('bld,dn->bldn', delta, A))
        # Equivalent to torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = tf.einsum('bld,bln,bld->bldn', delta, B, u)
        # Equivalent to einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = tf.zeros((b, d_in, n), dtype=deltaA.dtype)  # Initial state
        ys = []

        def body(i, x, ys):
            xi = deltaA[:, i] * x + deltaB_u[:, i]
            # Update state based on discretized A and B
            yi = tf.einsum('bdn,bn->bd', xi, C[:, i, :])
            # Compute output y based on current state x and input-dependent C
            ys = ys.write(i, yi)
            return i + 1, xi, ys

        def cond(i, x, ys):
            return i < l

        ys = tf.TensorArray(dtype=u.dtype, size=l)
        _, _, ys = tf.while_loop(cond, body, [0, tf.zeros((b, d_in, n), dtype=deltaA.dtype), ys])

        y = ys.stack()  # 将ys中的元素堆叠成一个张量
        y = tf.transpose(y, [1, 0, 2])  # 调整ys的维度以匹配原始的y，注意这里只有三个维度
        y += u * D  # Add input-dependent term D * u
#         print("selective_scan_tf output y shape:", y.shape)  # 打印输出y的形状
        return y

    def get_config(self):
        config = super(MambaBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_inner': self.d_inner,
            'd_state': self.d_state,
        '    dt_rank': self.dt_rank
        # 这里不包含 d_conv 属性
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResidualBlock(Layer):
    def __init__(self, d_model, d_inner, d_state, dt_rank, d_conv, conv_bias=True, use_bias=False, name_suffix="", **kwargs):
        # 添加一个name_suffix参数来帮助生成唯一的名称
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.use_bias = use_bias
        # 使用name_suffix来确保层名称的唯一性
        self.norm = RMSNorm(d_model, name='R1MSNorm'+name_suffix)
        self.mixer = MambaBlock(d_model, d_inner, d_state, dt_rank, d_conv, conv_bias, use_bias, name='1MambaBlock'+name_suffix)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_inner': self.d_inner,
            'd_state': self.d_state,
            'dt_rank': self.dt_rank,
            'd_conv': self.d_conv,
            'conv_bias': self.conv_bias,
            'use_bias': self.use_bias
        })
        return config



from tensorflow.keras.layers import Layer, Embedding, Dense

class Mamba(Layer):
    def __init__(self, vocab_size, d_model, n_layer, d_state=16, expand=2, dt_rank='auto', d_conv=4, conv_bias=True,
                 use_bias=False, **kwargs):
        super(Mamba, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.use_bias = use_bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank

        self.m_embedding = Embedding(self.vocab_size, self.d_model)
        self.Mamba_layers = [ResidualBlock(d_model=self.d_model, d_inner=self.d_inner, d_state=self.d_state, dt_rank=self.dt_rank, d_conv=self.d_conv, conv_bias=self.conv_bias, use_bias=self.use_bias, name_suffix=str(i)) for i in range(self.n_layer)]

        self.norm_f = RMSNorm(self.d_model)
        self.lm_head = Dense(self.vocab_size, use_bias=False)

    def call(self, inputs):
        x = self.m_embedding(inputs)
        for layer in self.Mamba_layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    # 省略get_config和from_config方法，除非你需要序列化这个层的配置

    def get_config(self):
        config = super(Mamba, self).get_config()
        config.update({
        'vocab_size': self.vocab_size,  # 不需要赋值，应该是使用当前实例的属性
        'd_model': self.d_model,        # 同上
        'n_layer': self.n_layer,        # 同上
        'd_state': self.d_state,
        'expand': self.expand,
        'dt_rank': self.dt_rank,
        'd_conv': self.d_conv,
        'conv_bias': self.conv_bias,
        'use_bias': self.use_bias
        })
        return config



    @classmethod
    def from_config(cls, config):
        return cls(**config)



# import math
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Layer, Conv1D, Dense, Embedding


# class RMSNorm(Layer):
#     def __init__(self, d_model, eps=1e-5, **kwargs):
#         super().__init__(**kwargs)
#         self.d_model = d_model  # 保存 d_model 参数
#         self.eps = eps
#         self.scale = self.add_weight(name='scale', shape=[d_model], initializer='ones', trainable=True)

#     def call(self, inputs):
#         variance = tf.math.reduce_mean(tf.math.square(inputs), axis=-1, keepdims=True)
#         norm_x = inputs * tf.math.rsqrt(variance + self.eps)
#         return self.scale * norm_x

#     def get_config(self):
#         config = super(RMSNorm, self).get_config()
#         config.update({
#             'd_model3': self.d_model,  # 现在可以安全地访问 self.d_model
#             'eps3': self.eps
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


# class MambaBlock(Layer):
#     def __init__(self, d_model, d_inner, d_state, dt_rank, d_conv, conv_bias=True, use_bias=False, **kwargs):
#         super().__init__(**kwargs)
#         self.d_model = d_model
#         self.d_inner = d_inner
#         self.d_state = d_state
#         self.dt_rank = dt_rank

#         self.in_proj = Dense(d_inner * 2, use_bias=use_bias)
#         self.conv1d = Conv1D(filters=d_inner, kernel_size=d_conv, groups=d_inner, padding='same',
#                              use_bias=conv_bias)
#         self.x_proj = Dense(dt_rank + d_state * 2, use_bias=False)
#         self.dt_proj = Dense(d_inner, use_bias=True)

#         self.A_log = self.add_weight(name='A_log', shape=[d_inner, d_state],
#                                      initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
#                                      trainable=True)
#         self.D = self.add_weight(name='D', shape=[d_inner], initializer='ones', trainable=True)

#         self.out_proj = Dense(d_model)

#         self.norm = RMSNorm(d_model)  # 使用RMSNorm自定义层

#     def call(self, inputs):
#         # 投影输入到更高维度并分割
#         x_and_res = self.in_proj(inputs)
#         x, res = tf.split(x_and_res, 2, axis=-1)

#         # 应用1D卷积
#         x = self.conv1d(x)
#         x = tf.nn.silu(x)

#         # 应用SSM
#         y = self.ssm(x, inputs)
#         y = y * tf.nn.silu(res)

#         # 输出投影
#         output = self.out_proj(y)

#         # 使用RMSNorm层
#         output = self.norm(output)

#         return output

#     def ssm(self, x, inputs):
#         A = -tf.exp(self.A_log)
#         D = self.D

#         x_dbl = self.x_proj(inputs)
#         delta, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)
#         delta = tf.nn.softplus(self.dt_proj(delta))

#         y = self.selective_scan_tf(x, delta, A, B, C, D)
#         return y

#     def selective_scan_tf(self, u, delta, A, B, C, D):
#         b, l, d_in = tf.shape(u)[0], tf.shape(u)[1], tf.shape(u)[2]
#         n = tf.shape(A)[1]

#         deltaA = tf.exp(tf.einsum('bld,dn->bldn', delta, A))
#         deltaB_u = tf.einsum('bld,bln,bld->bldn', delta, B, u)

#         x = tf.zeros((b, d_in, n), dtype=deltaA.dtype)  # Initial state
#         ys = []

#         def body(i, x, ys):
#             xi = deltaA[:, i] * x + deltaB_u[:, i]
#             yi = tf.einsum('bdn,bn->bd', xi, C[:, i, :])
#             ys = ys.write(i, yi)
#             return i + 1, xi, ys

#         def cond(i, x, ys):
#             return i < l

#         ys = tf.TensorArray(dtype=u.dtype, size=l)
#         _, _, ys = tf.while_loop(cond, body, [0, tf.zeros((b, d_in, n), dtype=deltaA.dtype), ys])

#         y = ys.stack()
#         y = tf.transpose(y, [1, 0, 2])
#         y += u * D
#         return y

#     def get_config(self):
#         config = super(MambaBlock, self).get_config()
#         config.update({
#             'd_model2': self.d_model,
#             'd_inner2': self.d_inner,
#             'd_state2': self.d_state,
#             'dt_rank2': self.dt_rank
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


# class ResidualBlock(Layer):
#     def __init__(self, d_model, d_inner, d_state, dt_rank, d_conv, conv_bias=True, use_bias=False, **kwargs):
#         super().__init__(**kwargs)
#         self.d_model = d_model
#         self.d_inner = d_inner
#         self.d_state = d_state
#         self.dt_rank = dt_rank
#         self.d_conv = d_conv
#         self.conv_bias = conv_bias
#         self.use_bias = use_bias
#         self.mixer = MambaBlock(d_model, d_inner, d_state, dt_rank, d_conv, conv_bias, use_bias)

#     def call(self, inputs):
#         normed_inputs = inputs  # No RMSNorm here, as it's now handled within MambaBlock
#         mixed_outputs = self.mixer(normed_inputs)
#         outputs = inputs + mixed_outputs
#         return outputs

#     def get_config(self):
#         config = super(ResidualBlock, self).get_config()
#         config.update({
#             'd_model1': self.d_model,
#             'd_inner1': self.d_inner,
#             'd_state1': self.d_state,
#             'dt_rank1': self.dt_rank,
#             'd_conv1': self.d_conv,
#             'conv_bias1': self.conv_bias,
#             'use_bias1': self.use_bias
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)


# class Mamba(Layer):
#     def __init__(self, vocab_size, d_model, n_layer, d_state=16, expand=2, dt_rank='auto', d_conv=4, conv_bias=True,
#                  use_bias=False, **kwargs):
#         super(Mamba, self).__init__(**kwargs)
#         self.vocab_size = vocab_size + (8 - vocab_size % 8 if vocab_size % 8 != 0 else 0)
#         self.d_model = d_model
#         self.n_layer = n_layer
#         self.d_state = d_state
#         self.expand = expand
#         self.d_conv = d_conv
#         self.conv_bias = conv_bias
#         self.use_bias = use_bias
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank

#         self.embedding = Embedding(self.vocab_size, self.d_model)
#         self.Mamba_layers = [ResidualBlock(d_model=self.d_model, d_inner=self.d_inner, d_state=self.d_state, dt_rank=self.dt_rank,
#                                            d_conv=self.d_conv, conv_bias=self.conv_bias, use_bias=self.use_bias) for _ in range(self.n_layer)]
#         self.norm_f = RMSNorm(self.d_model)
#         self.lm_head = Dense(self.vocab_size, use_bias=False)

#     def call(self, inputs):
#         x = self.embedding(inputs)
#         for layer in self.Mamba_layers:
#             x = layer(x)
#         x = self.norm_f(x)
#         logits = self.lm_head(x)
#         return logits

#     def get_config(self):
#         config = super(Mamba, self).get_config()
#         config.update({
#             'vocab_size4': self.vocab_size,
#             'd_model4': self.d_model,
#             'n_layer4': self.n_layer,
#             'd_state4': self.d_state,
#             'expand4': self.expand,
#             'dt_rank4': self.dt_rank,
#             'd_conv4': self.d_conv,
#             'conv_bias4': self.conv_bias,
#             'use_bias4': self.use_bias
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
