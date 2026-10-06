import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense

class UpdateEmbeddings(Layer):
    def __init__(self, num_msg_dims, num_hidden_units, from_to_ind, gather_ind, num_nodes, **kwargs):
        super().__init__(**kwargs)
        self._from_ind = tf.constant(from_to_ind[:, 0], dtype=tf.int32)
        self._to_ind = tf.constant(from_to_ind[:, 1], dtype=tf.int32)
        self._gather_ind = gather_ind
        self._num_nodes = num_nodes

        self._msg_mlp = Sequential([
            Dense(num_hidden_units, activation="tanh", use_bias=False),
            Dense(num_msg_dims, use_bias=False)
        ])
        self._embed_mlp = Sequential([
            Dense(num_hidden_units, activation="tanh", use_bias=False),
            Dense(num_msg_dims, use_bias=False)
        ])

    def call(self, h_from, h_to):
        features = tf.concat([
            tf.gather(h_from, self._from_ind, axis=1),
            tf.gather(h_to, self._to_ind, axis=1)
        ], axis=-1)

        messages = self._msg_mlp(features)
        messages_T = tf.transpose(messages, (1, 0, 2))
        m = tf.math.unsorted_segment_mean(messages_T, self._gather_ind, self._num_nodes)
        m = tf.transpose(m, (1, 0, 2))

        return self._embed_mlp(tf.concat([m, h_to], axis=-1))

class EW_GNN(tf.keras.Model):
    def __init__(self, H, num_embed_dims=20, num_hidden_units=40, num_iter=8, **kwargs):
        super().__init__(**kwargs)
        self.H = H
        self.num_iter = num_iter
        self.num_cn, self.num_vn = H.shape
        self.num_embed_dims = num_embed_dims

        edges = np.argwhere(H == 1)
        self.cn_gather = tf.constant(edges[:, 0], dtype=tf.int32)
        self.vn_gather = tf.constant(edges[:, 1], dtype=tf.int32)

        self.llr_to_embed = Dense(num_embed_dims, use_bias=False)
        self.embed_to_llr = Dense(1, use_bias=False)

        self.update_cn = UpdateEmbeddings(num_embed_dims, num_hidden_units, np.flip(edges, 1), self.cn_gather, self.num_cn)
        self.update_vn = UpdateEmbeddings(num_embed_dims, num_hidden_units, edges, self.vn_gather, self.num_vn)

        dummy_input = tf.zeros([1, self.num_vn], dtype=tf.float32)
        _ = self(dummy_input)

    @tf.function
    def call(self, llr):
        batch_size = tf.shape(llr)[0]
        llr_clipped = tf.clip_by_value(llr, -15.0, 15.0)

        h_vn = self.llr_to_embed(tf.expand_dims(llr_clipped, -1))
        h_cn = tf.zeros([batch_size, self.num_cn, self.num_embed_dims])

        all_iters = []
        for _ in range(self.num_iter):
            h_cn = self.update_cn(h_vn, h_cn)
            h_vn = self.update_vn(h_cn, h_vn)

            # ИСПРАВЛЕНИЕ: Добавлен Skip-Connection. Сеть учит поправки к исходному LLR!
            residual = tf.squeeze(self.embed_to_llr(h_vn), axis=-1)
            llr_out = llr_clipped + residual
            all_iters.append(llr_out)

        return all_iters