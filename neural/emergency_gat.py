import tensorflow as tf

class EmergencyAttentionLayer(tf.keras.layers.Layer):
    """
    Экстренное внимание (Emergency Attention):
    Взвешивает влияние проверочных узлов в зависимости от значения локального синдрома.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.att_dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, msg_cv, syndrome_per_edge):
        """
        msg_cv: [batch, edges] — сообщения от проверок
        syndrome_per_edge: [batch, edges] — 1 если синдром проверки нарушен, 0 если сошелся
        """
        feats = tf.stack([msg_cv, syndrome_per_edge], axis=-1)
        weights = self.att_dense(feats)
        return msg_cv * tf.squeeze(weights, axis=-1)