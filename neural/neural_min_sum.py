import tensorflow as tf
import numpy as np


class NeuralMinSum(tf.keras.Model):
    """
    Differentiable Neural Min-Sum декодер промышленной надежности (Numerical-Stable).
    1. Исключены микро-смещения (ликвидирована случайная инверсия знака).
    2. Реализован строгий подсчет дубликатов минимумов через segment_sum.
    3. Добавлен жесткий клиппинг сообщений от разгона на 4-циклах.
    4. Веса ограничены диапазоном [0.01, 2.0] для предотвращения отрицательных множителей.
    """

    def __init__(self, H, num_iter=8, **kwargs):
        super().__init__(**kwargs)
        self.H = H
        self.num_iter = num_iter
        self.num_cn, self.num_vn = H.shape

        edges = np.argwhere(H == 1)
        self.num_edges = edges.shape[0]
        self.cn_idx = tf.constant(edges[:, 0], dtype=tf.int32)
        self.vn_idx = tf.constant(edges[:, 1], dtype=tf.int32)

        self.w_cv = self.add_weight(shape=(self.num_edges,), initializer=tf.keras.initializers.Constant(0.8), trainable=True, name="w_cv")
        self.w_vc = self.add_weight(shape=(self.num_edges,), initializer="ones", trainable=True, name="w_vc")
        self.gamma = self.add_weight(shape=(self.num_vn,), initializer="ones", trainable=True, name="gamma")

        dummy_input = tf.zeros([1, self.num_vn], dtype=tf.float32)
        _ = self(dummy_input)

    @tf.function
    def call(self, llr):
        batch_size = tf.shape(llr)[0]

        # 1. Защита весов от ухода в отрицательную зону
        w_cv_safe = tf.clip_by_value(self.w_cv, 0.01, 2.0)
        w_vc_safe = tf.clip_by_value(self.w_vc, 0.01, 2.0)
        gamma_safe = tf.clip_by_value(self.gamma, 0.1, 3.0)

        # 2. Клиппинг входного LLR для предотвращения градиентного шока
        llr_scaled = tf.clip_by_value(llr * gamma_safe, -15.0, 15.0)
        msg_vc = tf.gather(llr_scaled, self.vn_idx, axis=1)
        all_iters = []

        for _ in range(self.num_iter):
            msg_vc_w = msg_vc * w_vc_safe

            signs = tf.math.sign(msg_vc_w)
            signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
            signs_T = tf.transpose(signs)
            sign_prod = tf.math.unsorted_segment_prod(signs_T, self.cn_idx, self.num_cn)
            sign_ext_T = tf.gather(sign_prod, self.cn_idx, axis=0) * signs_T
            sign_ext = tf.transpose(sign_ext_T)

            abs_val = tf.abs(msg_vc_w) + 1e-6
            abs_T = tf.transpose(abs_val)  # [edges, batch]

            min1 = tf.math.unsorted_segment_min(abs_T, self.cn_idx, self.num_cn)  # [num_cn, batch]
            min1_edge = tf.gather(min1, self.cn_idx, axis=0)  # [edges, batch]

            is_min = tf.cast(tf.abs(abs_T - min1_edge) < 1e-5, tf.float32)
            count_min = tf.math.unsorted_segment_sum(is_min, self.cn_idx, self.num_cn)
            count_min_edge = tf.gather(count_min, self.cn_idx, axis=0)

            INF = tf.constant(1e4, dtype=tf.float32)
            abs_masked = tf.where(is_min > 0.5, INF, abs_T)
            min2 = tf.math.unsorted_segment_min(abs_masked, self.cn_idx, self.num_cn)
            min2_edge = tf.gather(min2, self.cn_idx, axis=0)

            # МАТЕМАТИЧЕСКАЯ ТЕОРЕМА:
            # Если дубликатов >= 2, то исключение любого ребра все равно оставляет min1!
            # Только если минимум уникален (count == 1), то само это ребро получает min2.
            ext_min_T = tf.where(
                count_min_edge > 1.5,
                min1_edge,
                tf.where(is_min > 0.5, min2_edge, min1_edge)
            )

            ext_min_T = tf.maximum(ext_min_T, 1e-6)
            ext_min = tf.transpose(ext_min_T)

            msg_cv = sign_ext * ext_min * w_cv_safe

            msg_cv_T = tf.transpose(msg_cv)
            sum_cv_T = tf.math.unsorted_segment_sum(msg_cv_T, self.vn_idx, self.num_vn)
            sum_cv = tf.transpose(sum_cv_T)

            total_llr = tf.clip_by_value(llr_scaled + sum_cv, -20.0, 20.0)
            msg_vc = tf.clip_by_value(tf.gather(total_llr, self.vn_idx, axis=1) - msg_cv, -20.0, 20.0)

            all_iters.append(total_llr)

        return all_iters