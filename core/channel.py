import numpy as np


class CommunicationChannel:
    """
    Канал связи: генерация LLR для BPSK (Es = 1) и AWGN.
    Поддерживает нормальный режим и стресс-тесты.
    Исправление: поддержка генерации случайных кодовых слов,
    чтобы избежать коллапса нейросетей в тривиальное решение (все нули).
    """

    def __init__(self, n, k, G=None):
        self.n = n
        self.k = k
        self.coderate = k / n
        self.G = G  # Порождающая матрица

    def snr_to_noise_var(self, ebno_db):
        ebno_lin = 10.0 ** (ebno_db / 10.0)
        return 1.0 / (2.0 * self.coderate * ebno_lin)

    def generate_llr(self, batch_size, ebno_db, burst_prob=0.0, burst_scale=10.0, return_c=False):
        noise_var = self.snr_to_noise_var(ebno_db)
        stddev = np.sqrt(noise_var)

        if self.G is not None:
            # Генерация случайных информационных слов
            u = np.random.randint(0, 2, size=(batch_size, self.k), dtype=np.int8)
            # Кодирование
            c = np.dot(u, self.G) % 2
        else:
            # Иначе инвариантность (c = 0)
            c = np.zeros((batch_size, self.n), dtype=np.int8)

        # Модуляция BPSK: 0 -> +1.0, 1 -> -1.0
        x = (1.0 - 2.0 * c).astype(np.float32)
        noise = np.random.normal(0, stddev, size=(batch_size, self.n)).astype(np.float32)

        # Добавление импульсных помех
        if burst_prob > 0.0:
            burst_mask = (np.random.rand(batch_size, self.n) < burst_prob).astype(np.float32)
            noise += burst_mask * np.random.normal(0, stddev * np.sqrt(burst_scale), size=(batch_size, self.n))

        y = x + noise
        llr = (2.0 * y / noise_var).astype(np.float32)

        if return_c:
            return llr, c
        return llr