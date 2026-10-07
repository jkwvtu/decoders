import numpy as np


class CommunicationChannel:
    """Канал связи с поддержкой per-sample SNR (принцип NVIDIA)."""

    def __init__(self, n, k, G=None):
        self.n = n
        self.k = k
        self.coderate = k / n
        self.G = G

    def snr_to_noise_var(self, ebno_db):
        ebno_lin = 10.0 ** (ebno_db / 10.0)
        return 1.0 / (2.0 * self.coderate * ebno_lin)

    def generate_llr(self, batch_size, ebno_db, return_c=False):
        # Если передан кортеж (min_snr, max_snr), генерируем вектор разных SNR для батча
        if isinstance(ebno_db, (tuple, list)):
            snr_vals = np.random.uniform(ebno_db[0], ebno_db[1], size=(batch_size, 1)).astype(np.float32)
        else:
            snr_vals = np.full((batch_size, 1), ebno_db, dtype=np.float32)

        noise_var = self.snr_to_noise_var(snr_vals)
        stddev = np.sqrt(noise_var)

        if self.G is not None:
            u = np.random.randint(0, 2, size=(batch_size, self.k), dtype=np.int8)
            c = np.dot(u, self.G) % 2
        else:
            c = np.zeros((batch_size, self.n), dtype=np.int8)

        x = (1.0 - 2.0 * c).astype(np.float32)
        noise = np.random.normal(0, 1.0, size=(batch_size, self.n)).astype(np.float32) * stddev

        y = x + noise
        llr = (2.0 * y / noise_var).astype(np.float32)

        if return_c:
            return llr, c
        return llr