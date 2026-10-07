import numpy as np

class ORBGRANDDecoder:
    """
    Ordered Reliability Bits Guessing Random Additive Noise Decoding (ORBGRAND).
    Генерирует паттерны ошибок, отсортированные по надежности, и тестирует синдром H(r ^ e) = 0.
    """
    def __init__(self, H, max_queries=1000):
        self.H = H
        self.n = H.shape[1]
        self.max_queries = max_queries

    def decode(self, llr_batch):
        batch_size = llr_batch.shape[0]
        r = (llr_batch < 0).astype(np.int8)
        decoded = np.copy(r)

        for b in range(batch_size):
            y = llr_batch[b]
            syn = np.dot(self.H, r[b]) % 2
            if not np.any(syn):
                continue

            order = np.argsort(np.abs(y))
            found = False

            # Перебор 1-битовых и 2-битовых гипотез шума
            for q in range(min(self.n, self.max_queries)):
                e = np.zeros(self.n, dtype=np.int8)
                e[order[q]] = 1
                cand = r[b] ^ e
                if not np.any(np.dot(self.H, cand) % 2):
                    decoded[b] = cand
                    found = True
                    break

            if not found and self.max_queries > self.n:
                for i in range(min(30, self.n)):
                    for j in range(i + 1, min(30, self.n)):
                        e = np.zeros(self.n, dtype=np.int8)
                        e[order[i]] = 1
                        e[order[j]] = 1
                        cand = r[b] ^ e
                        if not np.any(np.dot(self.H, cand) % 2):
                            decoded[b] = cand
                            found = True
                            break
                    if found:
                        break

        return decoded