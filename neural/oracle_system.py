import numpy as np
import tensorflow as tf


class LiveNeuralOracle:
    """
    Архитектура Настоящего Оракула:
    1. Fast-Path: проверка синдрома жесткого решения.
    2. Экстренная активация Neural GNN/NMS.
    3. Самоконтроль: проверка, сошелся ли синдром.
    4. Fallback: гарантированный откат к OSD или BM.
    """

    def __init__(self, H, neural_model, fallback_decoder=None):
        self.H = H
        self.neural_model = neural_model
        self.fallback = fallback_decoder
        self.stats = {
            "total": 0,
            "fast_cleared": 0,
            "oracle_invoked": 0,
            "oracle_success": 0,
            "fallback_used": 0
        }

    def decode(self, llr_batch):
        batch_size = llr_batch.shape[0]
        self.stats["total"] += batch_size

        # 1. Fast-Path
        r = (llr_batch < 0).astype(np.int8)
        syndrome = np.dot(r, self.H.T) % 2
        clean_mask = (np.sum(syndrome, axis=1) == 0)

        self.stats["fast_cleared"] += int(np.sum(clean_mask))
        decoded = np.copy(r)

        err_indices = np.where(~clean_mask)[0]
        if len(err_indices) == 0:
            return decoded

        # 2. Активация Оракула (ИСПРАВЛЕНИЕ RETRACING)
        self.stats["oracle_invoked"] += len(err_indices)

        full_llrs = tf.constant(llr_batch, dtype=tf.float32)
        out_iters = self.neural_model(full_llrs)
        final_llr = out_iters[-1].numpy()
        oracle_bits_full = (final_llr < 0).astype(np.int8)

        # 3. Самоконтроль качества по всему батчу
        new_syn = np.dot(oracle_bits_full, self.H.T) % 2
        oracle_solved_full = (np.sum(new_syn, axis=1) == 0)

        for orig_idx in err_indices:
            if oracle_solved_full[orig_idx]:
                decoded[orig_idx] = oracle_bits_full[orig_idx]
                self.stats["oracle_success"] += 1
            else:
                # 4. Fallback
                self.stats["fallback_used"] += 1
                if self.fallback is not None:
                    fb_res = self.fallback.decode(llr_batch[orig_idx:orig_idx + 1])
                    decoded[orig_idx] = fb_res[0]
                else:
                    decoded[orig_idx] = oracle_bits_full[orig_idx]

        return decoded

    def print_diagnostics(self):
        tot = max(1, self.stats["total"])
        inv = max(1, self.stats["oracle_invoked"])
        print("\nДиагностика живого Оракула")
        print(f"Всего блоков:                {self.stats['total']}")
        print(
            f"Fast-Path (без нейросети):   {self.stats['fast_cleared']} ({self.stats['fast_cleared'] / tot * 100:.1f}%)")
        print(f"Экстренных запусков GNN:     {self.stats['oracle_invoked']}")
        print(
            f"Успех GNN (синдром закрыт):  {self.stats['oracle_success']} ({self.stats['oracle_success'] / inv * 100:.1f}%)")
        print(
            f"Срабатываний Fallback (OSD): {self.stats['fallback_used']} ({self.stats['fallback_used'] / inv * 100:.1f}%)")