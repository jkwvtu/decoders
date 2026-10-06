import numpy as np
import tensorflow as tf


class SmartReadinessBenchmark:
    """
    Бенчмарк для проверки качества сети.
    ИСПРАВЛЕНИЕ: Считаем реальный FER, а не синдромы, чтобы избежать обмана сети.
    """

    def __init__(self, target_fer=0.03, patience=15):
        self.target_fer = target_fer
        self.patience = patience
        self.best_fer = 1.0
        self.wait = 0

    def is_trained_sufficiently(self, model, channel, eval_snr=4.5):
        # Обязательно запрашиваем реальные биты c_batch!
        val_llr, val_c = channel.generate_llr(batch_size=2000, ebno_db=eval_snr, return_c=True)
        preds = model(tf.constant(val_llr))
        final_llr = preds[-1].numpy()
        bits = (final_llr < 0).astype(np.int8)

        # Честное сравнение с переданным словом
        fer = float(np.mean(np.any(bits != val_c, axis=1)))

        print(f"[SmartBenchmark] Validation FER at {eval_snr}dB: {fer:.4f} (Target: < {self.target_fer})")

        if fer <= self.target_fer:
            print(">>> Критерий достигнут: Сеть превзошла бейзлайн! <<<")
            return True

        if fer < self.best_fer:
            self.best_fer = fer
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(">>> Остановка: Модель вышла на плато сходимости. <<<")
                return True

        return False