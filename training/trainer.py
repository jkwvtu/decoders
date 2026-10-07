import time
import numpy as np
import tensorflow as tf


def train_neural_oracle(model, channel, H, epochs=15000, batch_size=256, lr_start=5e-4):
    _ = model(tf.zeros([1, channel.n], dtype=tf.float32))

    # 1. Косинусное затухание скорости обучения: от lr_start до 1% от lr_start
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr_start,
        decay_steps=epochs,
        alpha=0.01  # Финальный lr = lr_start * 0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(llr_tensor, y_true):
        with tf.GradientTape() as tape:
            all_iters = model(llr_tensor)
            loss = 0.0
            T = len(all_iters)
            for t, l_out in enumerate(all_iters):
                weight = (t + 1) / T
                loss += weight * bce(y_true, -l_out)

        grads = tape.gradient(loss, model.trainable_variables)
        grad_var_pairs = [
            (tf.clip_by_value(g, -1.0, 1.0), v)
            for g, v in zip(grads, model.trainable_variables) if g is not None
        ]
        if grad_var_pairs:
            optimizer.apply_gradients(grad_var_pairs)
        return loss

    print()
    print(f"Старт обучения (Run: {epochs} шагов, батч {batch_size})")
    print(f"Скорость обучения: {lr_start:.1e} -> {lr_start * 0.01:.1e} (Cosine Annealing)")
    print()

    best_fer = 1.0
    best_weights = None
    start_time = time.time()

    # Шаг логирования: каждые 500 шагов
    log_interval = max(100, epochs // 30)

    for ep in range(1, epochs + 1):
        # Обучение на разнородном шуме от 2.0 до 6.0 дБ в каждом батче
        llr_batch, c_batch = channel.generate_llr(batch_size, ebno_db=(2.0, 6.0), return_c=True)
        y_true_tensor = tf.constant(c_batch, dtype=tf.float32)
        llr_tensor = tf.constant(llr_batch, dtype=tf.float32)

        loss_val = train_step(llr_tensor, y_true_tensor)

        if ep % log_interval == 0 or ep == epochs:
            # Валидация на контрольном SNR 4.5 dB (2000 блоков)
            val_llr, val_c = channel.generate_llr(2000, ebno_db=4.5, return_c=True)
            preds = model(tf.constant(val_llr))
            bits = (preds[-1].numpy() < 0).astype(np.int8)
            val_fer = float(np.mean(np.any(bits != val_c, axis=1)))
            val_ber = float(np.mean(bits != val_c))

            # Текущий LR
            current_lr = lr_schedule(ep).numpy()
            elapsed_min = (time.time() - start_time) / 60.0

            # Чекпоинт лучших весов
            is_best = ""
            if val_fer < best_fer:
                best_fer = val_fer
                best_weights = model.get_weights()
                is_best = " [*ЛУЧШИЙ ЧЕКПОИНТ*]"

            print(
                f"Шаг {ep:5d}/{epochs} [{elapsed_min:5.1f} мин] | LR: {current_lr:.1e} | Loss: {loss_val:.4f} | Val FER: {val_fer:.4f} (BER: {val_ber:.2e}){is_best}")

    # Восстанавливаем лучшие веса
    if best_weights is not None:
        model.set_weights(best_weights)
        print(f"\n[Завершено]: Загружены лучшие веса с Val FER = {best_fer:.4f}")