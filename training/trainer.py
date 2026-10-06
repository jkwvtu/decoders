import tensorflow as tf
from training.curriculum import CurriculumScheduler
from training.smart_benchmark import SmartReadinessBenchmark


# ИСПРАВЛЕНИЕ: Увеличено число эпох (батчей) с 300 до 1500 для полноценной сходимости.
def train_neural_oracle(model, channel, H, epochs=1500, batch_size=256):
    _ = model(tf.zeros([1, channel.n], dtype=tf.float32))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    scheduler = CurriculumScheduler(epochs)
    benchmark = SmartReadinessBenchmark(target_fer=0.02, patience=15)

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

    print("--- Старт глубокого обучения (Deep Training) ---")
    for ep in range(1, epochs + 1):
        snr = scheduler.get_snr(ep)
        llr_batch, c_batch = channel.generate_llr(batch_size, ebno_db=snr, return_c=True)

        y_true_tensor = tf.constant(c_batch, dtype=tf.float32)
        llr_tensor = tf.constant(llr_batch, dtype=tf.float32)

        loss_val = train_step(llr_tensor, y_true_tensor)

        if ep % 100 == 0:  # Проверяем каждые 100 батчей
            print(f"Шаг {ep:4d}/{epochs} | SNR: {snr:.1f} dB | BCE Loss: {loss_val:.4f}")
            if benchmark.is_trained_sufficiently(model, channel, eval_snr=4.5):
                break