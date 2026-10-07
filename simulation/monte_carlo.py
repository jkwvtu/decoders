import numpy as np


def run_monte_carlo(decoder_fn, channel, snr_range, target_fer=20, max_batches=50, batch_size=256):
    ber_list, fer_list = [], []

    for snr in snr_range:
        bit_errors, frame_errors = 0, 0
        total_bits, total_frames = 0, 0

        current_max_batches = max_batches if snr < 4.8 else max_batches * 4

        for b_idx in range(1, current_max_batches + 1):
            llr_batch, c_batch = channel.generate_llr(batch_size, ebno_db=snr, return_c=True)
            decoded = decoder_fn(llr_batch)

            b_err = int(np.sum(decoded != c_batch))
            f_err = int(np.sum(np.any(decoded != c_batch, axis=1)))

            bit_errors += b_err
            frame_errors += f_err
            total_bits += batch_size * channel.n
            total_frames += batch_size

            print(
                f"\rSNR {snr:4.2f} dB | Батч {b_idx:3d}/{current_max_batches} | Ошибок: {frame_errors:2d}/{target_fer}",
                end="", flush=True)

            if frame_errors >= target_fer:
                break

        ber = bit_errors / total_bits
        fer = frame_errors / total_frames
        ber_list.append(ber)
        fer_list.append(fer)
        print(
            f"\rSNR: {snr:4.2f} dB | BER: {ber:.2e} | FER: {fer:.2e} | Ошибок блоков: {frame_errors}/{total_frames}   ")

    return ber_list, fer_list