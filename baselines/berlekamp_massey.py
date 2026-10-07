import numpy as np


class BerlekampMasseyDecoder:
    """Истинный алгебраический декодер БЧХ с корректной индексацией битов."""

    def __init__(self, bch_code):
        self.bch = bch_code
        self.gf = bch_code.gf
        self.n = bch_code.n
        self.t = bch_code.t

    def decode(self, llr_batch):
        batch_size = llr_batch.shape[0]
        r = (llr_batch < 0).astype(np.int8)
        decoded = np.copy(r)

        for b in range(batch_size):
            y_bits = r[b]

            # 1. Синдромы (Бит на позиции i соответствует x^{n - 1 - i})
            syndromes = [0] * (2 * self.t + 1)
            has_error = False
            for j in range(1, 2 * self.t + 1):
                val = 0
                for i in range(self.n):
                    if y_bits[i]:
                        power = ((self.n - 1 - i) * j) % (self.gf.order - 1)
                        val = self.gf.add(val, self.gf.exp_table[power])
                syndromes[j] = val
                if val != 0:
                    has_error = True

            if not has_error:
                continue

            # 2. Алгоритм Берлекэмпа-Мэсси (остается без изменений, он корректен)
            lambda_poly, b_poly = [1], [1]
            l_val, k_step = 0, 1

            for r_step in range(1, 2 * self.t + 1):
                delta = syndromes[r_step]
                for i in range(1, l_val + 1):
                    if i < len(lambda_poly):
                        delta = self.gf.add(delta, self.gf.mul(lambda_poly[i], syndromes[r_step - i]))

                if delta != 0:
                    t_poly = list(lambda_poly)
                    shifted_b = [0] * k_step + b_poly
                    scaled_b = [self.gf.mul(delta, coef) for coef in shifted_b]

                    max_len = max(len(lambda_poly), len(scaled_b))
                    new_lambda = [0] * max_len
                    for idx in range(max_len):
                        c1 = lambda_poly[idx] if idx < len(lambda_poly) else 0
                        c2 = scaled_b[idx] if idx < len(scaled_b) else 0
                        new_lambda[idx] = self.gf.add(c1, c2)

                    if 2 * l_val <= r_step - 1:
                        l_val = r_step - l_val
                        inv_delta = self.gf.inv(delta)
                        b_poly = [self.gf.mul(coef, inv_delta) for coef in t_poly]
                        k_step = 1
                    else:
                        k_step += 1
                    lambda_poly = new_lambda
                else:
                    k_step += 1

            # 3. Поиск Ченя (Коррекция локатора корня)
            error_locations = []
            for i in range(self.n):
                root_power = (self.n - 1 - i) % (self.gf.order - 1)
                inv_root_power = (self.gf.order - 1 - root_power) % (self.gf.order - 1)
                inv_alpha_i = self.gf.exp_table[inv_root_power]

                eval_sum, cur_power = 0, 1
                for coef in lambda_poly:
                    eval_sum = self.gf.add(eval_sum, self.gf.mul(coef, cur_power))
                    cur_power = self.gf.mul(cur_power, inv_alpha_i)

                if eval_sum == 0:
                    error_locations.append(i)

            if len(error_locations) <= self.t and len(error_locations) > 0:
                for pos in error_locations:
                    decoded[b, pos] ^= 1

        return decoded