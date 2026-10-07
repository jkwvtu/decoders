import numpy as np


class OSDDecoder:
    def __init__(self, G, order=1):
        self.G = G
        self.k, self.n = G.shape
        self.order = order

    def _gf2_elimination_mrip(self, G_perm):
        """Гауссово исключение со свапом зависимых столбцов."""
        M = np.copy(G_perm)
        cols_order = np.arange(self.n)

        row = 0
        for col in range(self.k):
            pivot = -1
            # Ищем единицу в текущем столбце
            for r in range(row, self.k):
                if M[r, col] == 1:
                    pivot = r
                    break

            if pivot == -1:
                found_swap = False
                for swap_col in range(self.k, self.n):
                    for r in range(row, self.k):
                        if M[r, swap_col] == 1:
                            M[:, [col, swap_col]] = M[:, [swap_col, col]]
                            cols_order[[col, swap_col]] = cols_order[[swap_col, col]]
                            pivot = r
                            found_swap = True
                            break
                    if found_swap: break

                if not found_swap:
                    break

            # Стандартный шаг Гаусса
            M[[row, pivot]] = M[[pivot, row]]
            for r in range(self.k):
                if r != row and M[r, col] == 1:
                    M[r] ^= M[row]
            row += 1

        return M, cols_order

    def decode(self, llr_batch):
        batch_size = llr_batch.shape[0]
        decoded = np.zeros((batch_size, self.n), dtype=np.int8)

        for b in range(batch_size):
            y = llr_batch[b]
            perm = np.argsort(np.abs(y))[::-1]
            G_perm = np.copy(self.G[:, perm])

            G_sys, col_swaps = self._gf2_elimination_mrip(G_perm)

            # Итоговая перестановка позиций с учетом свапов
            final_perm = perm[col_swaps]
            inv_perm = np.argsort(final_perm)

            hard_sorted = (y[final_perm] < 0).astype(np.int8)
            u_base = hard_sorted[:self.k]

            best_cw = None
            max_corr = -float('inf')

            # OSD-0 и OSD-1
            patterns = [np.zeros(self.k, dtype=np.int8)]
            if self.order >= 1:
                for i in range(self.k):
                    p = np.zeros(self.k, dtype=np.int8)
                    p[i] = 1
                    patterns.append(p)

            for p in patterns:
                u_cand = u_base ^ p
                c_perm = np.dot(u_cand, G_sys) % 2

                # Метрика правдоподобия (Максимизация корреляции)
                x_cand = 1.0 - 2.0 * c_perm
                corr = np.sum(x_cand * y[final_perm])

                if corr > max_corr:
                    max_corr = corr
                    best_cw = c_perm

            decoded[b] = best_cw[inv_perm]

        return decoded