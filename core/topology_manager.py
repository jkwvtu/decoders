import numpy as np


class TopologyManager:
    """
    Анализ и трансформация графа Таннера:
    1. Подсчет 4-циклов: (H * H^T)_{i,j} >= 2 порождает цикл длины 4.
    2. Генерация эквивалентных разреженных топологий с подавлением циклов.
    """

    @staticmethod
    def count_4_cycles(H):
        overlap = np.dot(H.astype(np.int32), H.T.astype(np.int32))
        cycles = 0
        m = H.shape[0]
        for i in range(m):
            for j in range(i + 1, m):
                c = overlap[i, j]
                if c > 1:
                    cycles += (c * (c - 1)) // 2
        return int(cycles)

    @staticmethod
    def generate_cycle_reduced_h(H, steps=200):
        """Эвристическое снижение плотности и 4-циклов сложением строк над GF(2)."""
        best_H = np.copy(H)
        best_cycles = TopologyManager.count_4_cycles(best_H)
        m = H.shape[0]

        for _ in range(steps):
            r1, r2 = np.random.choice(m, 2, replace=False)
            candidate_row = best_H[r1] ^ best_H[r2]
            if np.sum(candidate_row) == 0:
                continue

            candidate_H = np.copy(best_H)
            candidate_H[r1] = candidate_row

            # Проверяем, что матрица сохраняет полный ранг
            if np.linalg.matrix_rank(candidate_H.astype(float)) == m:
                cycles = TopologyManager.count_4_cycles(candidate_H)
                if cycles < best_cycles:
                    best_cycles = cycles
                    best_H = candidate_H

        return best_H, best_cycles