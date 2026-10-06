import numpy as np
from core.gf2_algebra import GF2m


class ExactBCH:
    """
    Математически точный генератор кодов БЧХ.
    Строит порождающий полином g(x) строго через циклотомические классы над GF(2^m).
    """

    def __init__(self, code_type="bch_63_45"):
        self.code_type = code_type
        if code_type == "bch_63_45":
            self.m, self.n, self.t = 6, 63, 3
            self.gf = GF2m(m=6, prim_poly=0x43)
        elif code_type == "bch_63_36":
            self.m, self.n, self.t = 6, 63, 5
            self.gf = GF2m(m=6, prim_poly=0x43)
        elif code_type == "bch_31_16":
            self.m, self.n, self.t = 5, 31, 3
            self.gf = GF2m(m=5, prim_poly=0x25)
        else:
            raise ValueError(f"Неизвестный тип кода: {code_type}")

        # 1. Точный расчет порождающего полинома g(x)
        self.g = self._find_generator_poly()
        self.k = self.n - (len(self.g) - 1)

        # 2. Построение систематических матриц G и H
        self.G, self.H = self._build_systematic_matrices()

    def _find_generator_poly(self):
        """Динамическое вычисление минимального полинома g(x)."""
        roots = set()
        for i in range(1, 2 * self.t + 1):
            val = i
            while val not in roots:
                roots.add(val)
                val = (val * 2) % (self.gf.order - 1)

        # Перемножаем двучлены (x - alpha^r) в поле GF(2^m)
        g_poly = [1]
        for r in roots:
            root_val = self.gf.exp_table[r]
            # Умножение текущего g_poly на (x + alpha^r)
            new_g = [0] * (len(g_poly) + 1)
            for j in range(len(g_poly)):
                new_g[j] = self.gf.add(new_g[j], g_poly[j])  # Сдвиг (умножение на x)
                new_g[j + 1] = self.gf.add(new_g[j + 1], self.gf.mul(g_poly[j], root_val))  # На alpha^r
            g_poly = new_g

        return np.array(g_poly, dtype=np.int8)

    def _poly_div_mod2(self, dividend, divisor):
        deg_g = len(divisor) - 1
        d = list(dividend)
        while len(d) >= len(divisor):
            if d[0] == 1:
                for j in range(len(divisor)):
                    d[j] ^= divisor[j]
            d.pop(0)

        rem = [0] * (deg_g - len(d)) + d
        return np.array(rem, dtype=np.int8)

    def _build_systematic_matrices(self):
        deg_g = len(self.g) - 1
        k = self.n - deg_g
        G_sys = np.zeros((k, self.n), dtype=np.int8)

        for i in range(k):
            dividend = [0] * (self.n - i)
            dividend[0] = 1

            rem = self._poly_div_mod2(dividend, self.g)
            G_sys[i, i] = 1
            G_sys[i, k:] = rem

        P = G_sys[:, k:]
        H_sys = np.hstack((P.T, np.eye(self.n - k, dtype=np.int8)))

        is_orthogonal = np.all(np.dot(G_sys.astype(int), H_sys.T.astype(int)) % 2 == 0)
        assert is_orthogonal, "КРИТИЧЕСКАЯ ОШИБКА: G и H не ортогональны!"

        return G_sys, H_sys