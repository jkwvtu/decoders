import numpy as np

class GF2m:
    """
    Точная арифметика поля Галуа GF(2^m).
    Поддерживает m=5 (p(x)=x^5+x^2+1, 0x25) и m=6 (p(x)=x^6+x+1, 0x43).
    """
    def __init__(self, m=6, prim_poly=0x43):
        self.m = m
        self.order = 1 << m
        self.prim_poly = prim_poly
        self.exp_table = np.zeros(2 * self.order, dtype=np.int32)
        self.log_table = np.zeros(self.order, dtype=np.int32)
        self._build_tables()

    def _build_tables(self):
        val = 1
        for i in range(self.order - 1):
            self.exp_table[i] = val
            self.log_table[val] = i
            val <<= 1
            if val & self.order:
                val ^= self.prim_poly
        for i in range(self.order - 1, 2 * self.order):
            self.exp_table[i] = self.exp_table[i - (self.order - 1)]

    def add(self, a, b):
        return a ^ b

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.exp_table[self.log_table[a] + self.log_table[b]]

    def inv(self, a):
        if a == 0:
            raise ZeroDivisionError("Деление на 0 в GF(2^m)")
        return self.exp_table[(self.order - 1) - self.log_table[a]]

    def poly_mul_gf2(self, p1, p2):
        """Умножение двоичных многочленов над GF(2)."""
        res = np.zeros(len(p1) + len(p2) - 1, dtype=np.int8)
        for i, c1 in enumerate(p1):
            if c1:
                for j, c2 in enumerate(p2):
                    if c2:
                        res[i + j] ^= 1
        return res