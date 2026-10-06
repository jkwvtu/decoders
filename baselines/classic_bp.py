import numpy as np

class ClassicMinSumBP:
    """Классический Min-Sum Belief Propagation на графе Таннера."""
    def __init__(self, H, num_iter=20):
        self.H = H
        self.num_iter = num_iter
        self.m, self.n = H.shape
        self.edges = np.argwhere(H == 1)

    def decode(self, llr_batch):
        batch_size = llr_batch.shape[0]
        msg_vc = np.zeros((batch_size, self.m, self.n), dtype=np.float32)

        for c, v in self.edges:
            msg_vc[:, c, v] = llr_batch[:, v]

        for _ in range(self.num_iter):
            msg_cv = np.zeros_like(msg_vc)
            # Check node update
            for c in range(self.m):
                v_nodes = np.where(self.H[c] == 1)[0]
                for v in v_nodes:
                    other_v = [ov for ov in v_nodes if ov != v]
                    signs = np.prod(np.sign(msg_vc[:, c, other_v] + 1e-12), axis=1)
                    mins = np.min(np.abs(msg_vc[:, c, other_v]), axis=1)
                    msg_cv[:, c, v] = signs * mins * 0.8  # Нормализующий фактор Min-Sum

            # Variable node update
            for v in range(self.n):
                c_nodes = np.where(self.H[:, v] == 1)[0]
                for c in c_nodes:
                    other_c = [oc for oc in c_nodes if oc != c]
                    msg_vc[:, c, v] = llr_batch[:, v] + np.sum(msg_cv[:, other_c, v], axis=1)

        total_llr = np.copy(llr_batch)
        for v in range(self.n):
            c_nodes = np.where(self.H[:, v] == 1)[0]
            total_llr[:, v] += np.sum(msg_cv[:, c_nodes, v], axis=1)

        return (total_llr < 0).astype(np.int8)