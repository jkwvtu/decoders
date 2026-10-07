import time
import numpy as np

# 1. Алгебра и канал
from core.bch_exact import ExactBCH
from core.topology_manager import TopologyManager
from core.channel import CommunicationChannel

# 2. Классические декодеры
from baselines.berlekamp_massey import BerlekampMasseyDecoder
from baselines.osd import OSDDecoder
from baselines.classic_bp import ClassicMinSumBP

# 3. Нейросетевые модели
from neural.ew_gnn import EW_GNN
from neural.neural_min_sum import NeuralMinSum
from neural.oracle_system import LiveNeuralOracle

# 4. Обучение, Симуляция и Построение графиков
from training.trainer import train_neural_oracle
from simulation.monte_carlo import run_monte_carlo
from simulation.plot_results import plot_ber_fer_curves


def main():
    print("=" * 65)
    print("ПОЛНЫЙ СРАВНИТЕЛЬНЫЙ АНАЛИЗ ДЕКОДЕРОВ БЧХ(63, 45)")
    print("=" * 65)

    bch = ExactBCH("bch_63_45")
    channel = CommunicationChannel(bch.n, bch.k, G=bch.G)

    print(f"[Код]: n={bch.n}, k={bch.k}, t={bch.t} (исправляет до {bch.t} ошибок)")
    print(f"[Топология]: 4-циклов в матрице H: {TopologyManager.count_4_cycles(bch.H)}")

    print("\n--- Инициализация декодеров ---")
    bm_decoder = BerlekampMasseyDecoder(bch)
    bp_decoder = ClassicMinSumBP(bch.H, num_iter=20)
    osd_decoder = OSDDecoder(bch.G, order=1)

    ew_gnn_model = EW_GNN(bch.H, num_embed_dims=20, num_hidden_units=40, num_iter=8)
    nms_model = NeuralMinSum(bch.H, num_iter=8)

    # ---------------- 3. ГЛУБОКОЕ НОЧНОЕ ОБУЧЕНИЕ ----------------
    t_start = time.time()

    # 1/2: EW-GNN обучается 15 000 шагов (основное время)
    print("\n[Обучение 1/2]: EW-GNN (Архитектура NVIDIA, 15 000 шагов)...")
    train_neural_oracle(ew_gnn_model, channel, bch.H, epochs=20000, batch_size=256, lr_start=5e-4)

    # 2/2: Neural Min-Sum обучается 4 000 шагов
    print("\n[Обучение 2/2]: Neural Min-Sum (4 000 шагов)...")
    train_neural_oracle(nms_model, channel, bch.H, epochs=4000, batch_size=256, lr_start=1e-3)

    print(f"\n>>> Все нейросети успешно обучены за {(time.time() - t_start) / 60:.1f} минут! <<<\n")

    # Живой Оракул оборачивает обученную NMS и OSD
    oracle_system = LiveNeuralOracle(bch.H, neural_model=nms_model, fallback_decoder=osd_decoder)

    # ---------------- 4. Симуляция Монте-Карло ----------------
    snr_grid = np.arange(2.0, 6.5, 0.75)
    target_frame_errors = 40

    print("=" * 65)
    print(f"СТАРТ СИМУЛЯЦИИ МОНТЕ-КАРЛО (Сетка SNR: {snr_grid} дБ)")
    print("=" * 65)

    print("\n[1/6] Симуляция: Berlekamp-Massey (Hard Decision)...")
    ber_bm, fer_bm = run_monte_carlo(bm_decoder.decode, channel, snr_grid, target_fer=target_frame_errors)

    print("\n[2/6] Симуляция: Classic Min-Sum BP (20 итераций)...")
    ber_bp, fer_bp = run_monte_carlo(bp_decoder.decode, channel, snr_grid, target_fer=target_frame_errors)

    print("\n[3/6] Симуляция: EW-GNN (8 итераций, ПОЛНОСТЬЮ ОБУЧЕННАЯ)...")

    def ew_gnn_decode_wrapper(llrs):
        out = ew_gnn_model(llrs)
        return (out[-1].numpy() < 0).astype(int)

    ber_ew_gnn, fer_ew_gnn = run_monte_carlo(ew_gnn_decode_wrapper, channel, snr_grid, target_fer=target_frame_errors)

    print("\n[4/6] Симуляция: Neural Min-Sum (8 итераций, ПОЛНОСТЬЮ ОБУЧЕННАЯ)...")

    def nms_decode_wrapper(llrs):
        out = nms_model(llrs)
        return (out[-1].numpy() < 0).astype(int)

    ber_nms, fer_nms = run_monte_carlo(nms_decode_wrapper, channel, snr_grid, target_fer=target_frame_errors)

    print("\n[5/6] Симуляция: Живой Нейросетевой Оракул (Live Oracle)...")
    ber_ora, fer_ora = run_monte_carlo(oracle_system.decode, channel, snr_grid, target_fer=target_frame_errors)
    oracle_system.print_diagnostics()

    print("\n[6/6] Симуляция: OSD-1 (Maximum Likelihood Bound)...")
    ber_osd, fer_osd = run_monte_carlo(osd_decoder.decode, channel, snr_grid, target_fer=target_frame_errors)

    # ---------------- 5. Построение графиков ----------------
    ber_results = {
        'Berlekamp-Massey (Hard)': ber_bm,
        'Classic Min-Sum (20 it)': ber_bp,
        'EW-GNN (8 it, NVIDIA)': ber_ew_gnn,
        'Neural Min-Sum (8 it)': ber_nms,
        'Live Neural Oracle': ber_ora,
        'OSD-1 (ML Bound)': ber_osd
    }

    fer_results = {
        'Berlekamp-Massey (Hard)': fer_bm,
        'Classic Min-Sum (20 it)': fer_bp,
        'EW-GNN (8 it, NVIDIA)': fer_ew_gnn,
        'Neural Min-Sum (8 it)': fer_nms,
        'Live Neural Oracle': fer_ora,
        'OSD-1 (ML Bound)': fer_osd
    }

    plot_ber_fer_curves(
        snr_grid=snr_grid,
        ber_dict=ber_results,
        fer_dict=fer_results,
        title_prefix="BCH(63,45)",
        filename="benchmark_results_overnight.png"
    )


if __name__ == "__main__":
    main()