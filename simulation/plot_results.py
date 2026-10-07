import matplotlib.pyplot as plt


def plot_ber_fer_curves(snr_grid, ber_dict, fer_dict, title_prefix="BCH(63,45)", filename="benchmark_results.png"):
    """
    Построение двухпанельного графика (BER и FER) в академическом стиле IEEE.

    snr_grid: массив значений Eb/N0 в дБ
    ber_dict: словарь вида {'Название метода': [значения BER]}
    fer_dict: словарь вида {'Название метода': [значения FER]}
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    styles = {
        'Berlekamp-Massey (Hard)': {'color': '#333333', 'marker': 'o', 'linestyle': '--', 'linewidth': 1.5},
        'Classic Min-Sum (20 it)': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-.', 'linewidth': 1.5},
        'EW-GNN (8 it, Old Model)': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-', 'linewidth': 2.0},
        'Neural Min-Sum (8 it)': {'color': '#1f77b4', 'marker': '^', 'linestyle': '-', 'linewidth': 2.0},
        'Live Neural Oracle': {'color': '#0000ff', 'marker': '*', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 10},
        'OSD-1 (ML Bound)': {'color': '#d62728', 'marker': 'v', 'linestyle': ':', 'linewidth': 2.0}
    }

    default_style = {'marker': 'x', 'linestyle': '-', 'linewidth': 1.5}

    ax_ber = axes[0]
    for label, ber_vals in ber_dict.items():
        st = styles.get(label, default_style)
        ax_ber.semilogy(snr_grid, ber_vals, label=label, **st)

    ax_ber.set_title(f'{title_prefix} — Bit Error Rate (BER)', fontsize=13, fontweight='bold')
    ax_ber.set_xlabel('$E_b/N_0$ [дБ]', fontsize=12)
    ax_ber.set_ylabel('BER', fontsize=12)
    ax_ber.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_ber.set_ylim([1e-5, 1.0])
    ax_ber.legend(fontsize=10, loc='lower left')

    ax_fer = axes[1]
    for label, fer_vals in fer_dict.items():
        st = styles.get(label, default_style)
        ax_fer.semilogy(snr_grid, fer_vals, label=label, **st)

    ax_fer.set_title(f'{title_prefix} — Frame Error Rate (FER / BLER)', fontsize=13, fontweight='bold')
    ax_fer.set_xlabel('$E_b/N_0$ [дБ]', fontsize=12)
    ax_fer.set_ylabel('FER (BLER)', fontsize=12)
    ax_fer.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_fer.set_ylim([1e-4, 1.0])
    ax_fer.legend(fontsize=10, loc='lower left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n[График сохранен]: '{filename}' (Разрешение 300 DPI)")
    plt.show()