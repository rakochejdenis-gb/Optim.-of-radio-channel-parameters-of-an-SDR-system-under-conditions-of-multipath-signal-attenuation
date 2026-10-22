
import numpy as np


def generate_bits(n_bits, seed=None):
    """Рівноймовірна випадкова послідовність бітів (0/1)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n_bits)



_NORM_QPSK = np.sqrt(2.0)

def qpsk_mod(bits):
    """QPSK: 0 -> +1, 1 -> -1 на кожній осі (Gray mapping)."""
    i = 1 - 2 * bits[0::2]
    q = 1 - 2 * bits[1::2]
    return (i + 1j * q) / _NORM_QPSK



_GRAY_4PAM = np.array([[-3, -1],
                       [+3, +1]])
_NORM_16QAM = np.sqrt(10.0)

def qam16_mod(bits):
    """16-QAM (Gray): (b0,b1) -> вісь I, (b2,b3) -> вісь Q."""
    b = bits.reshape(-1, 4)
    i = _GRAY_4PAM[b[:, 0], b[:, 1]]
    q = _GRAY_4PAM[b[:, 2], b[:, 3]]
    return (i + 1j * q) / _NORM_16QAM



if __name__ == '__main__':

    N_BITS = 16
    bits = generate_bits(N_BITS, seed=42)

    print(f"\n Згенеровано {N_BITS} випадкових бітів (seed=42):")
    print("    Біти:", bits)
    print(f"    Частка одиниць: {np.mean(bits):.2f} "
          f"(теоретично очікується ~0.50 для рівноймовірного джерела)")

    qpsk_symbols = qpsk_mod(bits)
    print(f"\n Модуляція QPSK (2 біти -> 1 символ):")
    print(f"    Отримано {len(qpsk_symbols)} символів з {N_BITS} бітів")
    print("    Пари бітів -> символ (I + jQ):")
    for k in range(len(qpsk_symbols)):
        b_pair = bits[2*k:2*k+2]
        print(f"      {b_pair} -> {qpsk_symbols[k]:.4f}")
    es_qpsk = np.mean(np.abs(qpsk_symbols) ** 2)
    print(f"    Середня енергія символу Es = {es_qpsk:.4f} "
          f"(очікується 1.0000 — коректне нормування)")

    qam_symbols = qam16_mod(bits)
    print(f"\n Модуляція 16-QAM (4 біти -> 1 символ):")
    print(f"    Отримано {len(qam_symbols)} символів з {N_BITS} бітів")
    print("    Четвірки бітів -> символ (I + jQ):")
    for k in range(len(qam_symbols)):
        b_quad = bits[4*k:4*k+4]
        print(f"      {b_quad} -> {qam_symbols[k]:.4f}")
    es_qam = np.mean(np.abs(qam_symbols) ** 2)
    print(f"    Середня енергія символу Es = {es_qam:.4f} "
          f"(на вибірці лише {len(qam_symbols)} символів статистично "
          f"коливається навколо 1.0)")

    N_LARGE = 200_000
    big_bits = generate_bits(N_LARGE, seed=1)
    es_qpsk_big = np.mean(np.abs(qpsk_mod(big_bits)) ** 2)
    es_qam_big = np.mean(np.abs(qam16_mod(big_bits)) ** 2)
    print(f"Перевірка нормування на великій вибірці ({N_LARGE} бітів):")
    print(f"    QPSK:   середня Es = {es_qpsk_big:.4f} (очікується 1.0000)")
    print(f"    16-QAM: середня Es = {es_qam_big:.4f} (очікується 1.0000)")
    print("    Висновок: на великій вибірці енергія обох модуляцій")
    print("    збігається з теоретичним значенням Es = 1 — нормування коректне.")
    
    print("ВИСНОВОК:")
    print(f"  - QPSK передає 2 біти/символ, {N_BITS} бітів -> "
          f"{len(qpsk_symbols)} символів")
    print(f"  - 16-QAM передає 4 біти/символ, {N_BITS} бітів -> "
          f"{len(qam_symbols)} символів")
    print("  - Обидві модуляції нормовано до Es = 1, тому їх можна коректно")
    print("    порівнювати за однакового Eb/N0 на наступних кроках.")
    
    import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

#  Побудова сузір'я сигналів 
N_PLOT_BITS_QPSK = 2000 * 2      # 2000 символів QPSK
N_PLOT_BITS_QAM = 2000 * 4       # 2000 символів 16-QAM

plot_bits_qpsk = generate_bits(N_PLOT_BITS_QPSK, seed=7)
plot_bits_qam = generate_bits(N_PLOT_BITS_QAM, seed=7)

plot_qpsk_symbols = qpsk_mod(plot_bits_qpsk)
plot_qam_symbols = qam16_mod(plot_bits_qam)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#  QPSK 
ax = axes[0]
ax.scatter(plot_qpsk_symbols.real, plot_qpsk_symbols.imag,
           color='blue', alpha=0.5, edgecolors='none', s=15)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title("Constellation Diagram: QPSK", fontsize=13)
ax.set_xlabel("In-Phase (I)")
ax.set_ylabel("Quadrature (Q)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")

# 16-QAM
ax = axes[1]
ax.scatter(plot_qam_symbols.real, plot_qam_symbols.imag,
           color='red', alpha=0.5, edgecolors='none', s=15)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_title("Constellation Diagram: 16-QAM", fontsize=13)
ax.set_xlabel("In-Phase (I)")
ax.set_ylabel("Quadrature (Q)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("constellations.png", dpi=150, bbox_inches="tight")

print(f"Побудовано сузір'я для {len(plot_qpsk_symbols)} символів QPSK")
print(f"та {len(plot_qam_symbols)} символів 16-QAM.")
print("Рисунок збережено у файл: constellations.png")
    