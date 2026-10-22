

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

#  Генерація бітів 
def generate_bits(n_bits, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n_bits)


#  Модулятори 
_NORM_QPSK = np.sqrt(2.0)
_GRAY_4PAM = np.array([[-3, -1],
                       [+3, +1]])
_NORM_16QAM = np.sqrt(10.0)

def qpsk_mod(bits):
    i = 1 - 2 * bits[0::2]
    q = 1 - 2 * bits[1::2]
    return (i + 1j * q) / _NORM_QPSK

def qam16_mod(bits):
    b = bits.reshape(-1, 4)
    i = _GRAY_4PAM[b[:, 0], b[:, 1]]
    q = _GRAY_4PAM[b[:, 2], b[:, 3]]
    return (i + 1j * q) / _NORM_16QAM

def modulate(bits, scheme):
    if scheme == "QPSK":
        return qpsk_mod(bits)
    if scheme == "16QAM":
        return qam16_mod(bits)
    raise ValueError("Невідома модуляція")


#  Демодулятори (приймач) 
def qpsk_demod(symbols):
    """Жорстке рішення за знаком I та Q."""
    bits = np.empty(2 * len(symbols), dtype=int)
    bits[0::2] = (symbols.real < 0).astype(int)
    bits[1::2] = (symbols.imag < 0).astype(int)
    return bits


def _pam4_bits(x):
    """Жорстке рішення однієї осі 16-QAM (x у масштабі рівнів ±1, ±3)."""
    first = (x > 0).astype(int)
    second = (np.abs(x) < 2).astype(int)
    return first, second


def qam16_demod(symbols):
    """Жорстке рішення 16-QAM: пороги 0 та ±2/sqrt(10)."""
    bi0, bi1 = _pam4_bits(symbols.real * _NORM_16QAM)
    bq0, bq1 = _pam4_bits(symbols.imag * _NORM_16QAM)
    bits = np.empty(4 * len(symbols), dtype=int)
    bits[0::4], bits[1::4], bits[2::4], bits[3::4] = bi0, bi1, bq0, bq1
    return bits


def demodulate(symbols, scheme):
    if scheme == "QPSK":
        return qpsk_demod(symbols)
    if scheme == "16QAM":
        return qam16_demod(symbols)
    raise ValueError("Невідома модуляція")


# Канал: AWGN + завмирання 
def add_awgn(symbols, eb_no_db, k, code_rate=1.0, seed=None):
    rng = np.random.default_rng(seed)
    eb_no_lin = 10 ** (eb_no_db / 10.0)
    es_no_lin = k * code_rate * eb_no_lin
    noise_std = np.sqrt(1.0 / (2.0 * es_no_lin))
    noise = noise_std * (rng.standard_normal(len(symbols))
                         + 1j * rng.standard_normal(len(symbols)))
    return symbols + noise


def generate_fading(n_symbols, block_len, k_factor=0.0, seed=None):
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_symbols / block_len))
    los = np.sqrt(k_factor / (k_factor + 1))
    scatter_std = np.sqrt(1.0 / (2 * (k_factor + 1)))
    h_blocks = los + scatter_std * (rng.standard_normal(n_blocks)
                                    + 1j * rng.standard_normal(n_blocks))
    return np.repeat(h_blocks, block_len)[:n_symbols]


def apply_fading_channel(symbols, eb_no_db, k, block_len,
                          k_factor=0.0, code_rate=1.0, seed=None):
    seed_fade = None if seed is None else seed
    seed_noise = None if seed is None else seed + 1_000_000
    h = generate_fading(len(symbols), block_len, k_factor, seed=seed_fade)
    rx = add_awgn(h * symbols, eb_no_db, k, code_rate, seed=seed_noise)
    return rx / h   

#  Код Хеммінга 
_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
])
_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
])
_SYNDROME_TO_POS = {}
for _pos in range(7):
    _val = _H[0, _pos] * 4 + _H[1, _pos] * 2 + _H[2, _pos] * 1
    _SYNDROME_TO_POS[_val] = _pos


def hamming74_encode(bits):
    d = bits.reshape(-1, 4)
    return ((d @ _G) % 2).reshape(-1)


def hamming74_decode(bits):
    """Декодер: обчислює синдром, виправляє 1 помилку/слово, повертає
    інформаційні біти."""
    c = bits.reshape(-1, 7).copy()
    syn = (c @ _H.T) % 2
    syn_val = syn[:, 0] * 4 + syn[:, 1] * 2 + syn[:, 2] * 1
    for word_idx, val in enumerate(syn_val):
        if val != 0:
            c[word_idx, _SYNDROME_TO_POS[val]] ^= 1
    return c[:, :4].reshape(-1)


def interleave(bits, depth):
    cw = bits.reshape(-1, depth, 7)
    return cw.transpose(0, 2, 1).reshape(-1)

def deinterleave(bits, depth):
    n_super = len(bits) // (depth * 7)
    cw = bits.reshape(n_super, 7, depth)
    return cw.transpose(0, 2, 1).reshape(-1)


#  Єдиний ланцюг приймача
def receive(rx_symbols, scheme, coded=False, interleave_depth=None):

    hard_bits = demodulate(rx_symbols, scheme)

    if not coded:
        return hard_bits

    if interleave_depth is not None:
        hard_bits = deinterleave(hard_bits, interleave_depth)

    return hamming74_decode(hard_bits)


#  Обчислення BER = f(Eb/N0)
def compute_ber_curve(eb_no_db_range, scheme, n_info_bits,
                       k_factor=None, block_len=20,
                       coded=False, interleave_depth=None, seed=0):
  
    k = 2 if scheme == "QPSK" else 4
    code_rate = 4 / 7 if coded else 1.0

    info_bits = generate_bits(n_info_bits, seed=seed)

    if coded:
        tx_bits = hamming74_encode(info_bits)
        if interleave_depth is not None:
            tx_bits = interleave(tx_bits, interleave_depth)
    else:
        tx_bits = info_bits

    tx_symbols = modulate(tx_bits, scheme)

    ber_list = []
    for i, eb_no_db in enumerate(eb_no_db_range):
        if k_factor is None:
            rx_symbols = add_awgn(tx_symbols, eb_no_db, k, code_rate,
                                  seed=seed + i)
        else:
            rx_symbols = apply_fading_channel(tx_symbols, eb_no_db, k,
                                              block_len, k_factor,
                                              code_rate, seed=seed + i)

        rec_bits = receive(rx_symbols, scheme, coded=coded,
                           interleave_depth=interleave_depth)
        ber = np.mean(info_bits != rec_bits)
        ber_list.append(ber)

    return ber_list


#  самоперевірка + BER-таблиця
if __name__ == '__main__':
  
    print("Самоперевірка демодулятора та декодера (канал без шуму)")
    

    N_CHECK_BITS = 40_000
    DEPTH_CHECK = 10
    check_bits = generate_bits(N_CHECK_BITS, seed=500)

    for scheme in ("QPSK", "16QAM"):
        tx = modulate(check_bits, scheme)
        rec = receive(tx, scheme, coded=False)
        errors = np.sum(check_bits != rec)
        print(f"\n[{scheme}] Некодований сигнал:")
        print(f"    Передано біт: {N_CHECK_BITS}, помилок демодуляції: {errors}")

    coded_check = hamming74_encode(check_bits)
    interleaved_check = interleave(coded_check, DEPTH_CHECK)
    tx_coded_check = qpsk_mod(interleaved_check)
    rec_coded_check = receive(tx_coded_check, "QPSK", coded=True,
                              interleave_depth=DEPTH_CHECK)
    errors_coded = np.sum(check_bits != rec_coded_check)
    print(f"\n[QPSK] Кодований сигнал (Хеммінг 7,4 + чергування):")
    print(f"    Передано інформаційних біт: {N_CHECK_BITS}, помилок декодування: {errors_coded}")

    print("\nОчікуваний результат за відсутності шуму: 0 помилок для всіх варіантів.\n")

    # BER = f(Eb/N0) для набору сценаріїв 
    EB_NO_RANGE = list(range(0, 31, 5))
    N_INFO_BITS = 4_000_000
    DEPTH = 50

    scenarios = {
        "QPSK, AWGN, без кодування":
            dict(scheme="QPSK", k_factor=None, coded=False),
        "QPSK, Rayleigh, без кодування":
            dict(scheme="QPSK", k_factor=0.0, coded=False),
        "QPSK, Rayleigh, Хеммінг без чергування.":
            dict(scheme="QPSK", k_factor=0.0, coded=True, interleave_depth=None),
        "QPSK, Rayleigh, Хеммінг + чергування":
            dict(scheme="QPSK", k_factor=0.0, coded=True, interleave_depth=DEPTH),
        "16-QAM, Rayleigh, Хеммінг + чергування":
            dict(scheme="16QAM", k_factor=0.0, coded=True, interleave_depth=DEPTH),
    }

    results = {}
    for name, params in scenarios.items():
        results[name] = compute_ber_curve(EB_NO_RANGE, n_info_bits=N_INFO_BITS,
                                          seed=42, **params)

    
    print(" BER = f(Eb/N0) для різних сценаріїв")
    
    col_w = 16
    print(f"{'Eb/N0, дБ':>10} | " + " | ".join(f"{name[:col_w]:>{col_w}}" for name in results))
    print("-" * (13 + len(results) * (col_w + 3)))
    for idx, eb_no_db in enumerate(EB_NO_RANGE):
        row = f"{eb_no_db:>10} | "
        row += " | ".join(f"{results[name][idx]:>{col_w}.2e}" for name in results)
        print(row)
        
        # Побудова графіка BER = f(Eb/N0)
def plot_ber_curves(eb_no_range, results, target_ber=1e-3, filename="ber_curves"):
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.dpi": 300,
    })

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for idx, (name, ber) in enumerate(results.items()):
        ax.semilogy(eb_no_range, ber,
                    marker=markers[idx % len(markers)],
                    linestyle=linestyles[idx % len(linestyles)],
                    color="black", markerfacecolor="white",
                    markersize=6, linewidth=1.1, label=name)

    # Лінія цільового порогу BER (за гіпотезою дослідження)
    ax.axhline(target_ber, color="gray", linewidth=0.8, linestyle=":", zorder=0)
    ax.text(eb_no_range[-1], target_ber * 1.3, f"$BER={target_ber:.0e}$".replace("e-0", "e-"),
            ha="right", va="bottom", fontsize=9, color="gray")

    ax.set_xlabel("$E_b/N_0$, дБ")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_xlim(min(eb_no_range), max(eb_no_range))
    ax.set_ylim(1e-6, 1)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(f"{filename}.png", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


# Побудова графіка одразу після розрахунку результатів
plot_ber_curves(EB_NO_RANGE, results, target_ber=1e-3, filename="ber_curves")
print("\nГрафік збережено у файли ber_curves.png / ber_curves.pdf")