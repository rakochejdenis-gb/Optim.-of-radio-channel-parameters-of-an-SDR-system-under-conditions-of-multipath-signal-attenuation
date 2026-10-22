
import numpy as np
from scipy.special import erfc


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

 # Додавання  AWGN до символів
def add_awgn(symbols, eb_no_db, k, code_rate=1.0, seed=None):
    """Додає AWGN до символів (Es = 1)."""
    rng = np.random.default_rng(seed)
    eb_no_lin = 10 ** (eb_no_db / 10.0)
    es_no_lin = k * code_rate * eb_no_lin
    noise_std = np.sqrt(1.0 / (2.0 * es_no_lin))
    noise = noise_std * (rng.standard_normal(len(symbols))
                         + 1j * rng.standard_normal(len(symbols)))
    return symbols + noise


def ber_theory_qpsk_awgn(eb_no_db):
    """Теоретичний BER QPSK/BPSK в AWGN: 0.5*erfc(sqrt(Eb/N0))."""
    eb_no_lin = 10 ** (eb_no_db / 10.0)
    return 0.5 * erfc(np.sqrt(eb_no_lin))



def generate_fading(n_symbols, block_len, k_factor=0.0, seed=None):
    """
    Генерує комплексний коефіцієнт завмирання h з блоковою структурою.
    Повертає масив h довжиною n_symbols (E[|h|^2] = 1).
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_symbols / block_len))

    los = np.sqrt(k_factor / (k_factor + 1))
    scatter_std = np.sqrt(1.0 / (2 * (k_factor + 1)))

    h_blocks = los + scatter_std * (rng.standard_normal(n_blocks)
                                    + 1j * rng.standard_normal(n_blocks))
    h = np.repeat(h_blocks, block_len)[:n_symbols]
    return h


def apply_fading_channel(symbols, eb_no_db, k, block_len,
                          k_factor=0.0, code_rate=1.0, seed=None):
   # Пропускає символи через канал h*s + n з когерентною компенсацією
    rng_seed_fade = None if seed is None else seed
    rng_seed_noise = None if seed is None else seed + 1_000_000

    h = generate_fading(len(symbols), block_len, k_factor, seed=rng_seed_fade)
    faded = h * symbols
    rx = add_awgn(faded, eb_no_db, k, code_rate, seed=rng_seed_noise)

    equalized = rx / h   
    return equalized, h


def ber_theory_qpsk_rayleigh(eb_no_db):
    """Теоретичний середній BER QPSK у релеївському завмиранні (K=0)."""
    g = 10 ** (eb_no_db / 10.0)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


#  Перевірка:
if __name__ == '__main__':

    N_BITS = 4_000_000
    bits = generate_bits(N_BITS, seed=200)
    tx = qpsk_mod(bits)
    BLOCK_LEN = 100  

    print("Порівняння каналів AWGN, Rayleigh (K=0) та Rician fading (K=3, K=10)")
    print(f"Блокове завмирання, довжина блоку = {BLOCK_LEN} символів")
    header = (f"{'Eb/N0':>6} | {'AWGN(сим)':>10} | {'AWGN(теор)':>10} | "
              f"{'Rayleigh(сим)':>13} | {'Rayleigh(теор)':>14} | "
              f"{'Rician fading K=3':>10} | {'Rician fading K=10':>11}")
    print(header)
    print("-" * len(header))

    for eb_no_db in [0, 5, 10, 15, 20, 25, 30]:
        rx_awgn = add_awgn(tx, eb_no_db, k=2, seed=eb_no_db)
        rec_awgn = np.empty(N_BITS, dtype=int)
        rec_awgn[0::2] = (rx_awgn.real < 0).astype(int)
        rec_awgn[1::2] = (rx_awgn.imag < 0).astype(int)
        ber_awgn = np.mean(bits != rec_awgn)
        ber_awgn_theo = ber_theory_qpsk_awgn(eb_no_db)

        rx_ray, _ = apply_fading_channel(tx, eb_no_db, k=2, block_len=BLOCK_LEN,
                                         k_factor=0.0, seed=eb_no_db + 1)
        rec_ray = np.empty(N_BITS, dtype=int)
        rec_ray[0::2] = (rx_ray.real < 0).astype(int)
        rec_ray[1::2] = (rx_ray.imag < 0).astype(int)
        ber_ray = np.mean(bits != rec_ray)
        ber_ray_theo = ber_theory_qpsk_rayleigh(eb_no_db)

        rx_ric3, _ = apply_fading_channel(tx, eb_no_db, k=2, block_len=BLOCK_LEN,
                                          k_factor=3.0, seed=eb_no_db + 2)
        rec_ric3 = np.empty(N_BITS, dtype=int)
        rec_ric3[0::2] = (rx_ric3.real < 0).astype(int)
        rec_ric3[1::2] = (rx_ric3.imag < 0).astype(int)
        ber_ric3 = np.mean(bits != rec_ric3)

        rx_ric10, _ = apply_fading_channel(tx, eb_no_db, k=2, block_len=BLOCK_LEN,
                                           k_factor=10.0, seed=eb_no_db + 3)
        rec_ric10 = np.empty(N_BITS, dtype=int)
        rec_ric10[0::2] = (rx_ric10.real < 0).astype(int)
        rec_ric10[1::2] = (rx_ric10.imag < 0).astype(int)
        ber_ric10 = np.mean(bits != rec_ric10)

        print(f"{eb_no_db:>6} | {ber_awgn:>10.2e} | {ber_awgn_theo:>10.2e} | "
              f"{ber_ray:>13.2e} | {ber_ray_theo:>14.2e} | "
              f"{ber_ric3:>10.2e} | {ber_ric10:>11.2e}")