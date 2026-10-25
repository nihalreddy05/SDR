"""Universal Signal Input Engine"""
import numpy as np
import pickle

# ------------------------------------------------------------------
# Text <-> Bits <-> IQ
# ------------------------------------------------------------------
def text_to_bits(text: str) -> np.ndarray:
    bits = []
    for ch in text:
        b = ord(ch)
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return np.array(bits, dtype=np.int32)

def bits_to_text(bits: np.ndarray) -> str:
    bits = np.asarray(bits).astype(int).reshape(-1)
    n = len(bits) // 8
    chars = []
    for i in range(n):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i * 8 + j])
        if 32 <= byte <= 126: chars.append(chr(byte))
        else: chars.append("·")
    return "".join(chars)

# ------------------------------------------------------------------
# Multimedia File <-> Bits (Required for MP3/Images)
# ------------------------------------------------------------------
def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert raw file bytes to a 1D array of bits."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.int32)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a 1D array of bits back into raw file bytes."""
    bits = np.asarray(bits).astype(np.uint8).reshape(-1)
    if len(bits) % 8 != 0:
        bits = bits[:-(len(bits) % 8)]
    arr = np.packbits(bits)
    return arr.tobytes()

# ------------------------------------------------------------------
# BPSK / QPSK modulation
# ------------------------------------------------------------------
def bpsk_modulate(bits: np.ndarray, sps: int = 8) -> np.ndarray:
    symbols = (1 - 2 * bits).astype(np.float32)
    return np.repeat(symbols, sps)

def bpsk_demodulate(iq_real: np.ndarray, sps: int = 8) -> np.ndarray:
    n = len(iq_real) // sps
    if n == 0: return np.array([], dtype=int)
    trimmed = iq_real[: n * sps]
    avg = trimmed.reshape(n, sps).mean(axis=1)
    return (avg < 0).astype(int)

def qpsk_modulate(bits: np.ndarray, sps: int = 8):
    n_sym = len(bits) // 2
    bits = bits[: n_sym * 2].reshape(n_sym, 2)
    I = np.where(bits[:, 0] == 0, 1.0, -1.0) / np.sqrt(2)
    Q = np.where(bits[:, 1] == 0, 1.0, -1.0) / np.sqrt(2)
    return np.repeat(I, sps).astype(np.float32), np.repeat(Q, sps).astype(np.float32)

def qpsk_demodulate(I: np.ndarray, Q: np.ndarray, sps: int = 8) -> np.ndarray:
    n = len(I) // sps
    if n == 0: return np.array([], dtype=int)
    I_avg = I[: n * sps].reshape(n, sps).mean(axis=1)
    Q_avg = Q[: n * sps].reshape(n, sps).mean(axis=1)
    bits0 = (I_avg < 0).astype(int)
    bits1 = (Q_avg < 0).astype(int)
    return np.stack([bits0, bits1], axis=1).reshape(-1)

def text_to_iq(text: str, modulation: str = "bpsk", sps: int = 8):
    bits = text_to_bits(text)
    if modulation.lower() == "bpsk":
        I = bpsk_modulate(bits, sps)
        Q = np.zeros_like(I)
    else:
        I, Q = qpsk_modulate(bits, sps)
    return (I + 1j * Q).astype(np.complex64)

# ------------------------------------------------------------------
# File parsers
# ------------------------------------------------------------------
def parse_npy(path: str) -> np.ndarray:
    arr = np.asarray(np.load(path, allow_pickle=True))
    if np.iscomplexobj(arr): return arr.astype(np.complex64)
    if arr.ndim == 2:
        if arr.shape[0] == 2: return (arr[0] + 1j * arr[1]).astype(np.complex64)
        if arr.shape[1] == 2: return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    if arr.ndim == 1: return arr.astype(np.complex64)
    raise ValueError(f"Unsupported .npy shape {arr.shape}")

def parse_bin(path: str, dtype=np.float32) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    if len(arr) % 2 != 0: arr = arr[:-1]
    I, Q = arr[::2], arr[1::2]
    return (I + 1j * Q).astype(np.complex64)

def parse_csv(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1: return arr.astype(np.complex64)
    if arr.shape[1] == 2: return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    if arr.shape[0] == 2: return (arr[0] + 1j * arr[1]).astype(np.complex64)
    raise ValueError("Unsupported CSV shape")

def parse_radioml(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        rml_data = pickle.load(f, encoding="latin1")
    keys = list(rml_data.keys())
    mod, snr = keys[np.random.randint(0, len(keys))]
    samples = rml_data[(mod, snr)]
    sample = samples[np.random.randint(0, len(samples))]
    iq = (sample[0] + 1j * sample[1]).astype(np.complex64)
    return np.tile(iq, 8)

# ------------------------------------------------------------------
# Synthetic generators
# ------------------------------------------------------------------
def generate_synthetic(signal_type: str = "sin", length: int = 1024, freq: float = 0.05):
    t = np.arange(length)
    if signal_type == "sin":
        I, Q = np.cos(2 * np.pi * freq * t), np.sin(2 * np.pi * freq * t)
    elif signal_type == "ofdm":
        I, Q = np.zeros(length), np.zeros(length)
        for k in range(1, 5):
            I += np.cos(2 * np.pi * freq * k * t) / k
            Q += np.sin(2 * np.pi * freq * k * t) / k
        I /= np.max(np.abs(I)) + 1e-9; Q /= np.max(np.abs(Q)) + 1e-9
    elif signal_type == "prbs":
        bits = np.random.default_rng(42).integers(0, 2, length // 8)
        I = np.repeat(1 - 2 * bits, 8).astype(np.float32)
        Q = np.zeros_like(I)
    else:
        I, Q = np.zeros(length), np.zeros(length)
    return (I + 1j * Q).astype(np.complex64)