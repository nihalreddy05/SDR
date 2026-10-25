"""Final Decoder helpers: Demod, Quantization, Audio, Metrics."""
import numpy as np
import io
import scipy.io.wavfile as wavfile
from .input_parsers import bpsk_demodulate, qpsk_demodulate, bits_to_text

def decode_iq_to_text(iq: np.ndarray, modulation: str = "bpsk", sps: int = 8) -> str:
    if len(iq) == 0: return ""
    if modulation.lower() == "bpsk":
        bits = bpsk_demodulate(np.real(iq), sps)
    else:
        bits = qpsk_demodulate(np.real(iq), np.imag(iq), sps)
    return bits_to_text(bits)

def scramble_text(text: str, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    chars = "X$&%@!Q#MZ*~?/+<>"
    n = max(8, len(text))
    return "".join(rng.choice(list(chars), size=n))

def quantize_bpsk(iq: np.ndarray) -> np.ndarray:
    """Forces the healed analog wave back into strict +1/-1 digital BPSK values."""
    I = np.real(iq)
    Q = np.imag(iq)
    # DO NOT subtract the mean! Text signals are naturally unbalanced.
    # Threshold strictly at 0: if > 0 -> +1.0, if < 0 -> -1.0
    I_quantized = np.where(I > 0, 1.0, -1.0)
    return (I_quantized + 1j * Q).astype(np.complex64)

def iq_to_audio_bytes(iq: np.ndarray, s_pct=0.0, e_pct=1.0, sample_rate: int = 8000) -> bytes:
    if len(iq) == 0: return b""
    duration_samples = 2 * sample_rate
    mid = int(len(iq) * (s_pct + e_pct) / 2) if s_pct != 0.0 else len(iq) // 2
    start = max(0, mid - duration_samples // 2)
    end = min(len(iq), mid + duration_samples // 2)
    iq_capped = iq[start:end]
    real_sig = np.real(iq_capped).astype(np.float32)
    max_val = np.max(np.abs(real_sig)) + 1e-9
    norm_sig = real_sig / max_val
    int_sig = (norm_sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, int_sig)
    return buf.getvalue()

def compute_ber(bits_clean: np.ndarray, bits_decoded: np.ndarray) -> float:
    n = min(len(bits_clean), len(bits_decoded))
    if n == 0: return 1.0
    return float(np.mean(bits_clean[:n] != bits_decoded[:n]))

def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    n = min(len(a), len(b))
    if n == 0: return 0.0
    a, b = a[:n], b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.abs(np.vdot(a, b)) / denom)