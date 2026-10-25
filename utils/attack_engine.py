"""Attack Configuration Sandbox & Forensic Isolation"""
import numpy as np
import torch

def fgsm_glitch(iq: np.ndarray, classifier=None, epsilon: float = 0.15, target_mod: int = 0, device: str = "cpu") -> np.ndarray:
    """FGSM: S_glitch = S + eps * sign(grad_S L)."""
    I = np.real(iq).astype(np.float32)
    Q = np.imag(iq).astype(np.float32)
    N = len(I)

    if classifier is not None:
        try:
            x = torch.tensor(np.stack([I, Q]), dtype=torch.float32).unsqueeze(0).to(device)
            x.requires_grad = True
            mod_logits, _ = classifier(x)
            # Maximize loss toward target label (i.e. push AWAY from target).
            target = torch.tensor([target_mod], dtype=torch.long).to(device)
            loss = torch.nn.functional.cross_entropy(mod_logits, target)
            loss.backward()
            grad = x.grad.detach().cpu().numpy()[0]
            perturb = epsilon * np.sign(grad)
            I = I + perturb[0]
            Q = Q + perturb[1]
        except Exception:
            # Fallback if gradient calculation fails
            rng = np.random.default_rng()
            I = I + epsilon * np.sign(rng.standard_normal(N))
            Q = Q + epsilon * np.sign(rng.standard_normal(N))
    else:
        # Pseudo-FGSM for when no model is provided
        rng = np.random.default_rng()
        I = I + epsilon * np.sign(rng.standard_normal(N))
        Q = Q + epsilon * np.sign(rng.standard_normal(N))

    return (I + 1j * Q).astype(np.complex64)

def burst_erasure(iq: np.ndarray, start_pct: float = 0.3, end_pct: float = 0.5) -> np.ndarray:
    """Zero out a window of samples (burst jamming / data erasure)."""
    out = iq.copy()
    N = len(out)
    s, e = int(N * start_pct), int(N * end_pct)
    out[s:e] = 0.0
    return out

def awgn(iq: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Additive white Gaussian noise at given SNR (dB)."""
    sig_power = np.mean(np.abs(iq) ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))
    )
    return (iq + noise).astype(np.complex64)

def isolate_glitch(s_clean: np.ndarray, s_attacked: np.ndarray) -> np.ndarray:
    """Forensic isolation: W_glitch = S_attacked - S_clean."""
    return (s_attacked - s_clean).astype(np.complex64)