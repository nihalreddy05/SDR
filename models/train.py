"""
Production-Grade Training Script for Self-Healing SDR
Trains:
1. DualTaskClassifier (Modulation + Threat Audit)
2. UNet1DInpainter (Generative Signal Healing)

Usage: python models/train.py
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import DualTaskClassifier
from models.unet_inpainter import UNet1DInpainter
from utils.input_parsers import bpsk_modulate, qpsk_modulate, generate_synthetic
from utils.attack_engine import fgsm_glitch, burst_erasure, awgn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

# ------------------------------------------------------------------
# Data Generation Helpers
# ------------------------------------------------------------------
def get_clean_signal(L, sps=8):
    """Generates a random clean I/Q signal of length L."""
    mod_type = np.random.randint(0, 4) # BPSK, QPSK, SIN, OFDM
    if mod_type == 0:
        bits = np.random.default_rng().integers(0, 2, L // sps)
        I = bpsk_modulate(bits, sps)
        Q = np.zeros(L, dtype=np.float32)
    elif mod_type == 1:
        bits = np.random.default_rng().integers(0, 2, 2 * (L // sps))
        I, Q = qpsk_modulate(bits, sps)
        # Ensure length is exact
        I, Q = np.pad(I, (0, max(0, L - len(I))))[:L], np.pad(Q, (0, max(0, L - len(Q))))[:L]
    else:
        sig = generate_synthetic("ofdm" if mod_type == 3 else "sin", L, 0.05)
        I, Q = np.real(sig), np.imag(sig)

    return I.astype(np.float32), Q.astype(np.float32)

def apply_random_attack(iq, fgsm_model=None):
    """Applies a random attack to the signal and returns the attacked IQ."""
    attack_type = np.random.choice(["erasure", "awgn", "hybrid", "fgsm"])
    if attack_type == "fgsm":
        # Simulate FGSM or use model if provided (pass None for simplicity in training loop)
        return fgsm_glitch(iq, classifier=None, epsilon=np.random.uniform(0.1, 0.4)), (0.0, 0.0)
    elif attack_type == "erasure":
        s = np.random.uniform(0.2, 0.5)
        e = s + np.random.uniform(0.1, 0.3)
        return burst_erasure(iq, s, e), (s, e)
    elif attack_type == "awgn":
        snr = np.random.uniform(-5, 10)
        return awgn(iq, snr), (0.0, 0.0)
    else: # Hybrid
        snr = np.random.uniform(0, 15)
        s = np.random.uniform(0.2, 0.5)
        e = s + np.random.uniform(0.1, 0.2)
        iq_atk = awgn(iq, snr)
        return burst_erasure(iq_atk, s, e), (s, e)

# ------------------------------------------------------------------
# Training the Classifier (Threat Auditor)
# ------------------------------------------------------------------
def train_classifier(epochs=40, batch_size=32, lr=1e-3, L=256):
    print("Training Dual-Task Classifier...")
    model = DualTaskClassifier(num_mod_classes=4).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        X, ym, yt = [], [], []
        for _ in range(batch_size):
            I, Q = get_clean_signal(L)
            mod_label = 0 # Simplified: assuming BPSK/QPSK focus for text, or random for general
            # Real logic: infer mod_label from get_clean_signal, but random is fine for diversity
            # Let's get the mod type back
            # (Redoing logic for label)
            mod_type = np.random.randint(0, 4)
            if mod_type == 0:
                bits = np.random.default_rng().integers(0, 2, L // 8)
                I = bpsk_modulate(bits, 8); Q = np.zeros(L, dtype=np.float32); mod_label = 0
            elif mod_type == 1:
                bits = np.random.default_rng().integers(0, 2, 2 * (L // 8))
                I, Q = qpsk_modulate(bits, 8); mod_label = 1
            else:
                s = generate_synthetic("ofdm" if mod_type == 3 else "sin", L, 0.05)
                I, Q = np.real(s), np.imag(s); mod_label = mod_type

            iq_clean = I + 1j * Q
            # Randomly decide if it's clean or attacked
            is_attacked = np.random.randint(0, 2)

            if is_attacked == 0:
                iq_final = awgn(iq_clean, np.random.uniform(10, 20)) # "Clean" has natural noise
            else:
                iq_atk, _ = apply_random_attack(iq_clean)
                iq_final = iq_atk

            X.append(np.stack([np.real(iq_final), np.imag(iq_final)]))
            ym.append(mod_label)
            yt.append(is_attacked)

        X = torch.from_numpy(np.array(X, dtype=np.float32)).to(DEVICE)
        ym = torch.from_numpy(np.array(ym)).long().to(DEVICE)
        yt = torch.from_numpy(np.array(yt)).long().to(DEVICE)

        mod_logits, threat_logits = model(X)
        loss = F.cross_entropy(mod_logits, ym) + F.cross_entropy(threat_logits, yt)

        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 5 == 0:
            print(f"  Epoch {ep}/{epochs} | Loss: {loss.item():.4f}")

    os.makedirs(W_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(W_DIR, "classifier.pth"))
    print("Saved classifier.pth")

# ------------------------------------------------------------------
# Training the U-Net (Self-Healing Inpainter)
# ------------------------------------------------------------------
def train_inpainter(epochs=100, batch_size=16, lr=1e-3):
    print("Training Generative U-Net Inpainter...")
    model = UNet1DInpainter(in_channels=3, out_channels=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        # Variable Length Bucketing: Train on different lengths each batch
        current_L = np.random.choice([256, 512, 1024, 2048])

        Xb, Yb = [], []
        for _ in range(batch_size):
            I_c, Q_c = get_clean_signal(current_L)
            iq_clean = I_c + 1j * Q_c

            # Apply a heavy "Inpainting" specific attack (Erasure + Light AWGN)
            # This focuses the U-Net on reconstruction
            s = np.random.uniform(0.2, 0.6)
            e = s + np.random.uniform(0.1, 0.3)
            iq_attacked = burst_erasure(iq_clean, s, e)
            iq_attacked = awgn(iq_attacked, np.random.uniform(5, 15)) # Some noise

            # Create Mask
            mask = np.ones(current_L, dtype=np.float32)
            s_idx, e_idx = int(current_L * s), int(current_L * e)
            mask[s_idx:e_idx] = 0

            # Input: Attacked I/Q + Mask
            # Target: Clean I/Q
            Xb.append(np.stack([np.real(iq_attacked), np.imag(iq_attacked), mask]))
            Yb.append(np.stack([I_c, Q_c]))

        # Convert to Tensors
        X = torch.from_numpy(np.array(Xb, dtype=np.float32)).to(DEVICE)
        Y = torch.from_numpy(np.array(Yb, dtype=np.float32)).to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, Y)

        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 10 == 0:
            print(f"  Epoch {ep}/{epochs} | Length: {current_L} | Loss: {loss.item():.6f}")

    os.makedirs(W_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(W_DIR, "inpainter.pth"))
    print("Saved inpainter.pth")

# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("=" * 50)

    # 1. Train Classifier
    train_classifier(epochs=40)

    print("=" * 50)

    # 2. Train Inpainter
    train_inpainter(epochs=100)

    print("=" * 50)
    print("Training Complete! Weights saved in /weights folder.")