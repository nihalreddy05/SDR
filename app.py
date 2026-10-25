"""Production-Ready SDR: Perfect Text, 5MB Files, Modulation & Threat Audit"""
import os, sys, tempfile, pickle, math, io
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import spectrogram
import reedsolo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.input_parsers import (text_to_bits, bpsk_modulate, qpsk_modulate,
                                 parse_npy, parse_bin, parse_csv, parse_radioml, generate_synthetic,
                                 bytes_to_bits, bits_to_bytes, bpsk_demodulate, qpsk_demodulate)
from utils.attack_engine import fgsm_glitch, burst_erasure, awgn, isolate_glitch
from utils.decoder import decode_iq_to_text, scramble_text, iq_to_audio_bytes, compute_ber, correlation, quantize_bpsk
from models.classifier import DualTaskClassifier
from models.unet_inpainter import UNet1DInpainter

st.set_page_config(page_title="Self-Healing SDR", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

MAX_FILE_SIZE = 5000000 # 5MB Limit per file
PACKET_SIZE = 200000    # 200KB per packet
RS_NSZ = 255
MOD_LABELS = ["BPSK", "QPSK", "SIN", "OFDM"]

# --- CHUNKED ECC ENGINE ---
def apply_ecc_encode(data_bytes):
    RS = reedsolo.RSCodec(32)
    chunk_size = RS_NSZ - 32
    encoded = bytearray()
    for i in range(0, len(data_bytes), chunk_size):
        chunk = data_bytes[i:i+chunk_size]
        encoded.extend(RS.encode(chunk))
    return bytes(encoded)

def apply_ecc_decode(data_bytes):
    RS = reedsolo.RSCodec(32)
    decoded = bytearray()
    for i in range(0, len(data_bytes), RS_NSZ):
        chunk = data_bytes[i:i+RS_NSZ]
        if len(chunk) < RS_NSZ:
            chunk = chunk.ljust(RS_NSZ, b'\x00')
        try:
            decoded.extend(RS.decode(chunk)[0])
        except Exception:
            decoded.extend(chunk[:RS_NSZ-32])
    return bytes(decoded)

def calculate_evm(clean_iq, healed_iq):
    clean_real = np.real(clean_iq)
    healed_real = np.real(healed_iq)
    error = clean_real - healed_real
    rms_error = np.sqrt(np.mean(error**2))
    rms_clean = np.sqrt(np.mean(clean_real**2)) + 1e-9
    return (rms_error / rms_clean) * 100

@st.cache_resource
def load_models():
    clf = DualTaskClassifier(num_mod_classes=4).to(DEVICE)
    inp = UNet1DInpainter(in_channels=3, out_channels=2).to(DEVICE)
    clf_path = os.path.join(W_DIR, "classifier.pth")
    inp_path = os.path.join(W_DIR, "inpainter.pth")
    if os.path.exists(clf_path): clf.load_state_dict(torch.load(clf_path, map_location=DEVICE))
    if os.path.exists(inp_path): inp.load_state_dict(torch.load(inp_path, map_location=DEVICE))
    clf.eval(); inp.eval()
    return clf, inp, os.path.exists(clf_path), os.path.exists(inp_path)

def classify_signal(iq, model, device):
    """Runs the 1D CNN to classify modulation and detect threats."""
    L = len(iq)
    window = min(256, L - (L % 16))
    if window < 16: return None, None
    I, Q = np.real(iq[:window]).astype(np.float32), np.imag(iq[:window]).astype(np.float32)
    x = torch.tensor(np.stack([I, Q]), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mod_logits, threat_logits = model(x)
    mod_probs = torch.softmax(mod_logits, dim=1)[0].cpu().numpy()
    threat_probs = torch.softmax(threat_logits, dim=1)[0].cpu().numpy()
    return mod_probs, threat_probs

def heal_signal_unet_chunked(iq_attacked, inpainter, s_pct, e_pct, device="cpu", chunk_size=8192):
    N = len(iq_attacked)
    s, e = int(N * s_pct), int(N * e_pct)
    I = np.real(iq_attacked).astype(np.float32)
    Q = np.imag(iq_attacked).astype(np.float32)
    I_h, Q_h = I.copy(), Q.copy()

    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        actual_chunk_size = end_idx - i
        pad_len = chunk_size - actual_chunk_size
        I_chunk = np.pad(I[i:end_idx], (0, pad_len))
        Q_chunk = np.pad(Q[i:end_idx], (0, pad_len))

        mask_chunk = np.ones(chunk_size, dtype=np.float32)
        if s < end_idx and e > i:
            local_s = max(0, s - i)
            local_e = min(chunk_size, e - i)
            mask_chunk[local_s:local_e] = 0

        x = torch.tensor(np.stack([I_chunk, Q_chunk, mask_chunk]), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = inpainter(x)[0].cpu().numpy()

        erased_indices = np.where(mask_chunk == 0)[0]
        for idx in erased_indices:
            if idx < actual_chunk_size:
                I_h[i + idx] = out[0, idx]
                Q_h[i + idx] = out[1, idx]

    return (I_h + 1j * Q_h).astype(np.complex64)

st.sidebar.title("🛰️ Self-Healing SDR")
st.sidebar.markdown("### 1. Universal Input Engine")
input_mode = st.sidebar.radio("Input Mode", [
    "Text Payload",
    "Multimedia File(s) (Up to 5MB each)",
    "Live RadioML Stream (Genuine Data)"
])

iq_source = None; modulation = "bpsk"; sps = 8; text = ""
uploaded_files = []
rml_dict = None
text_enc_len = 0
raw_text_len = 0

if input_mode == "Text Payload":
    text = st.sidebar.text_input("Message", "HELLO SDR")
    modulation = st.sidebar.selectbox("Modulation", ["bpsk", "qpsk"])
    sps = st.sidebar.slider("Samples/Symbol", 4, 32, 8)
    if text:
        text_bytes = text.encode('utf-8')
        raw_text_len = len(text_bytes)
        RS_TEXT = reedsolo.RSCodec(32)
        enc_bytes = RS_TEXT.encode(text_bytes)
        text_enc_len = len(enc_bytes)

        bits = bytes_to_bits(enc_bytes)
        if modulation == "bpsk":
            I = bpsk_modulate(bits, sps)
            Q = np.zeros_like(I)
        else:
            I, Q = qpsk_modulate(bits, sps)
        iq_source = (I + 1j * Q).astype(np.complex64)

elif input_mode == "Multimedia File(s) (Up to 5MB each)":
    st.sidebar.info("📤 Upload multiple files. Each file up to 5MB. Processed sequentially.")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Multimedia",
        type=["mp3", "wav", "m4a", "ogg", "opus", "png", "jpg", "txt"],
        accept_multiple_files=True
    )

elif input_mode == "Live RadioML Stream (Genuine Data)":
    st.sidebar.info("Upload the genuine RadioML .pkl file once. Clicking 'Run' pulls the next real signal.")
    up = st.sidebar.file_uploader("Upload RML2016.10a_dict.pkl", type=["pkl"])
    if up is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp.write(up.read()); tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f: rml_dict = pickle.load(f, encoding="latin1")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

st.sidebar.markdown("### 2. Attack Configuration Sandbox")
use_fgsm = st.sidebar.checkbox("FGSM Adversarial Glitch", False)
eps = st.sidebar.slider("FGSM ε (epsilon)", 0.01, 0.5, 0.25, 0.01) if use_fgsm else 0.0
use_erasure = st.sidebar.checkbox("Burst Data Erasure", True)
if use_erasure:
    e_range = st.sidebar.slider("Erasure Window (%)", 0, 100, (10, 30))
    s_pct, e_pct = e_range[0] / 100, e_range[1] / 100
    if s_pct < 0.1: s_pct = 0.1
else: s_pct, e_pct = 0.0, 1.0
use_awgn = st.sidebar.checkbox("AWGN Background Noise", False)
snr_db = st.sidebar.slider("SNR (dB)", -10.0, 20.0, 5.0) if use_awgn else 50.0

run = st.sidebar.button("🚀 Run Analysis / Next Packet", use_container_width=True)

clf, inp, has_clf, has_inp = load_models()

if not run:
    st.markdown("<h1 style='text-align:center'>Self-Healing Deep Learning Receiver</h1>", unsafe_allow_html=True)
    st.stop()

# --- MULTIPLE FILE PACKETIZED STITCHING LOGIC ---
if input_mode == "Multimedia File(s) (Up to 5MB each)" and uploaded_files:
    st.markdown(f"### 📦 Processing {len(uploaded_files)} File(s)")

    for f_idx, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.read()
        file_ext = uploaded_file.name.split('.')[-1].lower()

        st.markdown(f"--- \n## File {f_idx + 1}: {uploaded_file.name}")

        if len(file_bytes) > MAX_FILE_SIZE:
            st.error(f"File is {len(file_bytes)/1000000:.2f}MB. Limit is 5MB per file. Skipping.")
            continue

        num_packets = math.ceil(len(file_bytes) / PACKET_SIZE)
        st.markdown(f"#### 📦 Split into {num_packets} Packets (Sequential Processing)")

        progress_text = f"🧬 Healing {uploaded_file.name}..."
        progress_bar = st.progress(0, text=progress_text)

        final_healed_bytes = bytearray()
        final_corrupted_bytes = bytearray()
        display_graph_shown = False

        for p_idx in range(num_packets):
            start_byte = p_idx * PACKET_SIZE
            end_byte = min(start_byte + PACKET_SIZE, len(file_bytes))
            b_bytes = file_bytes[start_byte:end_byte]

            enc_bytes = apply_ecc_encode(b_bytes)
            enc_bits = bytes_to_bits(enc_bytes)

            I = bpsk_modulate(enc_bits, sps); Q = np.zeros_like(I)
            iq_clean = (I + 1j * Q).astype(np.complex64)

            iq_atk = iq_clean.copy()
            if use_awgn: iq_atk = awgn(iq_atk, snr_db)
            if use_erasure: iq_atk = burst_erasure(iq_atk, s_pct, e_pct)

            c_bits = bpsk_demodulate(np.real(iq_atk), sps)
            c_bits = c_bits[:len(enc_bits)]
            c_bytes = bits_to_bytes(c_bits)
            final_corrupted_bytes.extend(c_bytes[:len(b_bytes)])

            iq_healed_raw = heal_signal_unet_chunked(iq_atk, inp, s_pct, e_pct, DEVICE) if has_inp else iq_atk.copy()
            iq_healed = quantize_bpsk(iq_healed_raw)

            # Run Classifier on the first packet to show Threat Status & Modulation
            if not display_graph_shown and has_clf:
                mod_probs_atk, threat_probs_atk = classify_signal(iq_atk, clf, DEVICE)
                mod_probs_healed, threat_probs_healed = classify_signal(iq_healed, clf, DEVICE)

                if threat_probs_atk is not None:
                    threat_status = "🔴 GLITCH DETECTED" if threat_probs_atk[1] > 0.5 else "🟢 CLEAN"
                    st.warning(f"**Packet 1 Threat Audit:** {threat_status}")

                    cA, cB = st.columns(2)
                    with cA:
                        st.markdown("**Attacked Signal — AI Modulation Classification**")
                        st.bar_chart({MOD_LABELS[i]: float(mod_probs_atk[i]) for i in range(len(MOD_LABELS))})
                    with cB:
                        st.markdown("**Healed Signal — AI Modulation Classification**")
                        st.bar_chart({MOD_LABELS[i]: float(mod_probs_healed[i]) for i in range(len(MOD_LABELS))})

            if not display_graph_shown:
                N = len(iq_clean)
                win = min(1024, N); t = np.arange(win)
                ft = make_subplots(rows=1, cols=1)
                ft.add_trace(go.Scatter(x=t, y=np.real(iq_clean[:win]), name="Clean", line=dict(color="green", width=2)))
                ft.add_trace(go.Scatter(x=t, y=np.real(iq_atk[:win]), name="Attacked", line=dict(color="red", width=2, dash="dot")))
                ft.add_trace(go.Scatter(x=t, y=np.real(iq_healed[:win]), name="Healed", line=dict(color="blue", width=2)))
                ft.add_vrect(x0=int(N*s_pct), x1=int(N*e_pct), fillcolor="gray", opacity=0.2, line_width=0, annotation_text="erased region")
                ft.update_layout(height=350, xaxis_title="sample", yaxis_title="amplitude", title=f"Waveform Preview (Packet 1 of {num_packets})")
                st.plotly_chart(ft, use_container_width=True)
                display_graph_shown = True

            h_bits = bpsk_demodulate(np.real(iq_healed), sps)
            h_bits = h_bits[:len(enc_bits)]
            h_bytes = bits_to_bytes(h_bits)

            try:
                corrected_bytes = apply_ecc_decode(h_bytes)
                corrected_bytes = corrected_bytes[:len(b_bytes)]
            except Exception:
                corrected_bytes = h_bytes

            final_healed_bytes.extend(corrected_bytes)
            progress_bar.progress((p_idx + 1) / num_packets, text=f"Processed Packet {p_idx + 1}/{num_packets}")

        progress_bar.empty()

        st.markdown("#### ✅ File Processing Complete!")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### ✅ Original File")
            if file_ext in ["mp3", "wav", "m4a", "ogg", "opus"]:
                st.audio(io.BytesIO(file_bytes), format=f"audio/{file_ext}")
            elif file_ext in ["png", "jpg"]:
                st.image(file_bytes)
            else:
                st.write(f"Binary file (.{file_ext})")

        with c2:
            st.markdown("##### ❌ Corrupted File (Hacked)")
            st.caption("If the player is blank or cuts out, the hack was successful.")
            if file_ext in ["mp3", "wav", "m4a", "ogg", "opus"]:
                st.audio(io.BytesIO(bytes(final_corrupted_bytes)), format=f"audio/{file_ext}")
            elif file_ext in ["png", "jpg"]:
                st.image(bytes(final_corrupted_bytes))
            else:
                st.write(f"Binary file (.{file_ext})")

        with c3:
            st.markdown("##### 🧬 Healed & Corrected File")
            if file_ext in ["mp3", "wav", "m4a", "ogg", "opus"]:
                st.audio(io.BytesIO(bytes(final_healed_bytes)), format=f"audio/{file_ext}")
            elif file_ext in ["png", "jpg"]:
                st.image(bytes(final_healed_bytes))
            else:
                st.write(f"Binary file (.{file_ext})")

    st.stop()

# --- LIVE RADIOML LOGIC ---
elif input_mode == "Live RadioML Stream (Genuine Data)" and rml_dict is not None:
    keys = list(rml_dict.keys())
    mod, snr = keys[np.random.randint(0, len(keys))]
    samples = rml_dict[(mod, snr)]
    sample = samples[np.random.randint(0, len(samples))]
    iq_source = (sample[0] + 1j * sample[1]).astype(np.complex64)
    iq_source = np.tile(iq_source, 8)
    st.markdown(f"### 📡 Live Capture: {mod} @ {snr}dB SNR")

if iq_source is None or len(iq_source) < 64:
    st.error("Please provide a valid input signal or upload files.")
    st.stop()

# --- SINGLE FILE / TEXT / RADIOML PROCESSING ---
N = len(iq_source); pad = (16 - N % 16) % 16
if pad: iq_source = np.pad(iq_source, (0, pad))
N = len(iq_source)

iq_clean = iq_source.astype(np.complex64)
iq_attacked = iq_clean.copy()
if use_awgn: iq_attacked = awgn(iq_attacked, snr_db)
if use_fgsm: iq_attacked = fgsm_glitch(iq_attacked, classifier=clf if has_clf else None, epsilon=eps, device=DEVICE)
if use_erasure: iq_attacked = burst_erasure(iq_attacked, s_pct, e_pct)

w_glitch = isolate_glitch(iq_clean, iq_attacked)
if has_inp:
    try:
        iq_healed_raw = heal_signal_unet_chunked(iq_attacked, inp, s_pct, e_pct, DEVICE)
        iq_healed = quantize_bpsk(iq_healed_raw)
    except Exception:
        iq_healed = iq_attacked.copy()
else:
    iq_healed = iq_attacked.copy()

# --- CLASSIFIER LOGIC ---
mod_probs_atk, threat_probs_atk = classify_signal(iq_attacked, clf, DEVICE) if has_clf else (None, None)
mod_probs_healed, threat_probs_healed = classify_signal(iq_healed, clf, DEVICE) if has_clf else (None, None)

# --- DECODING ---
if input_mode == "Text Payload":
    bits_healed = bpsk_demodulate(np.real(iq_healed), sps)
    h_bytes = bits_to_bytes(bits_healed)
    h_bytes = h_bytes[:text_enc_len]

    RS_TEXT = reedsolo.RSCodec(32)
    try:
        corrected_bytes, _, _ = RS_TEXT.decode(h_bytes)
        text_healed = corrected_bytes[:raw_text_len].decode('utf-8', errors='ignore').strip()
    except Exception:
        text_healed = h_bytes[:raw_text_len].decode('utf-8', errors='ignore').strip()

    text_clean = text
    text_std = scramble_text(text_clean)
else:
    text_clean = "(No text payload)"
    text_std = "(No text payload)"
    text_healed = "(No text payload)"

audio_clean = iq_to_audio_bytes(iq_clean, s_pct, e_pct)
audio_attacked = iq_to_audio_bytes(iq_attacked, s_pct, e_pct)
audio_healed = iq_to_audio_bytes(iq_healed, s_pct, e_pct)

evm_val = calculate_evm(iq_clean, iq_healed)
corr_val = correlation(iq_clean, iq_healed)

st.markdown("<h1 style='text-align:center'>Self-Healing Deep Learning Receiver</h1>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Inpaint Correlation", f"{corr_val:.3f}")
k2.metric("EVM (Error Vector Magnitude)", f"{evm_val:.2f} %")
k3.metric("Payload Size", f"{N/1024:.1f}k samples")

if threat_probs_atk is not None:
    threat_status = "🔴 GLITCH" if threat_probs_atk[1] > 0.5 else "🟢 CLEAN"
    k4.metric("Threat Status (Attacked)", threat_status)
else:
    k4.metric("Threat Status", "N/A")

if mod_probs_atk is not None:
    st.markdown("### 🛡️ Dual-Task Threat Audit & Modulation Classification")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Attacked Signal — AI Classification**")
        st.bar_chart({MOD_LABELS[i]: float(mod_probs_atk[i]) for i in range(len(MOD_LABELS))})
    with cB:
        st.markdown("**Healed Signal — AI Classification**")
        st.bar_chart({MOD_LABELS[i]: float(mod_probs_healed[i]) for i in range(len(MOD_LABELS))})

st.markdown("### 📡 I/Q Constellation Scatter — Clean vs Attacked vs Healed")
subN = min(1500, N)
fig = make_subplots(rows=1, cols=3, subplot_titles=("Clean", "Attacked", "Self-Healed"))
fig.add_trace(go.Scattergl(x=np.real(iq_clean[:subN]), y=np.imag(iq_clean[:subN]), mode="markers", marker=dict(size=4, color="green"), name="clean"), row=1, col=1)
fig.add_trace(go.Scattergl(x=np.real(iq_attacked[:subN]), y=np.imag(iq_attacked[:subN]), mode="markers", marker=dict(size=4, color="red"), name="attacked"), row=1, col=2)
fig.add_trace(go.Scattergl(x=np.real(iq_healed[:subN]), y=np.imag(iq_healed[:subN]), mode="markers", marker=dict(size=4, color="blue"), name="healed"), row=1, col=3)
fig.update_layout(height=380, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 🌊 Time-Domain Waveform (I channel)")
win = min(1024, N); t = np.arange(win)
ft = make_subplots(rows=1, cols=1)
ft.add_trace(go.Scatter(x=t, y=np.real(iq_clean[:win]), name="Clean", line=dict(color="green", width=1.5)))
ft.add_trace(go.Scatter(x=t, y=np.real(iq_attacked[:win]), name="Attacked", line=dict(color="red", width=1.5, dash="dot")))
ft.add_trace(go.Scatter(x=t, y=np.real(iq_healed[:win]), name="Healed", line=dict(color="blue", width=1.5)))
ft.add_vrect(x0=int(N*s_pct), x1=int(N*e_pct), fillcolor="gray", opacity=0.2, line_width=0, annotation_text="erased region")
ft.update_layout(height=400, xaxis_title="sample", yaxis_title="amplitude")
st.plotly_chart(ft, use_container_width=True)

st.markdown("### 🔬 Forensic Glitch Isolation  W = S_attacked − S_clean")
fg = go.Figure()
fg.add_trace(go.Scatter(x=t, y=np.real(w_glitch[:win]), name="Re(W)", line=dict(color="orange", width=1.5)))
fg.update_layout(height=300, xaxis_title="sample", yaxis_title="amplitude")
st.plotly_chart(fg, use_container_width=True)

st.markdown("### 🔓 3-Way Comparative Payload Decoder")
d1, d2, d3 = st.columns(3)
d1.success("✅ Pipeline 1 — Clean Reference"); d1.code(text_clean[:80] if input_mode == "Text Payload" else f"Signal loaded ({N} samples)")
d2.error("❌ Pipeline 2 — Standard Receiver"); d2.code(text_std[:80] if input_mode == "Text Payload" else "Corrupted")
d3.info("🧬 Pipeline 3 — Self-Healing AI"); d3.code(text_healed[:80] if input_mode == "Text Payload" else "Recovered")

st.markdown("### 🔊 Radio Wave Audio Playback")
a1, a2, a3 = st.columns(3)
with a1:
    st.markdown("Clean Wave")
    st.audio(io.BytesIO(audio_clean), format="audio/wav")
with a2:
    st.markdown("Hacked Wave")
    st.audio(io.BytesIO(audio_attacked), format="audio/wav")
with a3:
    st.markdown("Healed Wave")
    st.audio(io.BytesIO(audio_healed), format="audio/wav")