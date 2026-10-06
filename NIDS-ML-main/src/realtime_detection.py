"""
realtime_detection.py  (replacement - drop into src/)

Live network intrusion detection.  Differences from the old version:
  * Bidirectional flows (replies are matched to the same flow, so all Bwd_* features work)
  * CIC-IDS2017-compatible feature definitions (see flow_features.py)
  * Continuous capture (AsyncSniffer) - no packets lost between 1-second sniff windows
  * Capture timestamps (packet.time), IAT in microseconds
  * Flows end on FIN, after 120 s, or after IDLE_TIMEOUT seconds without packets
  * Flows with fewer than 2 packets are skipped (the dataset has none, so the model never saw them)
  * Refuses to start if scaler.pkl is the broken identity scaler

The CSV log keeps the SAME columns as before, so dashboard.py keeps working.
"""
import os
import csv
import sys
import time
import threading

import joblib
import numpy as np
from scapy.all import AsyncSniffer, IP, TCP, UDP

from flow_features import Flow, FIN

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BINARY_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
BINARY_SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "models", "feature_names.txt")
MULTICLASS_MODEL_PATH = os.path.join(BASE_DIR, "models", "attack_classifier.pkl")
MULTICLASS_SCALER_PATH = os.path.join(BASE_DIR, "models", "multiclass_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "attack_label_encoder.pkl")
LOG_FILE = os.path.join(BASE_DIR, "logs", "realtime_detection.csv")

ATTACK_THRESHOLD = 0.50
FLOW_TIMEOUT_MAX = 120      # seconds - same flow timeout CICFlowMeter used for CIC-IDS2017
IDLE_TIMEOUT = 10           # seconds without packets -> flow is closed and classified
MIN_PACKETS = 2             # CIC-IDS2017 has NO 1-packet flows (min is 2 in all 8 files)
IFACE = os.environ.get("NIDS_IFACE") or None   # e.g. set NIDS_IFACE=Wi-Fi

# ============================================================
# STATE
# ============================================================
flows = {}
flows_lock = threading.Lock()
log_lock = threading.Lock()

binary_model = binary_scaler = None
multiclass_model = multiclass_scaler = label_encoder = None
FEATURE_NAMES = []


def load_artifacts():
    global binary_model, binary_scaler, multiclass_model, multiclass_scaler
    global label_encoder, FEATURE_NAMES

    print("=" * 70)
    print("REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 70)
    print("\nLoading models...")
    binary_model = joblib.load(BINARY_MODEL_PATH)
    binary_scaler = joblib.load(BINARY_SCALER_PATH)
    multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
    multiclass_scaler = joblib.load(MULTICLASS_SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        FEATURE_NAMES = [line.strip() for line in f if line.strip()]
    if len(FEATURE_NAMES) != 30:
        raise ValueError(f"Expected 30 features, found {len(FEATURE_NAMES)}")

    # Guard against the identity-scaler bug (run fix_scaler.py to repair it)
    if np.max(np.abs(binary_scaler.scale_ - 1.0)) < 1e-6:
        raise RuntimeError(
            "scaler.pkl is an IDENTITY scaler (scale=1, mean~0). "
            "Run fix_scaler.py to rebuild it before using the detector."
        )

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "Timestamp", "Source_IP", "Destination_IP", "Source_Port",
                "Destination_Port", "Protocol", "Packets", "Bytes",
                "Prediction", "Attack_Type", "Attack_Probability",
            ])
    print(f"Binary model features : {len(FEATURE_NAMES)}")
    print(f"Attack threshold      : {ATTACK_THRESHOLD}")


# ============================================================
# PACKET PARSING  (scapy packet -> plain dict)
# ============================================================
def packet_info(packet):
    """Extract what the flow builder needs. Returns None for non-IPv4 packets."""
    if not packet.haslayer(IP):
        return None
    ip = packet[IP]
    ihl = (ip.ihl or 5) * 4
    ip_len = ip.len if ip.len else len(ip)       # len==0 happens with segmentation offload

    sport = dport = 0
    flags = None
    window = None

    if packet.haslayer(TCP):
        tcp = packet[TCP]
        hdr = (tcp.dataofs or 5) * 4
        payload = max(0, ip_len - ihl - hdr)
        sport, dport = int(tcp.sport), int(tcp.dport)
        flags, window = int(tcp.flags), int(tcp.window)
    elif packet.haslayer(UDP):
        udp = packet[UDP]
        payload = max(0, (udp.len if udp.len else ip_len - ihl) - 8)
        # CIC-IDS2017 UDP flows show header lengths of 20/32/40 (never 8), so the
        # IP header length is used as the closest available approximation.
        hdr = ihl
        sport, dport = int(udp.sport), int(udp.dport)
    else:
        payload = max(0, ip_len - ihl)
        hdr = ihl

    return {
        "src_ip": ip.src, "dst_ip": ip.dst,
        "sport": sport, "dport": dport,
        "proto": int(ip.proto),
        "ts": float(packet.time),
        "payload": payload, "hdr": hdr,
        "flags": flags, "window": window,
    }


def flow_key(info):
    a = (info["src_ip"], info["sport"])
    b = (info["dst_ip"], info["dport"])
    return (info["proto"],) + tuple(sorted((a, b)))     # same key for both directions


# ============================================================
# FLOW TABLE
# ============================================================
def ingest(info):
    """Add one parsed packet to the flow table. Returns flows that just finished."""
    finished = []
    key = flow_key(info)
    with flows_lock:
        flow = flows.get(key)

        # hard 120 s flow timeout: close the old flow, start a new one
        if flow is not None and info["ts"] - flow.start_ts > FLOW_TIMEOUT_MAX:
            finished.append(flows.pop(key))
            flow = None

        if flow is None:
            flow = Flow(info["src_ip"], info["sport"], info["dst_ip"], info["dport"],
                        info["proto"], info["ts"])
            flows[key] = flow

        is_fwd = flow.is_forward(info["src_ip"], info["sport"])
        flow.add_packet(info["ts"], is_fwd, info["payload"], info["hdr"],
                        info["flags"], info["window"])

        # CICFlowMeter terminates a flow at the first FIN
        if info["flags"] is not None and info["flags"] & FIN:
            finished.append(flows.pop(key))
    return finished


def expire_idle_flows(now=None):
    now = time.time() if now is None else now
    finished = []
    with flows_lock:
        for key in [k for k, f in flows.items() if now - f.last_ts >= IDLE_TIMEOUT]:
            finished.append(flows.pop(key))
    return finished


# ============================================================
# CLASSIFICATION
# ============================================================
def classify_attack(X):
    try:
        pred = multiclass_model.predict(multiclass_scaler.transform(X))[0]
        return str(label_encoder.inverse_transform([pred])[0])
    except Exception as e:
        print(f"[WARNING] Multiclass classification failed: {e}")
        return "Unknown"


def process_flow(flow):
    if flow is None or flow.n_packets < MIN_PACKETS:
        return
    try:
        X = np.array([flow.to_vector(FEATURE_NAMES)], dtype=np.float64)
        attack_probability = float(binary_model.predict_proba(binary_scaler.transform(X))[0][1])

        if attack_probability >= ATTACK_THRESHOLD:
            prediction, attack_type = "ATTACK", classify_attack(X)
        else:
            prediction, attack_type = "BENIGN", "BENIGN"

        packets, total_bytes = flow.n_packets, flow.total_payload_bytes
        print(f"\n[{prediction}] {flow.fwd_ip}:{flow.fwd_port} -> {flow.bwd_ip}:{flow.bwd_port} "
              f"| Packets: {packets} | Bytes: {total_bytes} "
              f"| Attack Probability: {attack_probability * 100:.2f}%")
        if prediction == "ATTACK":
            print(f"   Attack Type: {attack_type}")

        with log_lock, open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                flow.fwd_ip, flow.bwd_ip, flow.fwd_port, flow.bwd_port, flow.proto,
                packets, total_bytes, prediction, attack_type,
                round(attack_probability, 6),
            ])
    except Exception as e:
        print(f"[ERROR] Flow processing failed: {e}")


def packet_handler(packet):
    info = packet_info(packet)
    if info is None:
        return
    for flow in ingest(info):
        process_flow(flow)


# ============================================================
# MAIN
# ============================================================
def main():
    load_artifacts()
    print("\n" + "=" * 70)
    print("REAL-TIME DETECTION STARTED")
    print(f"Interface: {IFACE or 'default'} | idle timeout: {IDLE_TIMEOUT}s | "
          f"max flow: {FLOW_TIMEOUT_MAX}s")
    print("Press Ctrl+C to stop.")
    print("=" * 70)

    sniffer = AsyncSniffer(iface=IFACE, prn=packet_handler, store=False)
    sniffer.start()
    try:
        while True:
            time.sleep(1)
            for flow in expire_idle_flows():
                process_flow(flow)
    except KeyboardInterrupt:
        print("\n\nStopping real-time detection...")
    finally:
        try:
            sniffer.stop()
        except Exception:
            pass
        with flows_lock:
            remaining = list(flows.values())
            flows.clear()
        for flow in remaining:
            process_flow(flow)
        print("Detection stopped.")


if __name__ == "__main__":
    main()