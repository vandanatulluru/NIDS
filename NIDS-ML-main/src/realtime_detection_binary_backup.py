"""
Real-Time Network Intrusion Detection
Flow-based live detection using Scapy + trained Random Forest.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime
import time
import csv

import joblib
import numpy as np
import pandas as pd

from scapy.all import sniff, IP, TCP, UDP


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.txt"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "realtime_detection.csv"


# ============================================================
# CONFIGURATION
# ============================================================

FLOW_TIMEOUT = 10
PREDICT_AFTER_PACKETS = 10


# ============================================================
# EXACT MODEL FEATURES
# ============================================================

FEATURES = [
    "Average_Packet_Size",
    "Bwd_Packet_Length_Std",
    "Bwd_Packet_Length_Mean",
    "Packet_Length_Variance",
    "Packet_Length_Mean",
    "Destination_Port",
    "Max_Packet_Length",
    "Avg_Bwd_Segment_Size",
    "Total_Length_of_Bwd_Packets",
    "Fwd_Packet_Length_Max",
    "Bwd_Packet_Length_Max",
    "Packet_Length_Std",
    "Fwd_Packet_Length_Mean",
    "Avg_Fwd_Segment_Size",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Fwd_Header_Length.1",
    "Fwd_Packet_Length_Min",
    "Total_Length_of_Fwd_Packets",
    "Subflow_Fwd_Bytes",
    "Total_Fwd_Packets",
    "Subflow_Bwd_Bytes",
    "Bwd_Header_Length",
    "Subflow_Fwd_Packets",
    "ACK_Flag_Count",
    "Flow_IAT_Mean",
    "Flow_IAT_Std",
    "Bwd_Packet_Length_Min",
    "URG_Flag_Count",
    "Fwd_Header_Length",
]


# ============================================================
# GLOBALS
# ============================================================

model = None
scaler = None
flows = {}


# ============================================================
# LOAD MODEL
# ============================================================

def load_components():

    global model
    global scaler

    print("Loading trained model...")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(
        f"Model loaded: {type(model).__name__}"
    )

    print(
        f"Features expected: {len(FEATURES)}"
    )


# ============================================================
# FLOW KEY
# ============================================================

def get_flow_key(packet):

    if not packet.haslayer(IP):
        return None

    ip = packet[IP]

    src_port = 0
    dst_port = 0

    if packet.haslayer(TCP):

        src_port = int(packet[TCP].sport)
        dst_port = int(packet[TCP].dport)

    elif packet.haslayer(UDP):

        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)

    protocol = int(ip.proto)

    endpoint_a = (
        ip.src,
        src_port
    )

    endpoint_b = (
        ip.dst,
        dst_port
    )

    # Bidirectional flow
    if endpoint_a <= endpoint_b:

        return (
            ip.src,
            src_port,
            ip.dst,
            dst_port,
            protocol,
        )

    return (
        ip.dst,
        dst_port,
        ip.src,
        src_port,
        protocol,
    )


# ============================================================
# CREATE FLOW
# ============================================================

def create_flow(packet):

    ip = packet[IP]

    src_port = 0
    dst_port = 0

    if packet.haslayer(TCP):

        src_port = int(packet[TCP].sport)
        dst_port = int(packet[TCP].dport)

    elif packet.haslayer(UDP):

        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)

    now = time.time()

    return {
        "src_ip": ip.src,
        "dst_ip": ip.dst,

        "src_port": src_port,
        "dst_port": dst_port,

        "protocol": int(ip.proto),

        "start_time": now,
        "last_time": now,

        "packet_times": [],
        "packet_sizes": [],

        "forward_sizes": [],
        "backward_sizes": [],

        "forward_packets": 0,
        "backward_packets": 0,

        "forward_bytes": 0,
        "backward_bytes": 0,

        "ack_count": 0,
        "urg_count": 0,

        "forward_header_length": 0,
        "backward_header_length": 0,

        "init_win_forward": 0,
        "init_win_backward": 0,

        "original_src": ip.src,
    }


# ============================================================
# UPDATE FLOW
# ============================================================

def update_flow(flow, packet):

    now = time.time()

    ip = packet[IP]

    packet_size = len(packet)

    flow["last_time"] = now

    flow["packet_times"].append(now)
    flow["packet_sizes"].append(packet_size)

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if ip.src == flow["original_src"]:

        flow["forward_packets"] += 1
        flow["forward_bytes"] += packet_size
        flow["forward_sizes"].append(packet_size)

        # IP header + transport header
        flow["forward_header_length"] += (
            getattr(ip, "ihl", 5) * 4
        )

        if packet.haslayer(TCP):

            flow["forward_header_length"] += (
                len(packet[TCP])
                - len(bytes(packet[TCP].payload))
            )

            if flow["init_win_forward"] == 0:

                flow["init_win_forward"] = int(
                    packet[TCP].window
                )

    else:

        flow["backward_packets"] += 1
        flow["backward_bytes"] += packet_size
        flow["backward_sizes"].append(packet_size)

        flow["backward_header_length"] += (
            getattr(ip, "ihl", 5) * 4
        )

        if packet.haslayer(TCP):

            flow["backward_header_length"] += (
                len(packet[TCP])
                - len(bytes(packet[TCP].payload))
            )

            if flow["init_win_backward"] == 0:

                flow["init_win_backward"] = int(
                    packet[TCP].window
                )

    # --------------------------------------------------------
    # TCP flags
    # --------------------------------------------------------

    if packet.haslayer(TCP):

        flags = int(packet[TCP].flags)

        if flags & 0x10:
            flow["ack_count"] += 1

        if flags & 0x20:
            flow["urg_count"] += 1


# ============================================================
# SAFE STATISTICS
# ============================================================

def safe_mean(values):

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def safe_std(values):

    if len(values) < 2:
        return 0.0

    return float(
        np.std(values)
    )


def safe_min(values):

    if not values:
        return 0.0

    return float(
        np.min(values)
    )


def safe_max(values):

    if not values:
        return 0.0

    return float(
        np.max(values)
    )


def safe_variance(values):

    if len(values) < 2:
        return 0.0

    return float(
        np.var(values)
    )


# ============================================================
# FLOW IAT
# ============================================================

def calculate_iat(flow):

    times = flow["packet_times"]

    if len(times) < 2:

        return 0.0, 0.0

    intervals = np.diff(times)

    return (
        float(np.mean(intervals)),
        float(np.std(intervals)),
    )


# ============================================================
# EXTRACT EXACT FEATURES
# ============================================================

def extract_features(flow):

    all_sizes = flow["packet_sizes"]

    fwd_sizes = flow["forward_sizes"]

    bwd_sizes = flow["backward_sizes"]

    iat_mean, iat_std = calculate_iat(
        flow
    )

    features = {

        "Average_Packet_Size":
            safe_mean(all_sizes),

        "Bwd_Packet_Length_Std":
            safe_std(bwd_sizes),

        "Bwd_Packet_Length_Mean":
            safe_mean(bwd_sizes),

        "Packet_Length_Variance":
            safe_variance(all_sizes),

        "Packet_Length_Mean":
            safe_mean(all_sizes),

        "Destination_Port":
            flow["dst_port"],

        "Max_Packet_Length":
            safe_max(all_sizes),

        "Avg_Bwd_Segment_Size":
            safe_mean(bwd_sizes),

        "Total_Length_of_Bwd_Packets":
            flow["backward_bytes"],

        "Fwd_Packet_Length_Max":
            safe_max(fwd_sizes),

        "Bwd_Packet_Length_Max":
            safe_max(bwd_sizes),

        "Packet_Length_Std":
            safe_std(all_sizes),

        "Fwd_Packet_Length_Mean":
            safe_mean(fwd_sizes),

        "Avg_Fwd_Segment_Size":
            safe_mean(fwd_sizes),

        "Init_Win_bytes_forward":
            flow["init_win_forward"],

        "Init_Win_bytes_backward":
            flow["init_win_backward"],

        "Fwd_Header_Length.1":
            flow["forward_header_length"],

        "Fwd_Packet_Length_Min":
            safe_min(fwd_sizes),

        "Total_Length_of_Fwd_Packets":
            flow["forward_bytes"],

        "Subflow_Fwd_Bytes":
            flow["forward_bytes"],

        "Total_Fwd_Packets":
            flow["forward_packets"],

        "Subflow_Bwd_Bytes":
            flow["backward_bytes"],

        "Bwd_Header_Length":
            flow["backward_header_length"],

        "Subflow_Fwd_Packets":
            flow["forward_packets"],

        "ACK_Flag_Count":
            flow["ack_count"],

        "Flow_IAT_Mean":
            iat_mean,

        "Flow_IAT_Std":
            iat_std,

        "Bwd_Packet_Length_Min":
            safe_min(bwd_sizes),

        "URG_Flag_Count":
            flow["urg_count"],

        "Fwd_Header_Length":
            flow["forward_header_length"],
    }

    return features


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(features):

    # Create exactly the same feature order
    # as the trained model.

    df = pd.DataFrame(
        [[
            features.get(
                feature,
                0.0
            )
            for feature in FEATURES
        ]],
        columns=FEATURES,
    )

    df = df.replace(
        [np.inf, -np.inf],
        0
    )

    df = df.fillna(0)

    # --------------------------------------------------------
    # Apply scaler only if its feature layout matches.
    # --------------------------------------------------------

    try:

        scaler_feature_names = list(
            scaler.feature_names_in_
        )

        complete = pd.DataFrame(
            0.0,
            index=[0],
            columns=scaler_feature_names,
        )

        for feature in FEATURES:

            if feature in complete.columns:

                complete.loc[
                    0,
                    feature
                ] = df.loc[
                    0,
                    feature
                ]

        scaled = scaler.transform(
            complete
        )

        scaled_df = pd.DataFrame(
            scaled,
            columns=scaler_feature_names,
        )

        df = scaled_df[
            FEATURES
        ]

    except Exception as error:

        print(
            "Scaler compatibility warning:",
            error
        )

        # If scaler cannot be matched, use
        # the numerical feature vector directly.

    return df


# ============================================================
# PREDICT
# ============================================================

def predict_flow(flow):

    features = extract_features(
        flow
    )

    X = prepare_model_input(
        features
    )

    prediction = int(
        model.predict(X)[0]
    )

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = float(
            model.predict_proba(X)[0][1]
        )

    return prediction, probability


# ============================================================
# LOG
# ============================================================

def log_detection(
    flow,
    prediction,
    probability,
):

    result = (
        "ATTACK"
        if prediction == 1
        else "BENIGN"
    )

    row = {
        "timestamp":
            datetime.now().isoformat(),

        "src_ip":
            flow["src_ip"],

        "src_port":
            flow["src_port"],

        "dst_ip":
            flow["dst_ip"],

        "dst_port":
            flow["dst_port"],

        "protocol":
            flow["protocol"],

        "packets":
            flow["forward_packets"]
            + flow["backward_packets"],

        "bytes":
            flow["forward_bytes"]
            + flow["backward_bytes"],

        "prediction":
            result,

        "attack_probability":
            probability,
    }

    write_header = (
        not LOG_FILE.exists()
    )

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# PROCESS FLOW
# ============================================================

def process_flow(
    flow_key,
    force=False,
):

    flow = flows.get(
        flow_key
    )

    if flow is None:
        return

    total_packets = (
        flow["forward_packets"]
        + flow["backward_packets"]
    )

    if (
        total_packets < 2
        and not force
    ):
        return

    try:

        prediction, probability = (
            predict_flow(flow)
        )

        result = (
            "ATTACK"
            if prediction == 1
            else "BENIGN"
        )

        if probability is not None:

            probability_text = (
                f"{probability * 100:.2f}%"
            )

        else:

            probability_text = "N/A"

        print("\n" + "-" * 60)

        print(
            f"Flow: "
            f"{flow['src_ip']}:{flow['src_port']} "
            f"-> "
            f"{flow['dst_ip']}:{flow['dst_port']}"
        )

        print(
            f"Packets: {total_packets}"
        )

        print(
            f"Bytes: "
            f"{flow['forward_bytes'] + flow['backward_bytes']}"
        )

        print(
            f"Prediction: {result}"
        )

        print(
            f"Attack Probability: "
            f"{probability_text}"
        )

        if result == "ATTACK":

            print(
                "🚨 ALERT: POSSIBLE NETWORK ATTACK DETECTED"
            )

        log_detection(
            flow,
            prediction,
            probability,
        )

    except Exception as error:

        print(
            f"\nPrediction error: {error}"
        )

    finally:

        flows.pop(
            flow_key,
            None
        )


# ============================================================
# PACKET HANDLER
# ============================================================

def packet_handler(packet):

    if not packet.haslayer(IP):
        return

    flow_key = get_flow_key(
        packet
    )

    if flow_key is None:
        return

    if flow_key not in flows:

        flows[flow_key] = (
            create_flow(packet)
        )

    update_flow(
        flows[flow_key],
        packet
    )

    total_packets = (
        flows[flow_key]["forward_packets"]
        + flows[flow_key]["backward_packets"]
    )

    if total_packets >= PREDICT_AFTER_PACKETS:

        process_flow(
            flow_key
        )


# ============================================================
# CLEANUP
# ============================================================

def cleanup_expired_flows():

    now = time.time()

    expired = []

    for key, flow in list(
        flows.items()
    ):

        if (
            now - flow["last_time"]
            >= FLOW_TIMEOUT
        ):

            expired.append(key)

    for key in expired:

        process_flow(
            key,
            force=True
        )


# ============================================================
# CAPTURE
# ============================================================

def start_capture():

    print("\n" + "=" * 60)
    print(
        "REAL-TIME NETWORK INTRUSION DETECTION"
    )
    print("=" * 60)

    print(
        "\nStarting packet capture..."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print(
        f"Flow timeout: "
        f"{FLOW_TIMEOUT} seconds"
    )

    print(
        f"Prediction after: "
        f"{PREDICT_AFTER_PACKETS} packets"
    )

    print(
        f"\nDetection log:"
        f"\n{LOG_FILE}"
    )

    try:

        sniff(
            prn=packet_handler,
            store=False,
            filter="ip",
            timeout=None,
        )

    except KeyboardInterrupt:

        print(
            "\n\nStopping packet capture..."
        )

    finally:

        # Don't attempt model predictions while
        # the user is interrupting the program.
        # Just clear active flows safely.

        flows.clear()

        print(
            "Packet capture stopped cleanly."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Missing model:\n{MODEL_PATH}"
        )

    if not SCALER_PATH.exists():

        raise FileNotFoundError(
            f"Missing scaler:\n{SCALER_PATH}"
        )

    load_components()

    # Verify that feature file matches our list.
    if FEATURE_PATH.exists():

        with open(
            FEATURE_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            saved_features = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if saved_features != FEATURES:

            raise ValueError(
                "The feature_names.txt file does not "
                "match the expected 30 features."
            )

    start_capture()


if __name__ == "__main__":
    main()