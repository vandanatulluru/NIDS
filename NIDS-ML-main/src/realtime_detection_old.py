import os
import csv
import time
import joblib
import numpy as np
from collections import defaultdict

from scapy.all import sniff, IP, TCP, UDP


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BINARY_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)

BINARY_SCALER_PATH = os.path.join(
    BASE_DIR, "data", "processed", "scaler.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    BASE_DIR, "models", "feature_names.txt"
)

MULTICLASS_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "attack_classifier.pkl"
)

MULTICLASS_SCALER_PATH = os.path.join(
    BASE_DIR, "models", "multiclass_scaler.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR, "models", "attack_label_encoder.pkl"
)

LOG_FILE = os.path.join(
    BASE_DIR, "logs", "realtime_detection.csv"
)


# ============================================================
# OPTIMIZED THRESHOLD
# ============================================================

ATTACK_THRESHOLD = 0.50

FLOW_TIMEOUT = 10
MIN_PACKETS = 2


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("REAL-TIME NETWORK INTRUSION DETECTION SYSTEM")
print("=" * 70)

print("\nLoading models...")

binary_model = joblib.load(
    BINARY_MODEL_PATH
)

binary_scaler = joblib.load(
    BINARY_SCALER_PATH
)

multiclass_model = joblib.load(
    MULTICLASS_MODEL_PATH
)

multiclass_scaler = joblib.load(
    MULTICLASS_SCALER_PATH
)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

with open(
    FEATURE_NAMES_PATH,
    "r",
    encoding="utf-8"
) as f:

    FEATURE_NAMES = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(f"Binary model features : {len(FEATURE_NAMES)}")
print(f"Attack threshold      : {ATTACK_THRESHOLD}")

if len(FEATURE_NAMES) != 30:

    raise ValueError(
        f"Expected 30 features, found {len(FEATURE_NAMES)}"
    )


# ============================================================
# FLOW STORAGE
# ============================================================

flows = {}


# ============================================================
# LOG FILE
# ============================================================

os.makedirs(
    os.path.dirname(LOG_FILE),
    exist_ok=True
)

if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Timestamp",
            "Source_IP",
            "Destination_IP",
            "Source_Port",
            "Destination_Port",
            "Protocol",
            "Packets",
            "Bytes",
            "Prediction",
            "Attack_Type",
            "Attack_Probability"
        ])


# ============================================================
# FLOW KEY
# ============================================================

def get_flow_key(packet):

    if not packet.haslayer(IP):
        return None

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    protocol = packet[IP].proto

    src_port = 0
    dst_port = 0

    if packet.haslayer(TCP):

        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport

    elif packet.haslayer(UDP):

        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    return (
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        protocol
    )


# ============================================================
# CREATE FLOW
# ============================================================

def create_flow(packet):

    key = get_flow_key(packet)

    if key is None:
        return None

    src_ip, dst_ip, src_port, dst_port, protocol = key

    packet_length = len(packet)

    now = time.time()

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,

        "src_port": src_port,
        "dst_port": dst_port,

        "protocol": protocol,

        "start_time": now,
        "last_time": now,

        "packet_times": [now],
        "packet_lengths": [packet_length],

        "forward_lengths": [packet_length],
        "backward_lengths": [],

        "forward_packets": 1,
        "backward_packets": 0,

        "total_bytes": packet_length,

        "fwd_bytes": packet_length,
        "bwd_bytes": 0,

        "fwd_header_length": 0,
        "bwd_header_length": 0,

        "ack_count": 0,
        "urg_count": 0,

        "init_win_forward": 0,
        "init_win_backward": 0,

        "fwd_iats": [],
        "bwd_iats": []
    }


# ============================================================
# UPDATE FLOW
# ============================================================

def update_flow(flow, packet):

    now = time.time()

    packet_length = len(packet)

    # --------------------------------------------------------
    # General packet information
    # --------------------------------------------------------

    if flow["packet_times"]:

        iat = now - flow["packet_times"][-1]

        flow["fwd_iats"].append(iat)

    flow["packet_times"].append(now)

    flow["packet_lengths"].append(packet_length)

    flow["forward_lengths"].append(packet_length)

    flow["forward_packets"] += 1

    flow["fwd_bytes"] += packet_length

    flow["total_bytes"] += packet_length

    flow["last_time"] = now

    # --------------------------------------------------------
    # TCP flags
    # --------------------------------------------------------

    if packet.haslayer(TCP):

        tcp = packet[TCP]

        if tcp.flags & 0x10:
            flow["ack_count"] += 1

        if tcp.flags & 0x20:
            flow["urg_count"] += 1

        if flow["init_win_forward"] == 0:

            flow["init_win_forward"] = int(
                tcp.window
            )


# ============================================================
# STATISTICAL HELPERS
# ============================================================

def safe_mean(values):

    if not values:
        return 0.0

    return float(np.mean(values))


def safe_std(values):

    if len(values) < 2:
        return 0.0

    return float(np.std(values))


def safe_min(values):

    if not values:
        return 0.0

    return float(np.min(values))


def safe_max(values):

    if not values:
        return 0.0

    return float(np.max(values))


def safe_variance(values):

    if len(values) < 2:
        return 0.0

    return float(np.var(values))


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(flow):

    lengths = flow["packet_lengths"]

    fwd = flow["forward_lengths"]

    bwd = flow["backward_lengths"]

    total_packets = (
        flow["forward_packets"]
        + flow["backward_packets"]
    )

    total_bytes = flow["total_bytes"]

    duration = max(
        flow["last_time"] - flow["start_time"],
        1e-6
    )

    packet_mean = safe_mean(lengths)

    packet_std = safe_std(lengths)

    packet_variance = safe_variance(lengths)

    fwd_mean = safe_mean(fwd)

    bwd_mean = safe_mean(bwd)

    fwd_std = safe_std(fwd)

    bwd_std = safe_std(bwd)

    fwd_max = safe_max(fwd)

    bwd_max = safe_max(bwd)

    fwd_min = safe_min(fwd)

    bwd_min = safe_min(bwd)

    flow_iat_mean = safe_mean(
        flow["fwd_iats"]
    )

    flow_iat_std = safe_std(
        flow["fwd_iats"]
    )

    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    feature_dict = {

        "Average_Packet_Size":
            packet_mean,

        "Bwd_Packet_Length_Std":
            bwd_std,

        "Bwd_Packet_Length_Mean":
            bwd_mean,

        "Packet_Length_Variance":
            packet_variance,

        "Packet_Length_Mean":
            packet_mean,

        "Destination_Port":
            flow["dst_port"],

        "Max_Packet_Length":
            safe_max(lengths),

        "Avg_Bwd_Segment_Size":
            bwd_mean,

        "Total_Length_of_Bwd_Packets":
            flow["bwd_bytes"],

        "Fwd_Packet_Length_Max":
            fwd_max,

        "Bwd_Packet_Length_Max":
            bwd_max,

        "Packet_Length_Std":
            packet_std,

        "Fwd_Packet_Length_Mean":
            fwd_mean,

        "Avg_Fwd_Segment_Size":
            fwd_mean,

        "Init_Win_bytes_forward":
            flow["init_win_forward"],

        "Init_Win_bytes_backward":
            flow["init_win_backward"],

        "Fwd_Header_Length.1":
            flow["fwd_header_length"],

        "Fwd_Packet_Length_Min":
            fwd_min,

        "Total_Length_of_Fwd_Packets":
            flow["fwd_bytes"],

        "Subflow_Fwd_Bytes":
            flow["fwd_bytes"],

        "Total_Fwd_Packets":
            flow["forward_packets"],

        "Subflow_Bwd_Bytes":
            flow["bwd_bytes"],

        "Bwd_Header_Length":
            flow["bwd_header_length"],

        "Subflow_Fwd_Packets":
            flow["forward_packets"],

        "ACK_Flag_Count":
            flow["ack_count"],

        "Flow_IAT_Mean":
            flow_iat_mean,

        "Flow_IAT_Std":
            flow_iat_std,

        "Bwd_Packet_Length_Min":
            bwd_min,

        "URG_Flag_Count":
            flow["urg_count"],

        "Fwd_Header_Length":
            flow["fwd_header_length"]
    }

    # --------------------------------------------------------
    # Ensure exact feature order
    # --------------------------------------------------------

    values = []

    for feature in FEATURE_NAMES:

        values.append(
            feature_dict.get(
                feature,
                0.0
            )
        )

    return np.array(
        values,
        dtype=np.float32
    ).reshape(1, -1)


# ============================================================
# MULTICLASS ATTACK CLASSIFICATION
# ============================================================

def classify_attack(X):

    try:

        X_scaled = multiclass_scaler.transform(X)

        prediction = multiclass_model.predict(
            X_scaled
        )[0]

        attack_type = label_encoder.inverse_transform(
            [prediction]
        )[0]

        return str(attack_type)

    except Exception as e:

        print(
            f"[WARNING] Multiclass classification failed: {e}"
        )

        return "Unknown"


# ============================================================
# PROCESS COMPLETED FLOW
# ============================================================

def process_flow(flow):

    if flow is None:
        return

    packets = (
        flow["forward_packets"]
        + flow["backward_packets"]
    )

    if packets < MIN_PACKETS:

        return

    try:

        # ----------------------------------------------------
        # Build features
        # ----------------------------------------------------

        X = build_features(flow)

        # ----------------------------------------------------
        # Binary scaling
        # ----------------------------------------------------

        X_scaled = binary_scaler.transform(
            X
        )

        # ----------------------------------------------------
        # Attack probability
        # ----------------------------------------------------

        probabilities = binary_model.predict_proba(
            X_scaled
        )

        attack_probability = float(
            probabilities[0][1]
        )

        # ----------------------------------------------------
        # OPTIMIZED THRESHOLD
        # ----------------------------------------------------

        if attack_probability >= ATTACK_THRESHOLD:

            prediction = "ATTACK"

            attack_type = classify_attack(
                X
            )

        else:

            prediction = "BENIGN"

            attack_type = "BENIGN"

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        probability_percent = (
            attack_probability * 100
        )

        print(
            f"\n[{prediction}] "
            f"{flow['src_ip']}:{flow['src_port']} -> "
            f"{flow['dst_ip']}:{flow['dst_port']} "
            f"| Packets: {packets} "
            f"| Bytes: {flow['total_bytes']} "
            f"| Attack Probability: "
            f"{probability_percent:.2f}%"
        )

        if prediction == "ATTACK":

            print(
                f"   Attack Type: {attack_type}"
            )

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        with open(
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([

                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                flow["src_ip"],

                flow["dst_ip"],

                flow["src_port"],

                flow["dst_port"],

                flow["protocol"],

                packets,

                flow["total_bytes"],

                prediction,

                attack_type,

                round(
                    attack_probability,
                    6
                )
            ])

    except Exception as e:

        print(
            f"[ERROR] Flow processing failed: {e}"
        )


# ============================================================
# PACKET HANDLER
# ============================================================

def packet_handler(packet):

    if not packet.haslayer(IP):

        return

    key = get_flow_key(packet)

    if key is None:

        return

    # --------------------------------------------------------
    # Create new flow
    # --------------------------------------------------------

    if key not in flows:

        flows[key] = create_flow(
            packet
        )

        return

    # --------------------------------------------------------
    # Update existing flow
    # --------------------------------------------------------

    update_flow(
        flows[key],
        packet
    )


# ============================================================
# EXPIRE OLD FLOWS
# ============================================================

def expire_flows():

    current_time = time.time()

    expired = []

    for key, flow in flows.items():

        if (
            current_time
            - flow["last_time"]
            >= FLOW_TIMEOUT
        ):

            expired.append(key)

    for key in expired:

        flow = flows.pop(key)

        process_flow(flow)


# ============================================================
# MAIN REAL-TIME LOOP
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "REAL-TIME DETECTION STARTED"
    )

    print(
        f"Attack threshold: "
        f"{ATTACK_THRESHOLD}"
    )

    print(
        "Flow timeout: "
        f"{FLOW_TIMEOUT} seconds"
    )

    print(
        "\nPress Ctrl+C to stop."
    )

    print("=" * 70)

    try:

        while True:

            # Capture packets for 1 second
            sniff(
                prn=packet_handler,
                timeout=1,
                store=False
            )

            # Process expired flows
            expire_flows()

    except KeyboardInterrupt:

        print(
            "\n\nStopping real-time detection..."
        )

        # Process remaining flows
        for key in list(flows.keys()):

            flow = flows.pop(key)

            process_flow(flow)

        print(
            "Detection stopped."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()