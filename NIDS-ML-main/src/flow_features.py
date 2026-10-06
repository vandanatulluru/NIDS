"""
flow_features.py - CIC-IDS2017-compatible flow feature builder.

Pure Python (no scapy), so it can be unit-tested against the real dataset.
Every convention below was VALIDATED on the raw CIC-IDS2017 CSVs:

  * Flow is bidirectional; the first packet defines the "forward" direction.
  * Lengths are TCP/UDP PAYLOAD bytes (not the whole frame).
  * Packet-length statistics (mean/std/variance/Average_Packet_Size) include the
    FIRST packet TWICE (CICFlowMeter quirk) and use the SAMPLE std/variance (n-1).
  * Fwd/Bwd length statistics do NOT have the duplicate; sample std (n-1).
  * Header length = sum of per-packet header bytes of that direction
    (TCP: data-offset*4).
  * Init_Win_bytes_* = TCP window of the first packet in that direction, -1 if none
    (UDP, or no packet seen in that direction).
  * IAT features are in MICROSECONDS, over all packets of the flow (n-1 intervals).
  * Direction-less counters: ACK_Flag_Count / URG_Flag_Count = packets carrying the flag.
"""
import math

FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _var(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return sum((x - m) ** 2 for x in v) / (n - 1)      # sample variance


def _std(v):
    return math.sqrt(_var(v))


class Flow:
    def __init__(self, fwd_ip, fwd_port, bwd_ip, bwd_port, proto, ts):
        self.fwd_ip, self.fwd_port = fwd_ip, fwd_port
        self.bwd_ip, self.bwd_port = bwd_ip, bwd_port
        self.proto = proto
        self.start_ts = ts
        self.last_ts = ts
        self.ts = []
        self.pkt_lengths = []        # first packet stored twice (CICFlowMeter quirk)
        self.fwd_lengths = []
        self.bwd_lengths = []
        self.fwd_hdr = 0
        self.bwd_hdr = 0
        self.ack = 0
        self.urg = 0
        self.init_win_fwd = None
        self.init_win_bwd = None

    # ------------------------------------------------------------------
    def is_forward(self, src_ip, src_port):
        return src_ip == self.fwd_ip and src_port == self.fwd_port

    @property
    def n_packets(self):
        return len(self.ts)

    @property
    def total_payload_bytes(self):
        return sum(self.fwd_lengths) + sum(self.bwd_lengths)

    # ------------------------------------------------------------------
    def add_packet(self, ts, is_fwd, payload_len, header_len, flags=None, window=None):
        if not self.ts:                           # first packet is counted twice
            self.pkt_lengths.append(payload_len)
        self.ts.append(ts)
        self.pkt_lengths.append(payload_len)
        self.last_ts = ts

        if is_fwd:
            self.fwd_lengths.append(payload_len)
            self.fwd_hdr += header_len
            if window is not None and self.init_win_fwd is None:
                self.init_win_fwd = int(window)
        else:
            self.bwd_lengths.append(payload_len)
            self.bwd_hdr += header_len
            if window is not None and self.init_win_bwd is None:
                self.init_win_bwd = int(window)

        if flags is not None:
            if flags & ACK:
                self.ack += 1
            if flags & URG:
                self.urg += 1

    # ------------------------------------------------------------------
    def to_feature_dict(self):
        n = len(self.ts)
        pl, fwd, bwd = self.pkt_lengths, self.fwd_lengths, self.bwd_lengths
        iats = [max(0.0, (self.ts[i] - self.ts[i - 1]) * 1e6) for i in range(1, n)]

        return {
            "Average_Packet_Size": sum(pl) / n if n else 0.0,
            "Bwd_Packet_Length_Std": _std(bwd),
            "Bwd_Packet_Length_Mean": _mean(bwd),
            "Packet_Length_Variance": _var(pl),
            "Packet_Length_Mean": _mean(pl),
            "Destination_Port": self.bwd_port,        # dst port of the initiating packet
            "Max_Packet_Length": max(pl) if pl else 0,
            "Avg_Bwd_Segment_Size": _mean(bwd),
            "Total_Length_of_Bwd_Packets": sum(bwd),
            "Fwd_Packet_Length_Max": max(fwd) if fwd else 0,
            "Bwd_Packet_Length_Max": max(bwd) if bwd else 0,
            "Packet_Length_Std": _std(pl),
            "Fwd_Packet_Length_Mean": _mean(fwd),
            "Avg_Fwd_Segment_Size": _mean(fwd),
            "Init_Win_bytes_forward": -1 if self.init_win_fwd is None else self.init_win_fwd,
            "Init_Win_bytes_backward": -1 if self.init_win_bwd is None else self.init_win_bwd,
            "Fwd_Header_Length.1": self.fwd_hdr,
            "Fwd_Packet_Length_Min": min(fwd) if fwd else 0,
            "Total_Length_of_Fwd_Packets": sum(fwd),
            "Subflow_Fwd_Bytes": sum(fwd),
            "Total_Fwd_Packets": len(fwd),
            "Subflow_Bwd_Bytes": sum(bwd),
            "Bwd_Header_Length": self.bwd_hdr,
            "Subflow_Fwd_Packets": len(fwd),
            "ACK_Flag_Count": self.ack,
            "Flow_IAT_Mean": _mean(iats),
            "Flow_IAT_Std": _std(iats),
            "Bwd_Packet_Length_Min": min(bwd) if bwd else 0,
            "URG_Flag_Count": self.urg,
            "Fwd_Header_Length": self.fwd_hdr,
        }

    def to_vector(self, feature_names):
        d = self.to_feature_dict()
        missing = [f for f in feature_names if f not in d]
        if missing:
            raise KeyError(f"Feature(s) not produced by flow_features: {missing}")
        return [float(d[f]) for f in feature_names]