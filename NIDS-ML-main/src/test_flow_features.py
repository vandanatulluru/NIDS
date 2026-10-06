"""
test_flow_features.py - verifies the real-time feature extraction.

  Part A  hand-checked rows copied from the real CIC-IDS2017 data
  Part B  replay of thousands of real dataset rows through Flow and comparison with
          the dataset's own values (validates the ARITHMETIC of every feature)
  Part C  scapy packets: payload/header parsing + bidirectional flow matching

Run from src/:   python test_flow_features.py [path\to\a\raw_CIC_csv ...]
(Part B uses data/raw/Tuesday-*.csv by default.)
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_features import Flow, ACK, URG, SYN, FIN

FAILS = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)

def mk(fwd_lens, bwd_lens, t0=0.0, dt=0.001):
    f = Flow("a", 1000, "b", 80, 6, t0)
    f.add_packet(t0, True, fwd_lens[0], 20)
    for i, L in enumerate(fwd_lens[1:] + [None] * 0):
        f.add_packet(t0 + dt * (i + 1), True, L, 20)
    for j, L in enumerate(bwd_lens):
        f.add_packet(t0 + dt * (len(fwd_lens) + j), False, L, 20)
    return f

# ---------------------------------------------------------------- Part A
print("\nPART A - hand-checked rows from the real dataset")
cases = [  # F, B, Packet_Length_Mean, Std, Variance, Average_Packet_Size
    (44, 240, 109.333333, 113.160653, 12805.33333, 164.0),
    (48, 80, 58.666667, 18.475209, 341.333333, 88.0),
    (55, 213, 107.666667, 91.221343, 8321.333333, 161.5),
]
for F, B, mean, std, var, avg in cases:
    d = mk([F], [B]).to_feature_dict()
    ok = (np.isclose(d["Packet_Length_Mean"], mean, rtol=1e-5) and
          np.isclose(d["Packet_Length_Std"], std, rtol=1e-5) and
          np.isclose(d["Packet_Length_Variance"], var, rtol=1e-5) and
          np.isclose(d["Average_Packet_Size"], avg, rtol=1e-5))
    check(f"fwd={F} bwd={B}", ok, f"(mean {d['Packet_Length_Mean']:.4f}, std {d['Packet_Length_Std']:.4f}, avg {d['Average_Packet_Size']})")
d = mk([50], []).to_feature_dict()
check("1-packet flow: bwd features zero, IAT zero, init_win_bwd=-1",
      d["Bwd_Packet_Length_Max"] == 0 and d["Flow_IAT_Mean"] == 0 and d["Init_Win_bytes_backward"] == -1)

# ---------------------------------------------------------------- Part B
print("\nPART B - replay of real dataset rows")
_raw = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
paths = sys.argv[1:] or glob.glob(os.path.join(_raw, "Tuesday*.csv"))
if not paths:
    print("  (skipped: no CSV given and no data/raw/Tuesday*.csv)")
else:
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)
    nf, nb = df.Total_Fwd_Packets, df.Total_Backward_Packets
    F, B = df.Total_Length_of_Fwd_Packets, df.Total_Length_of_Bwd_Packets

    def replay(sub, build, feats, label):
        if len(sub) == 0:
            print(f"  family {label}: no rows in the supplied CSVs - skipped (pass the PortScan CSV too)")
            return
        sub = sub.sample(min(len(sub), 20000), random_state=1)
        got = {f: [] for f in feats}; ref = {f: [] for f in feats}
        for _, r in sub.iterrows():
            d = build(r).to_feature_dict()
            for f in feats:
                got[f].append(d[f]); ref[f].append(r[f])
        print(f"  family {label}: {len(sub):,} rows")
        for f in feats:
            ok = np.isclose(np.array(got[f], float), np.array(ref[f], float), rtol=2e-3, atol=1e-3).mean()
            check(f"{f:30s}", ok > 0.995, f"match {ok:.4f}")

    def flags_for(r, k):                       # spread ACK/URG counts over k packets
        a, u = int(r.ACK_Flag_Count), int(r.URG_Flag_Count)
        return [(ACK if i < a else 0) | (URG if i < u else 0) for i in range(k)]

    def build_1_1(r):
        fl = flags_for(r, 2); dur = r.Flow_Duration / 1e6
        f = Flow("a", 1, "b", int(r.Destination_Port), 6, 0.0)
        f.add_packet(0.0, True, int(r.Total_Length_of_Fwd_Packets), int(r.Fwd_Header_Length), fl[0],
                     None if r.Init_Win_bytes_forward < 0 else r.Init_Win_bytes_forward)
        f.add_packet(dur, False, int(r.Total_Length_of_Bwd_Packets), int(r.Bwd_Header_Length), fl[1],
                     None if r.Init_Win_bytes_backward < 0 else r.Init_Win_bytes_backward)
        return f

    ALL = ["Average_Packet_Size", "Bwd_Packet_Length_Std", "Bwd_Packet_Length_Mean", "Packet_Length_Variance",
           "Packet_Length_Mean", "Destination_Port", "Max_Packet_Length", "Avg_Bwd_Segment_Size",
           "Total_Length_of_Bwd_Packets", "Fwd_Packet_Length_Max", "Bwd_Packet_Length_Max", "Packet_Length_Std",
           "Fwd_Packet_Length_Mean", "Avg_Fwd_Segment_Size", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
           "Fwd_Header_Length.1", "Fwd_Packet_Length_Min", "Total_Length_of_Fwd_Packets", "Subflow_Fwd_Bytes",
           "Total_Fwd_Packets", "Subflow_Bwd_Bytes", "Bwd_Header_Length", "Subflow_Fwd_Packets", "ACK_Flag_Count",
           "Flow_IAT_Mean", "Flow_IAT_Std", "Bwd_Packet_Length_Min", "URG_Flag_Count", "Fwd_Header_Length"]
    replay(df[(nf == 1) & (nb == 1)], build_1_1, ALL, "1 fwd + 1 bwd packet (all 30 features)")

    def build_1_2(r):                           # fwd=1, bwd=2 with bwd min+max == B
        dur = r.Flow_Duration / 1e6
        f = Flow("a", 1, "b", int(r.Destination_Port), 6, 0.0)
        f.add_packet(0.0, True, int(r.Total_Length_of_Fwd_Packets), 0)
        f.add_packet(dur / 2, False, int(r.Bwd_Packet_Length_Min), 0)
        f.add_packet(dur, False, int(r.Bwd_Packet_Length_Max), 0)
        return f
    sub = df[(nf == 1) & (nb == 2) & (B == df.Bwd_Packet_Length_Min + df.Bwd_Packet_Length_Max)]
    replay(sub, build_1_2,
           ["Average_Packet_Size", "Packet_Length_Mean", "Packet_Length_Std", "Packet_Length_Variance",
            "Bwd_Packet_Length_Mean", "Bwd_Packet_Length_Std", "Bwd_Packet_Length_Min", "Bwd_Packet_Length_Max",
            "Flow_IAT_Mean"], "1 fwd + 2 bwd packets (length + IAT-mean features)")

# ---------------------------------------------------------------- Part C
print("\nPART C - scapy parsing and bidirectional matching")
try:
    from scapy.all import Ether, IP, TCP, UDP, Raw
    import realtime_detection as rd

    def wire(pkt):                              # serialise + re-dissect like a real capture
        p = Ether(bytes(pkt)); p.time = pkt.time if hasattr(pkt, "time") else 0.0
        return p

    syn = Ether()/IP(src="10.0.0.1", dst="10.0.0.2")/TCP(sport=40000, dport=80, flags="S", window=29200,
                                                        options=[("MSS", 1460), ("SAckOK", b""), ("WScale", 7)])
    i = rd.packet_info(wire(syn))
    check("SYN: payload 0, TCP header 32 (20 + 12 option bytes), window 29200, SYN flag",
          i["payload"] == 0 and i["hdr"] == 32 and i["window"] == 29200 and i["flags"] & SYN, f"-> {i['payload']}, {i['hdr']}")

    data = Ether()/IP(src="10.0.0.1", dst="10.0.0.2")/TCP(sport=40000, dport=80, flags="PA")/Raw(b"x" * 50)
    i = rd.packet_info(wire(data))
    check("data packet: payload exactly 50, header 20", i["payload"] == 50 and i["hdr"] == 20, f"-> {i['payload']}, {i['hdr']}")

    tiny = Ether()/IP(src="10.0.0.2", dst="10.0.0.1")/TCP(sport=80, dport=40000, flags="A")     # padded to 60-byte frame
    p = wire(tiny); i = rd.packet_info(p)
    check("Ethernet padding is NOT counted as payload", i["payload"] == 0, f"-> {i['payload']}")

    u = Ether()/IP(src="10.0.0.1", dst="8.8.8.8")/UDP(sport=5353, dport=53)/Raw(b"q" * 30)
    i = rd.packet_info(wire(u))
    check("UDP: payload 30, ports parsed", i["payload"] == 30 and i["dport"] == 53, f"-> {i['payload']}")

    # a request/response conversation: replies must land in the SAME flow
    rd.flows.clear()
    t = 1000.0
    seq = [("10.0.0.1", "10.0.0.2", 40000, 80, "S", 0, 29200),
           ("10.0.0.2", "10.0.0.1", 80, 40000, "SA", 0, 14600),
           ("10.0.0.1", "10.0.0.2", 40000, 80, "A", 0, 29200),
           ("10.0.0.1", "10.0.0.2", 40000, 80, "PA", 100, 29200),
           ("10.0.0.2", "10.0.0.1", 80, 40000, "PA", 500, 14600)]
    finished = []
    for k, (s, d_, sp, dp, fl, n, w) in enumerate(seq):
        pkt = Ether()/IP(src=s, dst=d_)/TCP(sport=sp, dport=dp, flags=fl, window=w)/Raw(b"z" * n)
        p = wire(pkt); p.time = t + k * 0.01
        finished += rd.ingest(rd.packet_info(p))
    check("replies merged into ONE flow", len(rd.flows) == 1 and not finished, f"(flows={len(rd.flows)})")
    flow = list(rd.flows.values())[0]; d = flow.to_feature_dict()
    check("forward/backward packet counts 3 / 2", d["Total_Fwd_Packets"] == 3 and len(flow.bwd_lengths) == 2)
    check("Bwd features are non-zero now (the old detector always gave 0)",
          d["Total_Length_of_Bwd_Packets"] == 500 and d["Bwd_Packet_Length_Max"] == 500 and d["Init_Win_bytes_backward"] == 14600)
    check("Destination_Port = 80, Init_Win_forward = 29200", d["Destination_Port"] == 80 and d["Init_Win_bytes_forward"] == 29200)
    check("IAT is in microseconds (10 ms -> 10000)", np.isclose(d["Flow_IAT_Mean"], 10000, rtol=1e-3), f"(= {d['Flow_IAT_Mean']:.1f})")

    fin = wire(Ether()/IP(src="10.0.0.1", dst="10.0.0.2")/TCP(sport=40000, dport=80, flags="FA")); fin.time = t + 1
    finished = rd.ingest(rd.packet_info(fin))
    check("FIN terminates and emits the flow", len(finished) == 1 and len(rd.flows) == 0)

    rd.flows.clear()
    p = wire(Ether()/IP(src="1.1.1.1", dst="2.2.2.2")/TCP(sport=5, dport=22, flags="S")); p.time = 5000.0
    rd.ingest(rd.packet_info(p))
    check("idle flow is expired after IDLE_TIMEOUT", len(rd.expire_idle_flows(now=5000.0 + rd.IDLE_TIMEOUT + 1)) == 1)
except ImportError as e:
    print("  (skipped Part C:", e, ")")

print("\n" + "=" * 60)
print("ALL TESTS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)