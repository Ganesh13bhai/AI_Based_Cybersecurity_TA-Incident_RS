import json
from pathlib import Path


MAPPING = {

    "flow_duration": {
        "CIC-IDS2017": "Flow Duration",
        "CIC-IDS2018": "Flow Duration",
        "UNSW-NB15": "dur"
    },

    "protocol": {
        "CIC-IDS2017": None,
        "CIC-IDS2018": "Protocol",
        "UNSW-NB15": "proto"
    },

    "destination_port": {
        "CIC-IDS2017": "Destination Port",
        "CIC-IDS2018": None,
        "UNSW-NB15": None
    },

    "service": {
        "CIC-IDS2017": None,
        "CIC-IDS2018": None,
        "UNSW-NB15": "service"
    },

    "flow_forward_packets": {
        "CIC-IDS2017": "Total Fwd Packets",
        "CIC-IDS2018": "Total Fwd Packets",
        "UNSW-NB15": "spkts"
    },

    "flow_backward_packets": {
        "CIC-IDS2017": "Total Backward Packets",
        "CIC-IDS2018": "Total Backward Packets",
        "UNSW-NB15": "dpkts"
    },

    "flow_forward_bytes": {
        "CIC-IDS2017": "Total Length of Fwd Packets",
        "CIC-IDS2018": "Fwd Packets Length Total",
        "UNSW-NB15": "sbytes"
    },

    "flow_backward_bytes": {
        "CIC-IDS2017": "Total Length of Bwd Packets",
        "CIC-IDS2018": "Bwd Packets Length Total",
        "UNSW-NB15": "dbytes"
    },

    "flow_bytes_per_second": {
        "CIC-IDS2017": "Flow Bytes/s",
        "CIC-IDS2018": "Flow Bytes/s",
        "UNSW-NB15": None
    },

    "flow_packets_per_second": {
        "CIC-IDS2017": "Flow Packets/s",
        "CIC-IDS2018": "Flow Packets/s",
        "UNSW-NB15": None
    },

    "forward_packet_length_max": {
        "CIC-IDS2017": "Fwd Packet Length Max",
        "CIC-IDS2018": "Fwd Packet Length Max",
        "UNSW-NB15": None
    },

    "forward_packet_length_min": {
        "CIC-IDS2017": "Fwd Packet Length Min",
        "CIC-IDS2018": "Fwd Packet Length Min",
        "UNSW-NB15": None
    },

    "forward_packet_length_mean": {
        "CIC-IDS2017": "Fwd Packet Length Mean",
        "CIC-IDS2018": "Fwd Packet Length Mean",
        "UNSW-NB15": None
    },

    "forward_packet_length_std": {
        "CIC-IDS2017": "Fwd Packet Length Std",
        "CIC-IDS2018": "Fwd Packet Length Std",
        "UNSW-NB15": None
    },

    "backward_packet_length_max": {
        "CIC-IDS2017": "Bwd Packet Length Max",
        "CIC-IDS2018": "Bwd Packet Length Max",
        "UNSW-NB15": None
    },

    "backward_packet_length_min": {
        "CIC-IDS2017": "Bwd Packet Length Min",
        "CIC-IDS2018": "Bwd Packet Length Min",
        "UNSW-NB15": None
    },

    "backward_packet_length_mean": {
        "CIC-IDS2017": "Bwd Packet Length Mean",
        "CIC-IDS2018": "Bwd Packet Length Mean",
        "UNSW-NB15": None
    },

    "backward_packet_length_std": {
        "CIC-IDS2017": "Bwd Packet Length Std",
        "CIC-IDS2018": "Bwd Packet Length Std",
        "UNSW-NB15": None
    },

    "forward_iat_total": {
        "CIC-IDS2017": "Fwd IAT Total",
        "CIC-IDS2018": "Fwd IAT Total",
        "UNSW-NB15": None
    },

    "forward_iat_mean": {
        "CIC-IDS2017": "Fwd IAT Mean",
        "CIC-IDS2018": "Fwd IAT Mean",
        "UNSW-NB15": None
    },

    "forward_iat_std": {
        "CIC-IDS2017": "Fwd IAT Std",
        "CIC-IDS2018": "Fwd IAT Std",
        "UNSW-NB15": None
    },

    "forward_iat_max": {
        "CIC-IDS2017": "Fwd IAT Max",
        "CIC-IDS2018": "Fwd IAT Max",
        "UNSW-NB15": None
    },

    "forward_iat_min": {
        "CIC-IDS2017": "Fwd IAT Min",
        "CIC-IDS2018": "Fwd IAT Min",
        "UNSW-NB15": None
    },

    "backward_iat_total": {
        "CIC-IDS2017": "Bwd IAT Total",
        "CIC-IDS2018": "Bwd IAT Total",
        "UNSW-NB15": None
    },

    "backward_iat_mean": {
        "CIC-IDS2017": "Bwd IAT Mean",
        "CIC-IDS2018": "Bwd IAT Mean",
        "UNSW-NB15": None
    },

    "backward_iat_std": {
        "CIC-IDS2017": "Bwd IAT Std",
        "CIC-IDS2018": "Bwd IAT Std",
        "UNSW-NB15": None
    },

    "backward_iat_max": {
        "CIC-IDS2017": "Bwd IAT Max",
        "CIC-IDS2018": "Bwd IAT Max",
        "UNSW-NB15": None
    },

    "backward_iat_min": {
        "CIC-IDS2017": "Bwd IAT Min",
        "CIC-IDS2018": "Bwd IAT Min",
        "UNSW-NB15": None
    },

    "forward_packets_per_second": {
        "CIC-IDS2017": "Fwd Packets/s",
        "CIC-IDS2018": "Fwd Packets/s",
        "UNSW-NB15": None
    },

    "backward_packets_per_second": {
        "CIC-IDS2017": "Bwd Packets/s",
        "CIC-IDS2018": "Bwd Packets/s",
        "UNSW-NB15": None
    },

    "down_up_ratio": {
        "CIC-IDS2017": "Down/Up Ratio",
        "CIC-IDS2018": "Down/Up Ratio",
        "UNSW-NB15": None
    },

    "average_packet_size": {
        "CIC-IDS2017": "Average Packet Size",
        "CIC-IDS2018": "Avg Packet Size",
        "UNSW-NB15": None
    },

    "forward_mean_segment_size": {
        "CIC-IDS2017": "Avg Fwd Segment Size",
        "CIC-IDS2018": "Avg Fwd Segment Size",
        "UNSW-NB15": "smean"
    },

    "backward_mean_segment_size": {
        "CIC-IDS2017": "Avg Bwd Segment Size",
        "CIC-IDS2018": "Avg Bwd Segment Size",
        "UNSW-NB15": "dmean"
    },

    "syn_flag_count": {
        "CIC-IDS2017": "SYN Flag Count",
        "CIC-IDS2018": "SYN Flag Count",
        "UNSW-NB15": None
    },

    "ack_flag_count": {
        "CIC-IDS2017": "ACK Flag Count",
        "CIC-IDS2018": "ACK Flag Count",
        "UNSW-NB15": None
    },

    "rst_flag_count": {
        "CIC-IDS2017": "RST Flag Count",
        "CIC-IDS2018": "RST Flag Count",
        "UNSW-NB15": None
    },

    "fin_flag_count": {
        "CIC-IDS2017": "FIN Flag Count",
        "CIC-IDS2018": "FIN Flag Count",
        "UNSW-NB15": None
    },

    "psh_flag_count": {
        "CIC-IDS2017": "PSH Flag Count",
        "CIC-IDS2018": "PSH Flag Count",
        "UNSW-NB15": None
    },

    "urg_flag_count": {
        "CIC-IDS2017": "URG Flag Count",
        "CIC-IDS2018": "URG Flag Count",
        "UNSW-NB15": None
    },

    "initial_forward_window": {
        "CIC-IDS2017": "Init_Win_bytes_forward",
        "CIC-IDS2018": "Init Fwd Win Bytes",
        "UNSW-NB15": "swin"
    },

    "initial_backward_window": {
        "CIC-IDS2017": "Init_Win_bytes_backward",
        "CIC-IDS2018": "Init Bwd Win Bytes",
        "UNSW-NB15": "dwin"
    },

    "forward_data_packets": {
        "CIC-IDS2017": "act_data_pkt_fwd",
        "CIC-IDS2018": "Fwd Act Data Packets",
        "UNSW-NB15": None
    },

    "forward_segment_size_min": {
        "CIC-IDS2017": "min_seg_size_forward",
        "CIC-IDS2018": "Fwd Seg Size Min",
        "UNSW-NB15": None
    },

    "tcp_round_trip_time": {
        "CIC-IDS2017": None,
        "CIC-IDS2018": None,
        "UNSW-NB15": "tcprtt"
    },

    "tcp_syn_ack_time": {
        "CIC-IDS2017": None,
        "CIC-IDS2018": None,
        "UNSW-NB15": "synack"
    },

    "tcp_ack_data_time": {
        "CIC-IDS2017": None,
        "CIC-IDS2018": None,
        "UNSW-NB15": "ackdat"
    }
}


LABELS = {
    "CIC-IDS2017": {
        "original_label": "Label",
        "canonical_label": "canonical_label",
        "binary_label": "binary_label"
    },
    "CIC-IDS2018": {
        "original_label": "Label",
        "canonical_label": "canonical_label",
        "binary_label": "binary_label"
    },
    "UNSW-NB15": {
        "original_label": "attack_cat",
        "binary_label": "label",
        "canonical_label": "canonical_label"
    }
}


output = {
    "version": "1.0",
    "description": "Semantic feature mapping for CyberSecAI",
    "datasets": [
        "CIC-IDS2017",
        "CIC-IDS2018",
        "UNSW-NB15"
    ],
    "features": MAPPING,
    "labels": LABELS
}


output_path = Path("data/metadata/feature_mapping.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print("=" * 60)
print("CyberSecAI Feature Mapping")
print("=" * 60)
print(f"Canonical features: {len(MAPPING)}")
print(f"Saved to: {output_path}")