"""
01_preprocess.py
Feature extraction from PCAP files for DDoS detection

This script:
1. Reads raw PCAP files using a streaming reader (memory efficient)
2. Extracts 12 features from 100-packet chunks
3. Saves features to CSV for each attack type
4. Creates balanced dataset (1,000 samples per class)
"""

import os
import sys
import pandas as pd
import numpy as np
from scapy.utils import PcapReader
from scapy.layers.inet import IP, TCP, UDP, ICMP
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Configuration
# ============================================
CHUNK_SIZE = 100  # Number of packets per chunk
TARGET_IP = "192.168.0.69"  # Victim IP address

# PCAP files mapping (adjust paths to your setup)
PCAP_FILES = {
    'dns_amplification': 'data/raw_pcap/dns_amplification.pcap',
    'http_flood': 'data/raw_pcap/http_flood.pcap',
    'icmp_flood': 'data/raw_pcap/icmp_flood.pcap',
    'mixed_attack': 'data/raw_pcap/mixed_attack.pcap',
    'normal': 'data/raw_pcap/normal_1.pcap',  # Multiple files for normal
    'syn_flood': 'data/raw_pcap/syn_flood.pcap',
    'udp_flood': 'data/raw_pcap/udp_flood.pcap'
}

# Normal traffic additional files
NORMAL_PCAPS_ADDITIONAL = [
    'data/raw_pcap/normal_2.pcap',
    'data/raw_pcap/normal_3.pcap'
]

# Output directory
OUTPUT_DIR = 'data/extracted_features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# Feature extraction functions
# ============================================
def extract_packet_info(packet):
    """Extract basic information from a single packet"""
    # Get IP layer
    if IP not in packet:
        return None
    
    ip_layer = packet[IP]
    
    # Determine protocol
    if TCP in packet:
        protocol = 'tcp'
    elif UDP in packet:
        protocol = 'udp'
    elif ICMP in packet:
        protocol = 'icmp'
    else:
        protocol = 'other'
    
    # Get packet size
    packet_size = len(packet)
    
    # Get timestamp
    timestamp = float(packet.time)
    
    return {
        'timestamp': timestamp,
        'size': packet_size,
        'protocol': protocol
    }

def extract_chunk_features(chunk_packets):
    """Extract 12 features from a chunk of packets"""
    if len(chunk_packets) == 0:
        return None
    
    n = len(chunk_packets)
    
    # Packet sizes
    sizes = [p['size'] for p in chunk_packets]
    size_mean = np.mean(sizes)
    size_std = np.std(sizes)
    size_min = np.min(sizes)
    size_max = np.max(sizes)
    
    # Protocol counts
    tcp_count = sum(1 for p in chunk_packets if p['protocol'] == 'tcp')
    udp_count = sum(1 for p in chunk_packets if p['protocol'] == 'udp')
    icmp_count = sum(1 for p in chunk_packets if p['protocol'] == 'icmp')
    
    tcp_ratio = tcp_count / n
    udp_ratio = udp_count / n
    icmp_ratio = icmp_count / n
    
    # Timing metrics
    timestamps = [p['timestamp'] for p in chunk_packets]
    duration = timestamps[-1] - timestamps[0]
    
    if duration > 0:
        # Throughput (Mbps)
        total_bytes = sum(sizes)
        throughput_mbps = (total_bytes * 8) / (duration * 1_000_000)
        
        # Packet rate (pps)
        packet_rate_pps = n / duration
        
        # Jitter (standard deviation of inter-arrival times)
        inter_arrival = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if len(inter_arrival) > 0:
            jitter_ms = np.std(inter_arrival) * 1000
        else:
            jitter_ms = 0.0
    else:
        throughput_mbps = 0.0
        packet_rate_pps = 0.0
        jitter_ms = 0.0
    
    return {
        'total_packets': n,
        'size_mean': size_mean,
        'size_std': size_std,
        'size_min': size_min,
        'size_max': size_max,
        'tcp_ratio': tcp_ratio,
        'udp_ratio': udp_ratio,
        'icmp_ratio': icmp_ratio,
        'duration_sec': duration,
        'throughput_mbps': throughput_mbps,
        'jitter_ms': jitter_ms,
        'packet_rate_pps': packet_rate_pps
    }

def process_pcap_file(pcap_path, max_packets=None):
    """Process a PCAP file and extract features from chunks"""
    if not os.path.exists(pcap_path):
        print(f"Warning: File not found: {pcap_path}")
        return []
    
    print(f"Processing: {pcap_path}")
    
    features_list = []
    chunk_packets = []
    packet_count = 0
    
    # Streaming reader (memory efficient)
    with PcapReader(pcap_path) as reader:
        for packet in tqdm(reader, desc="Processing packets", unit="packets"):
            packet_info = extract_packet_info(packet)
            if packet_info is None:
                continue
            
            chunk_packets.append(packet_info)
            packet_count += 1
            
            # When chunk is full, extract features
            if len(chunk_packets) >= CHUNK_SIZE:
                features = extract_chunk_features(chunk_packets)
                if features:
                    features_list.append(features)
                chunk_packets = []
            
            # Stop if max_packets reached
            if max_packets and packet_count >= max_packets:
                break
    
    # Process remaining packets
    if len(chunk_packets) >= CHUNK_SIZE // 2:  # At least half chunk
        features = extract_chunk_features(chunk_packets)
        if features:
            features_list.append(features)
    
    print(f"  Extracted {len(features_list)} chunks")
    return features_list

def sample_features(features_list, n_samples=1000, strategy='two_stage'):
    """Sample features using two-stage strategy"""
    if len(features_list) <= n_samples:
        return features_list
    
    if strategy == 'two_stage':
        # First 500 samples from beginning
        first_stage = features_list[:n_samples//2]
        # Next 500 samples from after first stage
        start_idx = n_samples//2
        second_stage = features_list[start_idx:start_idx + n_samples//2]
        return first_stage + second_stage
    else:
        # Random sampling
        indices = np.random.choice(len(features_list), n_samples, replace=False)
        return [features_list[i] for i in indices]

# ============================================
# Main processing pipeline
# ============================================
def main():
    print("="*60)
    print("Feature Extraction from PCAP Files")
    print("="*60)
    
    all_data = {}
    
    # Process each attack type
    for attack_type, pcap_path in PCAP_FILES.items():
        print(f"\n--- Processing: {attack_type} ---")
        
        if attack_type == 'normal':
            # Process multiple normal PCAP files
            all_features = []
            for normal_pcap in [pcap_path] + NORMAL_PCAPS_ADDITIONAL:
                features = process_pcap_file(normal_pcap)
                all_features.extend(features)
        else:
            all_features = process_pcap_file(pcap_path)
        
        # Sample to 1,000 samples
        sampled_features = sample_features(all_features, n_samples=1000)
        
        # Add attack type label
        for features in sampled_features:
            features['attack_type'] = attack_type
        
        all_data[attack_type] = sampled_features
        print(f"  Final samples: {len(sampled_features)}")
        
        # Save individual CSV
        df = pd.DataFrame(sampled_features)
        df.to_csv(f"{OUTPUT_DIR}/{attack_type}_features.csv", index=False)
    
    # Create balanced dataset
    print("\n" + "="*60)
    print("Creating Balanced Dataset")
    print("="*60)
    
    all_dfs = []
    for attack_type, features_list in all_data.items():
        df = pd.DataFrame(features_list)
        all_dfs.append(df)
        print(f"{attack_type}: {len(df)} samples")
    
    # Concatenate all classes
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save final dataset
    final_df.to_csv('data/balanced_dataset.csv', index=False)
    
    print("\n" + "="*60)
    print("Final Dataset Summary")
    print("="*60)
    print(f"Total samples: {len(final_df)}")
    print(f"Class distribution:")
    print(final_df['attack_type'].value_counts())
    print(f"\n✓ Dataset saved to: data/balanced_dataset.csv")
    
    return final_df

if __name__ == "__main__":
    df = main()