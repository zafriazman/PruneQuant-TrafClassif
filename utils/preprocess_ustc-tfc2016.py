import os
import gc
import time
import random
import torch
import zipfile
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from scapy.all import rdpcap, IP, TCP, UDP, Padding


def unifying_packet_to_bytes(packet, total_length=1500, ip_header_length=20, tcp_udp_header_length=20):
    """
    Preprocess a packet to have a unified length of 1500 bytes.
    
    Parameters:
        packet (scapy.Packet): A single packet to preprocess.
        total_length (int): The total desired length of the packet in term of bytes.
        ip_header_length (int): The length of the IP header.
        tcp_udp_header_length (int): The length of the TCP/UDP header.
    
    Returns:
        bytes: The preprocessed packet data in raw bytes format.
    """

    packet_bytes = bytes()

    # Check if the packet is IP, TCP or UDP
    if IP in packet and (TCP in packet or UDP in packet):
        # For IP layer, we'll use 20 bytes (user specified) of zeros to mask it
        ip_header = bytes(ip_header_length)  # 20 bytes of zeros
        
        # For transport layer (TCP/UDP), we'll use the actual header, with truncate or zero pad to 20 bytes
        if TCP in packet:
            transport_header = bytes(packet[TCP])[:tcp_udp_header_length]
        elif UDP in packet:
            transport_header = bytes(packet[UDP])[:tcp_udp_header_length]
        
        # Pad to ensure it's exactly 20 bytes
        transport_header += bytes(tcp_udp_header_length - len(transport_header))
        
        # Calculate the length for the payload to fit into the desired total length
        desired_payload_length = total_length - ip_header_length - tcp_udp_header_length

        # Extract the actual payload from the packet
        payload = bytes(packet[IP].payload.payload)[:desired_payload_length]  # Truncate if longer
        payload += bytes(desired_payload_length - len(payload))  # Pad with zeros if shorter
        
        # Concatenate the parts and then pad or truncate to meet the total length
        packet_bytes = ip_header + transport_header + payload
        
        # print(packet_bytes.hex())
        return packet_bytes
    else:
        return None

def normalize_packet_bytes(packet_bytes):
    """
    Normalize the packet bytes to a range of [0, 1].
    
    Parameters:
        packet_bytes (bytes): The raw packet data as a bytes object.
    
    Returns:
        np.ndarray: A numpy array of normalized packet data.
    """
    # Convert the bytes object to a numpy array of type uint8 (to handle byte values)
    packet_array = np.frombuffer(packet_bytes, dtype=np.uint8)
    
    # Normalize the array by dividing each element by 255 (1 bytes max val is 255)
    normalized_array = packet_array / 255.0
    
    return normalized_array


def preprocess_and_select_packets(file_paths, selection_limit=7000):
    all_packets = []
    for file_path in file_paths:
        packets = rdpcap(file_path)
        for packet in packets:
            if IP in packet and (TCP in packet or UDP in packet):
                all_packets.append(packet)  # Aggregate packets from all files

    # Randomly select up to 7000 packets
    print(f"Number of packet: {len(all_packets)}")
    if len(all_packets) > selection_limit:
        selected_indices = random.sample(range(len(all_packets)), selection_limit)
        selected_packets = [all_packets[i] for i in selected_indices]
    else:
        selected_packets = all_packets

    processed_packets = []
    for packet in selected_packets:
        packet_bytes = unifying_packet_to_bytes(packet)
        if packet_bytes is not None:
            normalized_packet = normalize_packet_bytes(packet_bytes)
            processed_packets.append(normalized_packet)

    return processed_packets

def preprocess_dataset():
    '''
    Check if already preprocess dataset. If not, preprocess it
    '''

    extracted_pcaps_dir = os.path.join('data', 'datasets', 'ustc-tfc2016-extracted-pcaps')
    final_path = os.path.join('data', 'datasets', 'ustc-tfc2016-pytorch')
    final_hash = {}
    random.seed(2024)
    

    if os.path.exists(os.path.join(final_path, 'Zeus.pt')):
        print("Prepocessed data already exists, skip preprocessing")
        return
    else:
        os.makedirs(final_path)
    


    # Easiest way to capture the filename without regex
    class_naming = {
        'BitTorrent': ['BitTorrent'],
        'Cridex' : ['Cridex'],
        'Facetime' : ['Facetime'],
        'FTP' : ['FTP'],
        'Geodo' : ['Geodo'],
        'Gmail' : ['Gmail'],
        'Htbot' : ['Htbot'],
        'Miuref' : ['Miuref'],
        'MySQL' : ['MySQL'],
        'Neris' : ['Neris'],
        'Nsis-ay' : ['Nsis-ay'],
        'Outlook' : ['Outlook'],
        'Shifu' : ['Shifu'],
        'Skype' : ['Skype'],
        'SMB' : ['SMB-1', 'SMB-2'],
        'Tinba' : ['Tinba'],
        'Virut' : ['Virut'],
        'Weibo' : ['Weibo-1', 'Weibo-2', 'Weibo-3', 'Weibo-4'],
        'WorldOfWarcraft' : ['WorldOfWarcraft'],
        'Zeus' : ['Zeus'],
    }


    # Initialize a dictionary with the declared class name
    final_hash = {key: {'filelist': []} for key in class_naming}


    # Get all the files that is in extracted directory,
    # and create a hashmap that contains the filelist, according to the class
    for filename in os.listdir(extracted_pcaps_dir):

        if filename.endswith('.pcap') or filename.endswith('.pcapng'):
            pcap_file = os.path.join(extracted_pcaps_dir, filename)

            for key, variations in class_naming.items():
                # if one of the value in class_naming matches the file name, put in dictionary (final_hash)
                if any(variation.lower() in filename.lower() for variation in variations):
                    final_hash[key]['filelist'].append(pcap_file)



    # Create the dataset that can be used with Pytorch's data loader
    for class_name, data in final_hash.items():

        start_time = time.time()
        gc.collect()  # Suggest to the garbage collector to free unused memory

        processed_packets = preprocess_and_select_packets(data['filelist'], 7000)

        current_duration = round(time.time() - start_time)
        
        all_packets_np = np.array(processed_packets)  # Convert list of NumPy arrays to a single NumPy array

        # Convert the list of packets to a PyTorch tensor and save
        class_tensor = torch.tensor(all_packets_np, dtype=torch.float32)
        torch.save(class_tensor, os.path.join(final_path, f"{class_name}.pt"))

        print(f"Finish parsing {class_name} pcap files ... for {current_duration}s")



if __name__ == "__main__":
    preprocess_dataset()   
