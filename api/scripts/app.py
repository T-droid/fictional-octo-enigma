#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
import torch.nn as nn
import io
import os
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

#model class
class NDISModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NDISModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  
        self.fc2 = nn.Linear(512, 256)       
        self.fc3 = nn.Linear(256, 128)       
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc1(x))  
        x = self.dropout(x)         
        x = self.relu(self.fc2(x))  
        x = self.dropout(x)         
        x = self.relu(self.fc3(x))  
        x = self.fc4(x)             
        return x
    
# Web socket connection management
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()
       

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "NIDS_Model_1.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
input_dim = 47  
output_dim = 15  #changed from 2 to 15 classes
model = NDISModel(input_dim=input_dim, output_dim=output_dim)
torch.save(model.state_dict(), model_path)

model.eval()


column_mapping = {
    'Dst Port': 'Destination Port',
    'Flow Duration': 'Flow Duration',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Flow IAT Std': 'Flow IAT Std',
    'Flow IAT Max': 'Flow IAT Max',
    'Flow IAT Min': 'Flow IAT Min',
    'Fwd IAT Tot': 'Fwd IAT Total',  # Rename to match the expected name
    'Fwd IAT Mean': 'Fwd IAT Mean',
    'Fwd IAT Std': 'Fwd IAT Std',
    'Fwd IAT Max': 'Fwd IAT Max',
    'Fwd IAT Min': 'Fwd IAT Min',
    'Bwd IAT Tot': 'Bwd IAT Total',  # Rename to match the expected name
    'Bwd IAT Mean': 'Bwd IAT Mean',
    'Bwd IAT Std': 'Bwd IAT Std',
    'Bwd IAT Max': 'Bwd IAT Max',
    'Bwd IAT Min': 'Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    'Bwd PSH Flags': 'Bwd PSH Flags',
    'Fwd URG Flags': 'Fwd URG Flags',
    'Bwd URG Flags': 'Bwd URG Flags',
    'Fwd Header Len': 'Fwd Header Length',  # Rename to match the expected name
    'Bwd Header Len': 'Bwd Header Length',  # Rename to match the expected name
    'Fwd Pkts/s': 'Fwd Packets/s',  # Ensure naming consistency
    'Bwd Pkts/s': 'Bwd Packets/s',
    'Pkt Len Min': 'Min Packet Length',  # Rename to match the expected name
    'FIN Flag Cnt': 'FIN Flag Count',  # Rename to match the expected name
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'CWE Flag Count': 'CWE Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count',
    'Down/Up Ratio': 'Down/Up Ratio',
    'Pkt Size Avg': 'Pkt Size Avg',
    'Fwd Seg Size Avg': 'Fwd Seg Size Avg',
    'Bwd Seg Size Avg': 'Bwd Seg Size Avg',
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',  # Rename to match the expected name
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',  # Rename to match the expected name
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts': 'Subflow Fwd Pkts',
    'Subflow Fwd Byts': 'Subflow Fwd Byts',
    'Subflow Bwd Pkts': 'Subflow Bwd Pkts',
    'Subflow Bwd Byts': 'Subflow Bwd Byts',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',  # Rename to match the expected name
    'Init Bwd Win Byts': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',  # Rename to match the expected name
    'Fwd Seg Size Min': 'min_seg_size_forward',  # Rename to match the expected name
    'Active Mean': 'Active Mean',
    'Active Std': 'Active Std',
    'Active Max': 'Active Max',
    'Active Min': 'Active Min',
    'Idle Mean': 'Idle Mean',
    'Idle Std': 'Idle Std',
    'Idle Max': 'Idle Max',
    'Idle Min': 'Idle Min',
    'Label': 'Label'
}
selected_columns = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Min', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Bwd Packets/s', 'Min Packet Length', 'FIN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Idle Std'
]

vulnarability_mapping = {
    '0': 'Benign',
    '1': 'Bot',
    '2': 'DoS Hulk',
    '3': 'DoS GoldenEye',
    '4': 'DoS Hulk',
    '5': 'DoS Slowhttptest',
    '6': 'DoS Slowloris',
    '7': 'FTP-Patator',
    '8': 'Heartbleed',
    '9': 'infiltration',
    '10': 'PortScan',
    '11': 'SSH-Patator',
    '12': 'Web Attack Brute Force',
    '13': 'Web Attack Sql Injection',
    '14': 'Web Attack XSS'
}


def process_csv(file: UploadFile):
    try:
        # Load the uploaded file into a pandas DataFrame
        contents = file.file.read()
        data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Rename columns
        data = data.rename(columns=column_mapping)

        # Check for missing columns
        missing_columns = [col for col in selected_columns if col not in data.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )

        # Select required columns
        data = data[selected_columns]

        # Convert to PyTorch tensor] Operation not permitted
        data_tensor = torch.tensor(data.values, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            outputs = model(data_tensor)
            _, predictions = torch.max(outputs, dim=1)
            predictions = predictions.tolist()
        # Add predictions to the DataFrame
        data["predictions"] = [vulnarability_mapping[str(pred)] for pred in predictions]

        # Convert result to JSON
        # In process_csv function, modify the final part to:
        predictions_with_index = [{"row": idx, "prediction": pred} for idx, pred in enumerate(data["predictions"].tolist())]
        print(predictions_with_index),
        return predictions_with_index

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
import scapy.all as scapy
import numpy as np
import time
from collections import defaultdict

# Track flow data
flows = defaultdict(dict)

# List of selected columns/features required for the model
selected_columns = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Min', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Bwd Packets/s', 'Min Packet Length', 'FIN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Idle Std'
]

# Feature extraction for a given packet
def extract_packet_features(packet, flow_key):
    flow = flows[flow_key]
    
    if packet.haslayer(scapy.IP):
        ip_src = packet[scapy.IP].src
        ip_dst = packet[scapy.IP].dst
        proto = packet[scapy.IP].proto
        destination_port = packet.sport if proto == 6 else (packet.dport if proto == 17 else 0)  # Capture destination port for TCP/UDP
        
        # Initialize flow if it's the first packet
        if 'start_time' not in flow:
            flow['start_time'] = time.time()
        flow['last_time'] = time.time()

        # Track total packets and bytes
        flow['total_packets'] = flow.get('total_packets', 0) + 1
        flow['total_bytes'] = flow.get('total_bytes', 0) + len(packet)

        # Extract flags (FIN, RST, PSH, ACK, URG) if it's a TCP packet
        if packet.haslayer(scapy.TCP):
            flags = packet[scapy.TCP].flags
            flow['FIN'] = flow.get('FIN', 0) + (1 if 'F' in flags else 0)
            flow['RST'] = flow.get('RST', 0) + (1 if 'R' in flags else 0)
            flow['PSH'] = flow.get('PSH', 0) + (1 if 'P' in flags else 0)
            flow['ACK'] = flow.get('ACK', 0) + (1 if 'A' in flags else 0)
            flow['URG'] = flow.get('URG', 0) + (1 if 'U' in flags else 0)
        
        # Compute Flow Duration (time difference between first and last packet)
        flow['duration'] = flow['last_time'] - flow['start_time']
        
        # Calculate Inter-packet Time (IAT)
        flow['iat'] = flow.get('iat', []) + [time.time() - flow['last_time']]
        
        # Min/Max Packet Length
        flow['min_packet_len'] = min(flow.get('min_packet_len', float('inf')), len(packet))
        flow['max_packet_len'] = max(flow.get('max_packet_len', 0), len(packet))
        
        # Calculate Flow Bytes/s and Flow Packets/s
        flow['flow_bytes_per_s'] = flow['total_bytes'] / flow['duration'] if flow['duration'] > 0 else 0
        flow['flow_packets_per_s'] = flow['total_packets'] / flow['duration'] if flow['duration'] > 0 else 0
        
        # Calculate Forward and Backward IAT Mean/Std/Min
        if 'fwd_iat' not in flow:
            flow['fwd_iat'] = []
        if 'bwd_iat' not in flow:
            flow['bwd_iat'] = []
        
        # Separate Forward and Backward Packets (can be done based on direction, src -> dst is fwd, dst -> src is bwd)
        if packet[scapy.IP].src == ip_src:
            flow['fwd_iat'].append(time.time() - flow['last_time'])
        else:
            flow['bwd_iat'].append(time.time() - flow['last_time'])
        
        # Return extracted features (after processing the packet)
        return flow
    return None

# Function to calculate safe statistics from lists
def safe_statistic(lst, stat_type):
    if len(lst) > 0:
        if stat_type == 'mean':
            return np.mean(lst)
        elif stat_type == 'std':
            return np.std(lst)
        elif stat_type == 'min':
            return np.min(lst)
        elif stat_type == 'max':
            return np.max(lst)
    return 0  # Default to 0 if list is empty

# Function to capture packets and extract features
async def capture_traffic_and_extract_features():
    def packet_callback(packet):
        if packet.haslayer(scapy.TCP):
            if packet[scapy.TCP].flags == "F":
                return 
            
            ip_src = packet[scapy.IP].src
            ip_dst = packet[scapy.IP].dst
            flow_key = (ip_src, ip_dst)

            extract_packet_features(packet, flow_key)
            flow = flows[flow_key]

            features = [
                flow.get('destination_port', 0),
                flow['duration'], flow['total_packets'], flow['total_bytes'],
                flow.get('fwd_packet_len_max', 0), flow.get('fwd_packet_len_min', 0),
                safe_statistic(flow.get('fwd_iat', []), 'mean'),
                safe_statistic(flow.get('fwd_iat', []), 'std'),
                safe_statistic(flow.get('fwd_iat', []), 'min'),
                safe_statistic(flow.get('bwd_iat', []), 'min'),
                safe_statistic(flow.get('bwd_iat', []), 'sum'),
                safe_statistic(flow.get('bwd_iat', []), 'mean'),
                safe_statistic(flow.get('bwd_iat', []), 'std'),
                safe_statistic(flow.get('bwd_iat', []), 'max'),
                flow['FIN'], flow['RST'], flow['PSH'], flow['ACK'], flow['URG'],
                flow.get('fwd_header_len', 0), flow.get('bwd_header_len', 0),
                flow.get('bwd_packets_per_s', 0), flow.get('min_packet_len', 0),
                flow['flow_bytes_per_s'], flow['flow_packets_per_s']
            ]
            
            # Convert features to tensor and get prediction
            features_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                output = model(features_tensor)
                _, prediction = torch.max(output, dim=1)
                attack_type = vulnarability_mapping[str(prediction.item())]

            # Prepare prediction data
            prediction_data = {
                "source_ip": ip_src,
                "destination_ip": ip_dst,
                "attack_type": attack_type,
                "timestamp": time.time()
            }

            # Broadcast prediction to all connected clients
            asyncio.create_task(manager.broadcast(prediction_data))
    
    try:
        scapy.sniff(prn=packet_callback, store=0, filter="ip")
    except Exception as e:
        print(f"Error occurred during sniffing: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await capture_traffic_and_extract_features()
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV file and get predictions.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    predictions = process_csv(file)
    return JSONResponse(content={"predictions": predictions})



@app.get("/")
async def root():
    return {"message": "Welcome to the NIDS API!"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)