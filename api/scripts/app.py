#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
import torch.nn as nn
import io
import os

#model class
class NDISModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NDISModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


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
output_dim = 2
model = NDISModel(input_dim=input_dim, output_dim=output_dim)
torch.save(model.state_dict(), model_path)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))

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
    '4': 'DoS slowloris',
    '5': 'DoS Slowhttptest',
    '6': 'DoS Slowloris',
    '7': 'DoS Slowhttptest',
    '8': 'DoS Hulk',
    '9': 'DoS Slowhttptest',
    '10': 'DoS Slowhttptest',
    '11': 'DoS Slowhttptest',
    '12': 'DoS Slowhttptest',
    '13': 'DoS Slowhttptest',
    '14': 'DoS Slowhttptest',
    '15': 'DoS Slowhttptest',
    '16': 'DoS Slowhttptest',
    '17': 'DoS Slowhttptest',
    '18': 'DoS Slowhttptest',
    '19': 'DoS Slowhttptest',
    '20': 'DoS Slowhttptest',
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

        # Convert to PyTorch tensor
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