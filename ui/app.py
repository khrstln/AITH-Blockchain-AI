import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

INPUT_SIZE = 13
HIDDEN_SIZE = INPUT_SIZE * 4
NUM_LAYERS = 2
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 50
PREDICT_HORIZON = 50
BATCH_SIZE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALER = joblib.load(r"models\tx_scaler.pkl")
NUMERIC_COLS = ["cost", "gas", "gas_fee_cap", "gas_price"]
GAS_INDEX = NUMERIC_COLS.index("gas_price")
CENTER = SCALER.center_[GAS_INDEX]
SCALE = SCALER.scale_[GAS_INDEX]


class BasicBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(BasicBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class SequenceDataset(Dataset):
    def __init__(self, df: pl.DataFrame, seq_length: int, target_col: str):
        self.seq_length = seq_length
        feature_columns = [col for col in df.columns if col != target_col]
        self.features = df.select(feature_columns).to_numpy()
        self.targets = df[target_col].to_numpy()

    def __len__(self):
        return self.features.shape[0] - self.seq_length

    def __getitem__(self, idx):
        seq_x = self.features[idx : idx + self.seq_length]
        target = self.targets[idx + self.seq_length]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)


@st.cache_resource
def load_model():
    model = BasicBiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    model.load_state_dict(torch.load("models/bi_lstm/best_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_data
def load_data():
    return pl.read_parquet("data/processed/test.parquet")


def predict_future(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for seq, target in dataloader:
            seq = seq.to(device)
            target = target.to(device)
            output = model(seq)
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions


st.set_page_config(page_title="Gas Price Forecast", layout="centered")
st.title("Прогноз Gas price")

model = load_model()
df = load_data()

chart_placeholder = st.empty()

while True:
    for i in range(0, len(df)):
        df_to_predict = df.slice(i, SEQUENCE_LENGTH + PREDICT_HORIZON)
        dataset = SequenceDataset(df_to_predict, SEQUENCE_LENGTH, "gas_price")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        preds = predict_future(model, dataloader, DEVICE).flatten()
        preds_unscaled = preds * SCALE + CENTER

        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=j) for j in range(len(preds))]

        fig = go.Figure(
            go.Scatter(
                x=timestamps,
                y=preds_unscaled,
                mode="lines+markers",
                line=dict(color="#FF9900", width=2),
                marker=dict(color="#FF9900", size=6, line=dict(color="white", width=0.5)),
                name="Gas price",
            )
        )

        fig.update_layout(
            title="",
            xaxis_title="Время",
            yaxis_title="Gas price",
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font_color="white",
            xaxis=dict(gridcolor="gray", tickformat="%H:%M:%S"),
            yaxis=dict(gridcolor="gray"),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(1)
    time.sleep(2)
