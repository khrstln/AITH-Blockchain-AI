import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import pyarrow.parquet as pq


INPUT_SIZE = 13
HIDDEN_SIZE = INPUT_SIZE * 4
NUM_LAYERS = 2
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 50
PREDICT_HORIZON = 50
BATCH_SIZE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        preds = predict_future(model, dataloader, DEVICE)

        preds = np.array(preds)

        xticks = ["" for _ in range(0, SEQUENCE_LENGTH + 1, 5)]
        for i in range(1, SEQUENCE_LENGTH // 5 + 1):
            xticks[i] = f"{i} сек"

        xticks[0] = "Now"

        fig, ax = plt.subplots()
        ax.plot(preds, label="Gas price")
        ax.set_title("Прогноз Gas price на ближайшие 10 секунд")
        ax.set_xlabel("Время")
        ax.set_ylabel("Значение Gas price")
        ax.set_xticks(range(0, SEQUENCE_LENGTH + 1, 5))
        ax.set_xticklabels(xticks, rotation=-45)
        ax.legend()

        chart_placeholder.pyplot(fig)

        time.sleep(0.5)

    time.sleep(2)
