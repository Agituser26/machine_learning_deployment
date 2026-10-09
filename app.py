
import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")

def load_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file missing: {filename}")
    return path

class LSTMAnomalyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

scaler = joblib.load(load_file("scaler.joblib"))
le = joblib.load(load_file("label_encoder.joblib"))
feature_columns = joblib.load(load_file("feature_columns.joblib"))

num_features = scaler.n_features_in_
num_classes = len(le.classes_)

model = LSTMAnomalyClassifier(
    input_size=num_features,
    hidden_size=64,
    num_layers=2,
    num_classes=num_classes
)

state = torch.load(
    load_file("lstm_anomaly_model.pth"),
    map_location=device,
    weights_only=True
)
model.load_state_dict(state)
model.eval()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Electrical Panel Anomaly Detection API"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, list) or not data:
            return jsonify({
                "error": "Send a non-empty JSON list of sample objects."
            }), 400

        input_df = pd.DataFrame(data)

        if "Timestamp" in input_df.columns:
            input_df = input_df.drop(columns=["Timestamp"])

        # Match the categories and columns used during training.
        categorical = [
            c for c in ["Sensor_ID", "Location"]
            if c in input_df.columns
        ]

        input_encoded = pd.get_dummies(
            input_df,
            columns=categorical,
            drop_first=True
        )

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        input_scaled = scaler.transform(input_encoded)
        input_seq = input_scaled.reshape(
            len(input_scaled), 1, num_features
        )

        tensor = torch.tensor(input_seq, dtype=torch.float32)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            indices = logits.argmax(dim=1).numpy()

        labels = le.inverse_transform(indices)

        return jsonify({
            "prediction": labels.tolist(),
            "confidence": probabilities.max(dim=1).values.tolist()
        })

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
