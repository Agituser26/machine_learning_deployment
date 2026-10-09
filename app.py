import torch
import torch.nn as nn
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import torch
import joblib
app = Flask(__name__)
# Ensure model exists in local workspace and save it
if 'model' in globals():
    torch.save(model.state_dict(), 'lstm_anomaly_model.pth')
    print("Successfully saved 'lstm_anomaly_model.pth'")
else:
    print("Error: 'model' variable was not found in the current session. Please run the model training cell first.")

# Save the scaler and encoder
if 'scaler' in globals():
    joblib.dump(scaler, 'scaler.joblib')
    print("Successfully saved 'scaler.joblib'")
if 'le' in globals():
    joblib.dump(le, 'label_encoder.joblib')
    print("Successfully saved 'label_encoder.joblib'")

# Define the LSTM model class (must be defined to load the state_dict)
class LSTMAnomalyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAnomalyClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- Load the saved model components --- 
# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the scaler and label encoder
scaler = joblib.load('scaler.joblib')
le = joblib.load('label_encoder.joblib')

# Get num_features and num_classes from the loaded components
# Assuming num_features was implicitly determined by the scaler's fit state or X_scaled shape
# Here we get it from the scaler.  It represents the number of features *after* one-hot encoding.
num_features_loaded = scaler.n_features_in_
num_classes_loaded = len(le.classes_)

# Instantiate the model with the correct parameters
model = LSTMAnomalyClassifier(input_size=num_features_loaded, hidden_size=64, num_layers=2, num_classes=num_classes_loaded).to(device)

# Load the model's state dictionary
model.load_state_dict(torch.load('lstm_anomaly_model.pth', map_location=device))
model.eval() # Set the model to evaluation mode

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) # Get data posted as JSON
            # Ensure input data keys match the original DataFrame columns used for training features
            # Exclude 'Timestamp', 'Anomaly_Label', and the one-hot encoded columns (handled by pd.get_dummies)
            
            # Create a DataFrame from the input data
            # The input should be a list of dictionaries, each dict representing a sample
            input_df = pd.DataFrame(data)
            
            # Drop 'Timestamp' if present, as it was dropped during training
            if 'Timestamp' in input_df.columns:
                input_df = input_df.drop(columns=['Timestamp'])
                
            # One-hot encode categorical features ('Sensor_ID', 'Location')
            # Ensure the columns match what was used in training, including the `drop_first=True`
            input_df_encoded = pd.get_dummies(input_df, columns=['Sensor_ID', 'Location'], drop_first=True)
            
            # Reindex to ensure all columns from training are present, filling missing with 0 (for one-hot)
            # This is crucial if some categories are missing in the inference input
            training_columns = X_encoded.columns.tolist() # Assuming X_encoded is still available from previous cell
            input_df_encoded = input_df_encoded.reindex(columns=training_columns, fill_value=0)

            # Convert to numpy array and scale
            input_X = input_df_encoded.values
            input_X_scaled = scaler.transform(input_X)
            
            # Reshape for LSTM (samples, sequence_length, features)
            input_X_seq = input_X_scaled.reshape(input_X_scaled.shape[0], 1, num_features_loaded)

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(input_X_seq, dtype=torch.float32).to(device)

            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
            
            _, predicted_class_idx = torch.max(output.data, 1)
            
            # Decode the predicted class index back to the original label
            predicted_labels = le.inverse_transform(predicted_class_idx.cpu().numpy())

            return jsonify({'prediction': predicted_labels.tolist()})

        except Exception as e:
            return jsonify({'error': str(e)}), 500


# Save the Flask app code to a file
with open('app.py', 'w') as f:
    f.write("""import torch
import torch.nn as nn
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Define the LSTM model class (must be defined to load the state_dict)
class LSTMAnomalyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAnomalyClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the scaler and label encoder
scaler = joblib.load('scaler.joblib')
le = joblib.load('label_encoder.joblib')

# Get num_features and num_classes from the loaded components
num_features_loaded = scaler.n_features_in_ # Get from scaler
num_classes_loaded = len(le.classes_)

# Instantiate the model with the correct parameters
model = LSTMAnomalyClassifier(input_size=num_features_loaded, hidden_size=64, num_layers=2, num_classes=num_classes_loaded).to(device)

# Load the model's state dictionary
model.load_state_dict(torch.load('lstm_anomaly_model.pth', map_location=device))
model.eval() # Set the model to evaluation mode

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) # Get data posted as JSON
            input_df = pd.DataFrame(data)
            
            if 'Timestamp' in input_df.columns:
                input_df = input_df.drop(columns=['Timestamp'])
                
            input_df_encoded = pd.get_dummies(input_df, columns=['Sensor_ID', 'Location'], drop_first=True)
            
            # Load X_encoded columns to ensure consistent feature order and presence
            # In a production setting, this list of columns would be saved along with the model.
            # For this example, we'll assume X_encoded columns are globally available or hardcoded if needed.
            # For robust production, save `X_encoded.columns.tolist()` during training.
            
            # Re-creating a dummy X_encoded.columns.tolist() for demonstration
            # In a real scenario, this list would be loaded from a saved file.
            # For now, let's derive it from the current X_encoded which was run in the notebook
            global X_encoded # Access the X_encoded from the notebook's global scope
            training_columns = X_encoded.columns.tolist() 

            input_df_encoded = input_df_encoded.reindex(columns=training_columns, fill_value=0)

            input_X = input_df_encoded.values
            input_X_scaled = scaler.transform(input_X)
            
            input_X_seq = input_X_scaled.reshape(input_X_scaled.shape[0], 1, num_features_loaded)

            input_tensor = torch.tensor(input_X_seq, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_tensor)
            
            _, predicted_class_idx = torch.max(output.data, 1)
            
            predicted_labels = le.inverse_transform(predicted_class_idx.cpu().numpy())

            return jsonify({'prediction': predicted_labels.tolist()})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
""")

print("Flask API code saved to app.py")
