from flask import Flask, jsonify, request
import tensorflow as tf
import pickle
import os
import numpy as np

INTERVAL_NEURAL_NETWORK = 'interval_neural_network.keras'
INTERVAL_INPUT_SCALER = 'input_scaler_interval.pkl'
INTERVAL_OUTPUT_SCALER = 'output_scaler_interval.pkl'
CONTINUOUS_NEURAL_NETWORK = 'neural_network.keras'
CONTINUOUS_INPUT_SCALER = 'input_scaler.pkl'
CONTINUOUS_OUTPUT_SCALER = 'output_scaler.pkl'
DECISION_TREE = 'decision_tree.pkl'
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route('/trainingSession', methods=['GET'])
def generateTrainigSession():
    ctl = request.args.get('ctl', type=float)
    remaining_days = request.args.get('remaining_days', type=int)
    distance = request.args.get('distance', type=int)
    time = request.args.get('time', type=int)

    data = {
        'fatigue': ctl,
        'goal_distance': distance,
        'goal_time': time,
        'remaining_days': remaining_days
    }

    session_type = predict_decision_tree_values(data, DECISION_TREE)
    if session_type == 'Interval':
        output = predict_neural_network_values(data, INTERVAL_NEURAL_NETWORK, INTERVAL_INPUT_SCALER, INTERVAL_OUTPUT_SCALER)
        result = process_interval_output(output)
    else:
        output = predict_neural_network_values(data, CONTINUOUS_NEURAL_NETWORK, CONTINUOUS_INPUT_SCALER, CONTINUOUS_OUTPUT_SCALER)
        result = process_continuous_output(output)

    return jsonify(result)

def predict_decision_tree_values(data: dict, file_name: str) -> str:
    path = os.path.join(FILE_PATH, file_name)
    with open(path, 'rb') as f:
        model = pickle.load(f)

    data_array = np.array([list(data.values())])
    output = model.predict(data_array)
    return output[0]

def normalize_data(data, file_name: str):
    path = os.path.join(FILE_PATH, file_name)
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    scaler.fit(data)
    normalized_data = scaler.transform(data)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    return normalized_data

def denormalize_data(data, file_name: str):
    path = os.path.join(FILE_PATH, file_name)
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler.inverse_transform(data)

def predict_neural_network_values(
        data: dict,
        file_name: str,
        input_scaler: str,
        output_scaler: str
    ):

    path = os.path.join(FILE_PATH, file_name)
    model = tf.keras.models.load_model(path)
    data_array = np.array([list(data.values())])
    data_array = normalize_data(data_array, input_scaler)
    output = model.predict(data_array)
    output = denormalize_data(output, output_scaler)
    return output[0]

def process_interval_output(output):
    distance = output[0]
    pace = output[2]
    time = ((distance / 1000) * pace) * 60
    return {'distance': int(distance), 'seconds': int(time), 'times': int(output[1]), 'hr': int(output[3])}

def process_continuous_output(output):
    distance = output[0]
    pace = output[1]
    time = ((distance / 1000) * pace) * 60
    return {'distance': int(distance), 'seconds': int(time), 'times': 0, 'hr': int(output[2])}

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 4001)))