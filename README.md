## Overview
This project predicts fraudulent transactions using a neural network built with TensorFlow. The dataset contains transaction types, amounts, and balances to classify transactions as fraudulent or not.
## Dataset
Features: Type, amount, old balance, and new balance.
Target: Fraudulent transaction indicator (isFraud).
## Workflow
### Data Preprocessing:
Mapping categorical features to numerical values.
Splitting the data into training and test sets.
Standardizing features.
### Model:
Neural network with multiple layers and ReLU activation.
Output layer with sigmoid activation for binary classification.
### Training and Evaluation:
Training for 5 epochs with a batch size of 512.
Evaluating accuracy and identifying anomalies.
## Results
Accuracy: (Add after running)

Anomalies Detected: (Add after running)

## How to Run

Install dependencies:
bash


pip install -r requirements.txt

Run the script:
bash


python fraud_detection_nn.py

## Dependencies

numpy

pandas

scikit-learn

tensorflow
