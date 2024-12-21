
# Transaction Fraud Detection Using Neural Networks

## Overview
This project predicts fraudulent transactions using a neural network built with TensorFlow. The dataset contains transaction types, amounts, and balances to classify transactions as fraudulent or not.

## Dataset
- **Features**: Type, amount, old balance, and new balance.
- **Target**: Fraudulent transaction indicator (`isFraud`).

## Workflow
1. **Data Preprocessing**:
   - Mapping categorical features to numerical values.
   - Splitting the data into training and test sets.
   - Standardizing features.

2. **Model**:
   - Neural network with multiple layers and ReLU activation.
   - Output layer with sigmoid activation for binary classification.

3. **Training and Evaluation**:
   - Training for 5 epochs with batch size 512.
   - Evaluating accuracy and identifying anomalies.

## Results
- **Accuracy**: (Add after running)
- **Anomalies Detected**: (Add after running)

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python fraud_detection_nn.py
   ```

## Dependencies
- numpy
- pandas
- scikit-learn
- tensorflow
#   T r a n s a c t i o n _ F r a u d _ D e t e c t i o n 
 
 
