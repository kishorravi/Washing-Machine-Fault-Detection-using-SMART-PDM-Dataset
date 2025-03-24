# 🧠 Washing Machine Fault Detection using SMART-PDM Dataset

This repository presents a complete machine learning pipeline for **automatic fault detection** in washing machines. Leveraging real-world data from the **SMART-PDM dataset**, we aim to distinguish between normal operations and faulty behavior using sensor data (power and vibration).

---

## 📌 Project Goal

To build a machine learning model capable of:
- Detecting whether a washing machine is operating normally (Working) or experiencing a fault (Failure)
- Supporting predictive maintenance in smart home appliances
- Demonstrating real-world utility of sensor fusion and ML in appliance monitoring

---

## 🗂 Dataset Description

### ✅ Origin
The **SMART-PDM** dataset was collected from washing machines in a **repair center located in Lisbon, Portugal**, and stored at a facility in **Coimbra, Portugal**. It was developed under the **SMART Predictive Device Maintenance (SMART-PDM)** research project.

### ✅ Files Included
- `stream_labels.csv`: Descriptive labels and metadata for each washing cycle
- `<device_id>_<timestamp_begin>_<timestamp_end>_slow.csv`: Slow stream data (1 Hz)
- `<device_id>_<timestamp_begin>_<timestamp_end>_fast.csv`: Fast stream data (2048 Hz)

### ✅ Slow Stream (`*_slow.csv`)
- 1 sample per second
- Contains **Active Power (ActP)** data

### ✅ Fast Stream (`*_fast.csv`)
- 2048 samples per second
- Captures **current and vibration** sensor data

---

## 📋 Labeling Structure

In `stream_labels.csv`, each row represents one complete washing cycle. The key columns include:

| Column             | Description                                        |
|--------------------|----------------------------------------------------|
| `timestamp_begin`  | Start of the washing cycle (EPOCH)                |
| `timestamp_end`    | End of the washing cycle                          |
| `device_id`        | Sensor/machine ID                                 |
| `brand`            | Brand of the machine                              |
| `model`            | Model of the machine                              |
| `program`          | Washing program selected                          |
| `temperature`      | Washing temperature setting (°C)                  |
| `spin`             | Spin speed (RPM)                                  |
| `load`             | Load condition (e.g., Empty, Full)                |
| `failure`          | `Working` or known fault (e.g., Heating, Motor)   |
| `observations`     | Additional notes                                  |

### 🎯 Binary Classification Setup

We simplify the `failure` field into two categories:

| Value        | Label |
|--------------|-------|
| Working      | 0     |
| Any failure  | 1     |

---

## 🧠 Machine Learning Pipeline

### ✅ Preprocessing
- Merged `stream_labels.csv` with each sensor file using `device_id`, `timestamp_begin`, and `timestamp_end`
- Dropped timestamp and non-numeric features
- Encoded `Working` as 0 and all other `failure` types as 1

### ✅ Model Used
- **RandomForestClassifier** from Scikit-learn
  - `n_estimators=100`
  - `random_state=42`
- Chosen for:
  - Strong performance on tabular data
  - Robustness to noise and feature scaling
  - Easy interpretability

---

## 📈 Model Results

### ✅ Classification Report

| Metric        | Normal (0) | Fault (1) | Overall |
|---------------|------------|-----------|---------|
| Precision     | 1.00       | 1.00      | 1.00    |
| Recall        | 1.00       | 1.00      | 1.00    |
| F1-Score      | 1.00       | 1.00      | 1.00    |
| Support       | 84,655     | 18,982    | 103,637 |




