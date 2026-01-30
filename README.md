# MedMamba-Based Medical Prediction Framework
This project constructs the UserWME model, implementing a WeMap evaluation model that integrates VMamba feature mapping with KAN (Kolmogorov–Arnold Networks), and provides a complete pipeline for data preprocessing, model training, and inference.

---

## 🛠️ Environment Setup
1.  Clone the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
## ⚙️ Train model
```bash
python 01train_Map_S_VMamba.py
```
```bash
python 01predict_Map_S_VMamba.py
```
```bash
python 02nomalize.py
```
```bash
python 03train_Map_KAN_VMamba.py
```
```bash
python 03predict_Map_KAN_VMamba.py
```
## Data
