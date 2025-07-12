# 💓 Blood Pressure Estimation Using CNN-BiLSTM Hybrid Model

This project predicts Systolic (SBP) and Diastolic (DBP) blood pressure from Photoplethysmography (PPG) signals using a hybrid deep learning model combining CNN, BiLSTM, and scalogram-based features.  

It includes a **Streamlit Web App** for easy user interaction: upload PPG+ABP data, visualize preprocessing, generate scalograms, and see BP predictions in mmHg.

---

## 🚀 Features

✅ Upload CSV with PPG+ABP signals  
✅ Bandpass filtering, wavelet denoising, baseline removal  
✅ Signal segmentation  
✅ Scalogram generation from PPG segments  
✅ Scalogram and signal quality checks  
✅ CNN-BiLSTM model prediction  
✅ SBP/DBP prediction in mmHg  
✅ BP category interpretation (Normal, Elevated, Hypertension Stages)  
✅ Temporary session-based data storage  

---

## 📂 Project Structure

.
├── app/
│ └── streamlit_app.py # Streamlit frontend
├── scripts/
│ ├── preprocess_utils.py # Signal preprocessing
│ ├── filter_utils.py # Quality filters for signals
│ ├── segment_utils.py # Segmentation logic
│ ├── scalogram_utils.py # Scalogram generation
│ ├── scalogram_filter_utils.py # Scalogram quality checks
│ ├── model_utils.py # Model/scaler load
│ └── predict_utils.py # Model inference
├── models/
│ ├── best_hybrid_model.h5 # Trained CNN-BiLSTM model
│ └── label_scaler.save # Label scaler for inverse_transform
├── user_data_upload/ # Session uploads (temp)
└── README.md



---

## 🧪 How It Works

1️⃣ **Upload**: User provides a CSV with **PPG** and **ABP** columns.  
2️⃣ **Preprocessing**: Bandpass filtering, wavelet denoising, baseline removal.  
3️⃣ **Quality Check**: Filters noisy or too-short signals.  
4️⃣ **Segmentation**: Extracts usable windows from PPG/ABP.  
5️⃣ **Scalogram**: Generates time-frequency images.  
6️⃣ **Model Prediction**: CNN-BiLSTM uses scalograms + PPG to predict SBP/DBP.  
7️⃣ **Output**: SBP/DBP in mmHg with hypertension stage interpretation.

---

## ⚙️ Setup & Installation

> Requirements
- Python 3.8+
- Virtual environment recommended

### Install dependencies:

```bash
pip install -r requirements.txt
```
Example requirements.txt:

streamlit
numpy
pandas
opencv-python
scikit-learn
tensorflow
Pillow

##▶️ Running the Streamlit App

```
cd app
streamlit run streamlit_app.py
```
📈 Model Details
Hybrid architecture: CNN (for scalogram images) + BiLSTM (for PPG signals)

Regression output: SBP and DBP in mmHg

Trained on preprocessed PPG-ABP datasets

Scaler saved for inverse-transform of model predictions

🩸 BP Category Rules
Normal: SBP < 120 and DBP < 80

Elevated (Prehypertension): SBP 120–129 and DBP < 80

Hypertension Stage 1: SBP 130–139 or DBP 80–89

Hypertension Stage 2: SBP ≥ 140 or DBP ≥ 90

⚠️ For medical/clinical use, this logic can be customized or refined.

🗄️ Data Privacy
✅ User uploads are stored in unique session folders
✅ No long-term data retention
✅ Temporary folders can be cleared manually

💻 Example CSV Format
PPG	ABP
0.823	110.5
0.826	111.0
...	...

CSV must include PPG and ABP columns.

🤝 Contributing
PRs and issues welcome! Please:

Keep functions pure and modular

Follow existing code style

Include clear docstrings

📜 License
MIT License © 2025 BhavyaShukla

❤️ Acknowledgments
Streamlit for UI

TensorFlow/Keras for model training

OpenCV, SciPy, NumPy for signal/image processing
