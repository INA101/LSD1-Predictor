# 🧬 LSD1 Inhibitor Activity Predictor
A machine learning web application that predicts compound inhibitory potency (pIC50) against Lysine-Specific Histone Demethylase 1 (LSD1).

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📁 Project Files

- `app.py` - Streamlit web application
- `LGBMRegressor.pkl` - Trained machine learning model  
- `LSD1.ipynb` - Jupyter notebook with model development
- `requirements.txt` - Python dependencies

## 💡 Usage

### Single Compound Prediction
1. Enter SMILES strings in the text area
2. Click "Predict Activity" 
3. View pIC50 predictions and download results

### Batch Prediction
1. Upload CSV with 'SMILES' and 'Compound_ID' columns
2. Process multiple compounds at once
3. Download predictions as CSV

## 🎯 About

This app predicts binding affinity of chemical compounds against LSD1 (KDM1A), an important epigenetic target in cancer research and drug discovery.

**Built with:** Python, Streamlit, RDKit, LightGBM
