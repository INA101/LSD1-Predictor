import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle 

# Page configuration
st.set_page_config(
    page_title="LSD1 Inhibitor Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def compute_descriptors(smiles):
    embeddings = []
    if type(smiles) == list:
        for desc in smiles:
            molecule = Chem.MolFromSmiles(desc)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
            fingerprint_array = np.array(fingerprint)
            embeddings.append(fingerprint_array)
        df = pd.DataFrame(embeddings)
    
    elif type(smiles) == str:
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
        fingerprint_array = np.array(fingerprint)
        df = pd.DataFrame([fingerprint_array])
    return df

# Load model
model = load_model('LGBMRegressor.pkl')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #2e86ab;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown('<div class="main-header">🧬 LSD1 Inhibitor Activity Predictor</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sub-header">
Machine learning prediction of compound inhibitory potency (pIC50) against epigenetic target LSD1 (KDM1A)
</div>
""", unsafe_allow_html=True)

# Sidebar for additional information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts the **pIC50** value of chemical compounds against **Lysine-Specific Histone Demethylase 1 (LSD1)**.
    
    **pIC50** represents the negative logarithm of the half-maximal inhibitory concentration, where higher values indicate stronger binding affinity.
    
    **How to use:**
    - **Single Prediction**: Enter SMILES strings
    - **Batch Prediction**: Upload a CSV file with SMILES and Compound_ID columns
    """)
    
    st.header("📊 Model Info")
    st.write("Model: LightGBM Regressor")
    st.write("Descriptors: Morgan Fingerprints (radius=2, nBits=2048)")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Prediction Method")
    prediction_method = st.radio(
        "Choose prediction method:",
        ["Predict from SMILES", "Submit a CSV"],
        help="Select how you want to input compounds for prediction"
    )

with col2:
    if prediction_method == "Predict from SMILES":
        st.subheader("🔬 Enter SMILES Strings")
        smiles_input = st.text_area(
            "Enter one or more SMILES strings (one per line):",
            height=150,
            placeholder="CCO - Ethanol\nCC(=O)O - Acetic acid\n...",
            help="Enter SMILES notation for chemical compounds"
        )
        
        if st.button("🚀 Predict Activity", type="primary", use_container_width=True):
            if smiles_input.strip():
                with st.spinner("Computing descriptors and making predictions..."):
                    smiles_list = [s.strip() for s in smiles_input.splitlines() if s.strip()]
                    valid_smiles = []
                    invalid_smiles = []
                    
                    # Validate SMILES
                    for smile in smiles_list:
                        mol = Chem.MolFromSmiles(smile)
                        if mol is not None:
                            valid_smiles.append(smile)
                        else:
                            invalid_smiles.append(smile)
                    
                    if valid_smiles:
                        descriptors_df = compute_descriptors(valid_smiles)
                        predictions = model.predict(descriptors_df)
                        
                        # Create results with better formatting
                        results_df = pd.DataFrame({
                            'SMILES': valid_smiles,
                            'Predicted_pIC50': predictions,
                            'Activity_Level': ['High' if x > 6 else 'Medium' if x > 5 else 'Low' for x in predictions]
                        })
                        
                        st.markdown('<div class="success-box">✅ Predictions completed successfully!</div>', unsafe_allow_html=True)
                        
                        # Display statistics
                        avg_pic50 = results_df['Predicted_pIC50'].mean()
                        st.metric("Average Predicted pIC50", f"{avg_pic50:.2f}")
                        
                        # Display results in an attractive way
                        st.subheader("📈 Prediction Results")
                        
                        # Use tabs for different views
                        tab1, tab2 = st.tabs(["📊 Table View", "📋 Summary"])
                        
                        with tab1:
                            # Style the dataframe
                            styled_df = results_df.style.background_gradient(
                                subset=['Predicted_pIC50'], 
                                cmap='RdYlGn_r'  # Red-Yellow-Green (reversed for higher=better)
                            )
                            st.dataframe(styled_df, use_container_width=True)
                        
                        with tab2:
                            for idx, row in results_df.iterrows():
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <strong>Compound {idx+1}:</strong> {row['SMILES']}<br>
                                    <strong>pIC50:</strong> {row['Predicted_pIC50']:.2f}<br>
                                    <strong>Activity:</strong> {row['Activity_Level']}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name='lsd1_predictions.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    if invalid_smiles:
                        st.warning(f"⚠️ {len(invalid_smiles)} invalid SMILES ignored: {', '.join(invalid_smiles)}")
            else:
                st.error("❌ Please enter at least one valid SMILES string.")

    else:  # CSV Upload
        st.subheader("📁 Upload CSV File")
        st.write("Upload a CSV file containing 'SMILES' and 'Compound_ID' columns.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="CSV must contain 'SMILES' and 'Compound_ID' columns"
        )
        
        if uploaded_file is not None:
            # Preview the uploaded file
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(data)} compounds.")
            
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            
            if st.button("🚀 Predict Activity for All Compounds", type="primary", use_container_width=True):
                with st.spinner("Processing compounds and making predictions..."):
                    if 'SMILES' not in data.columns or 'Compound_ID' not in data.columns:
                        st.error("❌ CSV must contain 'SMILES' and 'Compound_ID' columns.")
                    else:
                        smiles_list = data['SMILES'].tolist()
                        compound_ids = data['Compound_ID'].tolist()
                        
                        valid_data = []
                        invalid_data = []
                        
                        # Validate SMILES
                        for i, smile in enumerate(smiles_list):
                            mol = Chem.MolFromSmiles(str(smile))
                            if mol is not None:
                                valid_data.append((compound_ids[i], smile))
                            else:
                                invalid_data.append((compound_ids[i], smile))
                        
                        if valid_data:
                            valid_compound_ids, valid_smiles = zip(*valid_data)
                            descriptors_df = compute_descriptors(list(valid_smiles))
                            predictions = model.predict(descriptors_df)
                            
                            results_df = pd.DataFrame({
                                'Compound_ID': valid_compound_ids,
                                'SMILES': valid_smiles,
                                'Predicted_pIC50': predictions,
                                'Activity_Level': ['High' if x > 6 else 'Medium' if x > 5 else 'Low' for x in predictions]
                            })
                            
                            st.markdown('<div class="success-box">✅ Batch predictions completed!</div>', unsafe_allow_html=True)
                            
                            # Display batch statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Compounds", len(results_df))
                            with col2:
                                st.metric("Average pIC50", f"{results_df['Predicted_pIC50'].mean():.2f}")
                            with col3:
                                high_act = len(results_df[results_df['Activity_Level'] == 'High'])
                                st.metric("High Activity", high_act)
                            
                            st.subheader("📊 Prediction Results")
                            st.dataframe(results_df.style.background_gradient(
                                subset=['Predicted_pIC50'], 
                                cmap='RdYlGn_r'
                            ), use_container_width=True)
                            
                            # Download button
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download All Predictions",
                                data=csv,
                                file_name='lsd1_batch_predictions.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                        
                        if invalid_data:
                            invalid_df = pd.DataFrame(invalid_data, columns=['Compound_ID', 'SMILES'])
                            st.warning(f"⚠️ {len(invalid_data)} compounds with invalid SMILES were skipped")
                            with st.expander("View invalid compounds"):
                                st.dataframe(invalid_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "LSD1 Inhibitor Prediction App | Built with Streamlit & RDKit"
    "</div>", 
    unsafe_allow_html=True
)