# Water_prediction
# 💧 Water Potability Prediction Dashboard

### 🎯 **Project Overview**
This project is an interactive **Streamlit web dashboard** designed to predict **whether water is potable (safe for drinking)** based on its physicochemical properties such as pH, hardness, solids, chloramines, sulfate, and more.

The prediction model is trained using **machine learning algorithms** developed in the Jupyter Notebook `ML_PROJECT.ipynb`, and the trained model is stored in `model.pkl`.

---

## 🧠 **Key Features**
- ✅ Predicts **whether water is potable or not** in real time.
- 📊 **Interactive dashboard** built using **Streamlit**.
- 📈 **Visual representations** of input water quality parameters.
- 🧮 Integrated **Machine Learning model (.pkl)** trained with Scikit-learn.
- ⚙️ Easy to run locally and extend for deployment.

---

## 🏗️ **Tech Stack**
| Component | Technology |
|------------|------------|
| **Frontend/UI** | Streamlit |
| **Backend / ML Model** | Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Streamlit Charts |
| **Language** | Python 3.8+ |

---

## 📂 **Project Structure**
```
📦 Water-Potability-Dashboard
├── dashboard.py          # Streamlit app file
├── model.pkl             # Trained machine learning model
├── ML_PROJECT.ipynb      # Notebook for model training and analysis
├── Report.pdf            # Project report and documentation
└── README.md             # Project details and setup guide
```

---

## ⚙️ **Setup Instructions**

### 1️⃣ Clone or Download the Project
```bash
git clone https://github.com/<Ishaagit>/water-potability-.git
cd dashboard.py
```

### 2️⃣ Install Dependencies
Make sure Python and pip are installed, then run:
```bash
pip install streamlit scikit-learn pandas numpy matplotlib
```

### 3️⃣ Run the Dashboard
```bash
streamlit run dashboard.py
```

✅ Streamlit will launch automatically in your browser at:
```
http://localhost:8501
```

---

## 💡 **How It Works**
1. The dashboard takes user input for:
   - pH  
   - Hardness  
   - Solids  
   - Chloramines  
   - Sulfate  
   - Conductivity  
   - Organic Carbon  
   - Trihalomethanes  
   - Turbidity  
2. These values are passed to the trained ML model (`model.pkl`).
3. The model predicts:
   - **1 → Potable (Safe for Drinking)**  
   - **0 → Not Potable (Unsafe for Drinking)**  
4. The dashboard then displays:
   - Prediction result (✅ Safe / ❌ Not Safe)
   - Probability score (if available)
   - Visualization of parameter distribution.

---

## 📊 **Model Development**
The machine learning model was built and evaluated in `ML_PROJECT.ipynb` using:
- **Algorithms:** Logistic Regression, Random Forest, Decision Tree
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
- **Steps:**
  1. Data cleaning and preprocessing
  2. Feature scaling using StandardScaler
  3. Splitting data into train-test sets
  4. Training and saving the best-performing model:
     ```python
     import pickle
     pickle.dump(model, open('model.pkl', 'wb'))
     ```

---

## 📈 **Sample Output**
| Parameter | Value |
|------------|--------|
| pH | 7.2 |
| Hardness | 180 mg/L |
| Solids | 40000 ppm |
| Chloramines | 6.5 ppm |
| Sulfate | 330 ppm |
| Conductivity | 450 µS/cm |
| Organic Carbon | 10 ppm |
| Trihalomethanes | 80 µg/L |
| Turbidity | 3 NTU |

➡️ **Prediction:** ✅ *Water is Potable (Safe for Drinking)*

---

## 🧩 **Future Enhancements**
- Integration of **real-time sensor data** for live monitoring.
- Use of **deep learning models** for higher accuracy.
- Deployment on **Streamlit Cloud / Render / Heroku**.
- Addition of water quality comparison charts by location.

---



