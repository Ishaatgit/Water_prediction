
import streamlit as st
import pandas as pd, numpy as np, pickle, io, os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

st.set_page_config(layout="wide", page_title="Water Potability Dashboard")

st.title("💧 Water Potability Prediction Dashboard ")


# -- Utility functions
def load_model(path='/mnt/data/model.pkl'):
    if not os.path.exists(path):
        return None, None
    with open(path,'rb') as f:
        data = pickle.load(f)
    # support both direct estimator or dict with keys
    if isinstance(data, dict) and 'model' in data:
        return data['model'], data.get('feature_names', None)
    else:
        return data, None

model, feature_names = load_model()

st.sidebar.header("Model & Data")
uploaded_model = st.sidebar.file_uploader("Upload a model.pkl to replace the demo model", type=['pkl','pickle'])
uploaded_csv = st.sidebar.file_uploader("Or upload your water_potability CSV to retrain", type=['csv'])

if uploaded_model is not None:
    try:
        obj = pickle.load(uploaded_model)
        if isinstance(obj, dict) and 'model' in obj:
            model = obj['model']
            feature_names = obj.get('feature_names', None)
        else:
            model = obj
        st.sidebar.success("Model uploaded and loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV uploaded. Retraining on uploaded data (quick retrain)...")
        # Basic attempt to retrain if it contains expected columns
        common_cols = [c for c in df.columns if c.lower() in {n.lower() for n in ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']}]
        if len(common_cols) >= 6:
            # quick retrain using these columns and first found numeric target 'Potability' if exists
            X = df[common_cols].select_dtypes(include=[int,float]).dropna()
            if 'Potability' in df.columns:
                y = df.loc[X.index, 'Potability']
            elif 'potability' in df.columns:
                y = df.loc[X.index, 'potability']
            else:
                st.sidebar.warning("Uploaded CSV doesn't have 'Potability' target column. Cannot retrain without target.")
                y = None
            if y is not None:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
                pipeline.fit(X, y)
                model = pipeline
                feature_names = list(X.columns)
                st.sidebar.success("Retrain completed (quick). Model replaced in-app (not saved to disk).")
        else:
            st.sidebar.error("Uploaded CSV doesn't contain expected feature columns. Please upload the original dataset or a similar one.")

    except Exception as e:
        st.sidebar.error(f"CSV load/retrain failed: {e}")

if model is None:
    st.error("No model available. Please upload a model.pkl file.")
    st.stop()

# If feature names unknown, try to infer from model if possible
if feature_names is None:
    try:
        if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf'), 'feature_names_in_'):
            feature_names = list(model.named_steps['clf'].feature_names_in_)
    except:
        feature_names = None

st.sidebar.subheader("Input Controls")
st.sidebar.markdown("Adjust feature values for prediction. Default values are typical ranges.")

# Default feature list if unknown
default_features = {
    'ph':7.0,'Hardness':200,'Solids':20000,'Chloramines':6,'Sulfate':300,'Conductivity':400,'Organic_carbon':8,'Trihalomethanes':60,'Turbidity':3
}

fnames = feature_names if feature_names is not None else list(default_features.keys())

user_input = {}
cols = st.columns(3)
for i, fname in enumerate(fnames):
    # choose appropriate widget based on typical ranges
    if fname.lower()=='ph':
        val = cols[i%3].slider(fname, 0.0, 14.0, float(default_features.get(fname,7.0)))
    elif fname.lower() in ['hardness','sulfate','conductivity']:
        val = cols[i%3].slider(fname, 0.0, 2000.0, float(default_features.get(fname,200)))
    elif fname.lower() in ['solids']:
        val = cols[i%3].slider(fname, 0.0, 100000.0, float(default_features.get(fname,20000)))
    elif fname.lower() in ['trihalomethanes']:
        val = cols[i%3].slider(fname, 0.0, 500.0, float(default_features.get(fname,60)))
    elif fname.lower() in ['organic_carbon','organic carbon']:
        val = cols[i%3].slider(fname, 0.0, 100.0, float(default_features.get(fname,8)))
    else:
        val = cols[i%3].slider(fname, 0.0, 1000.0, float(default_features.get(fname,0)))
    user_input[fname] = val

input_df = pd.DataFrame([user_input], columns=fnames)

st.subheader("Prediction & Explanation")
col1, col2 = st.columns([2,1])

with col1:
    st.write("Input values")
    st.dataframe(input_df.T, use_container_width=True)
    # Model prediction
    try:
        preds = model.predict(input_df)
        probs = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_df)
        st.markdown("""**Prediction:**""")
        st.write(preds[0])
        if probs is not None:
            st.markdown("""**Prediction probabilities:**""")
            st.write(dict(zip(['Not Potable','Potable'], probs[0])))
    except Exception as e:
        st.error(f"Prediction failed: {e}")


with col2:
    st.markdown("**Feature importance / coefficients (approx)**")
    try:
        import numpy as np
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=fnames)
            st.bar_chart(importance)
        elif hasattr(model, "coef_"):
            coefs = np.ravel(model.coef_)
            coef_df = pd.DataFrame({"feature": fnames, "coef": coefs}).sort_values("coef", key=abs, ascending=False)
            st.bar_chart(coef_df.set_index("feature")["coef"])
        else:
            st.info("Feature importance not available for this model.")
    except Exception as e:
        st.error(f"Unable to display feature importance: {e}")

st.markdown('---')
st.subheader('Quick Visuals from Input Context')

vcol1, vcol2 = st.columns(2)
with vcol1:
    st.write('Distribution context (synthetic demo)')
    df_demo = pd.DataFrame({k: [v] for k, v in default_features.items()})
    st.bar_chart(df_demo.T)

with vcol2:
    st.write('Prediction probability gauge (demo)')
    try:
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(input_df)[0, 1]
            st.metric('Probability Potable', f"{p*100:.1f}%", delta=None)
        else:
            st.info('Model has no probability output.')
    except Exception as e:
        st.error(f"Failed to compute probability: {e}")

st.markdown('---')
st.caption("This app is a demonstrative dashboard. Replace model.pkl with your trained model.pkl from ML_PROJECT.ipynb for real predictions.")

