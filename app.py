# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import os
import time

st.set_page_config(page_title="Smart Fraud Engine UI", layout="wide")

@st.cache_resource
def load_assets():
   
    base_dir = os.getcwd()
    paths_to_check = [
        os.path.join(base_dir, "models", "fraud_pipeline.joblib"),
        os.path.join(base_dir, "fraud_pipeline.joblib"),
        "models/fraud_pipeline.joblib",
        "fraud_pipeline.joblib"
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            return joblib.load(path)
    return None

pipeline = load_assets()

if pipeline is None:
    st.error("⚠️ Missing compiled model. Please run `python train.py` first.")
else:
    st.title("🛡️ Credit Card Fraud Monitoring & Investigation Deck")
    st.caption("Real-Time Behavioral Analysis & Explainable Compliance Auditing")
    
    @st.cache_data
    def get_dashboard_data():
        if os.path.exists("credit_card_fraud_10k.csv"):
            return pd.read_csv("credit_card_fraud_10k.csv")
        return None

    raw_data = get_dashboard_data()
    
    if raw_data is None:
        st.warning("Please run `python train.py` to generate the dataset.")
    else:
        
        drop_cols = [c for c in ['transaction_id', 'is_fraud'] if c in raw_data.columns]
        X_raw = raw_data.drop(columns=drop_cols)
        X_trans = pipeline['preprocessor'].transform(X_raw)
        
        raw_probs = pipeline['model'].predict_proba(X_trans)[:, 1]
        cal_probs = pipeline['calibrator'].transform(raw_probs)
        risk_scores = np.round(cal_probs * 100, 1)
        
        view_df = raw_data.copy()
        view_df['Risk Score'] = risk_scores
        
        tab1, tab2, tab3 = st.tabs([
            "Live Alerts & Deep Dive", 
            "Interactive Fraud Explorer", 
            "Streaming Simulator"
        ])
        
        with tab1:
            st.sidebar.header("Alert Settings")
            
            # dynamically calculate the 90th percentile to ensure the queue is never empty on load
            smart_default = max(5, int(view_df['Risk Score'].quantile(0.90)))
            alert_level = st.sidebar.slider("Minimum Risk Level to Trigger Alert", 0, 100, smart_default)
            
            alert_queue = view_df[view_df['Risk Score'] >= alert_level].sort_values(by='Risk Score', ascending=False)
            
            st.subheader(f"⚠️ Live Suspected Transaction Queue ({len(alert_queue)} active matches)")
            st.dataframe(alert_queue[['transaction_id', 'amount', 'merchant_category', 'Risk Score']], use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Deep Dive & Counterfactual Modeler")
            
            if not alert_queue.empty:
                selected_id = st.selectbox("Select Transaction to Audit:", alert_queue['transaction_id'].unique())
                
                if selected_id:
                    tx_row = raw_data[raw_data['transaction_id'] == selected_id].iloc[0]
                    tx_features = pd.DataFrame([tx_row]).drop(columns=drop_cols)
                    tx_proc = pipeline['preprocessor'].transform(tx_features)
                    score_metric = view_df[view_df['transaction_id'] == selected_id]['Risk Score'].values[0]
                    
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.metric("Continuous Risk Score", f"{score_metric} / 100", 
                                  delta="HIGH THREAT" if score_metric > 75 else "REVIEW", delta_color="inverse")
                        st.write("**Transaction Metadata:**")
                        st.dataframe(tx_features.T.rename(columns={0: 'Value'}), use_container_width=True)
                        
                    with c2:
                        explainer = shap.TreeExplainer(pipeline['model'])
                        raw_shap = explainer.shap_values(tx_proc)
                        
                        if isinstance(raw_shap, list):
                            val_array = raw_shap[1][0]
                        elif len(raw_shap.shape) == 3:
                            val_array = raw_shap[0, :, 1]
                        else:
                            val_array = raw_shap[0]
                            
                        shap_summary = pd.DataFrame({
                            'Attribute': pipeline['feature_names'],
                            'Impact Weight': val_array
                        }).sort_values(by='Impact Weight', key=abs, ascending=False).head(5)
                        
                        fig = px.bar(shap_summary, x='Impact Weight', y='Attribute', orientation='h',
                                     title="What drove this score? (SHAP Values)",
                                     color='Impact Weight', color_continuous_scale=px.colors.sequential.Reds)
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### 🔄 Live Counterfactual Engine")
                    st.caption("Drag the sliders to instantly simulate how different behaviors impact the final risk score.")
                    
                    # 3-column layout to make the live updates obvious
                    col_cf1, col_cf2, col_cf3 = st.columns([1, 1, 1])
                    
                    with col_cf1:
                        cf_amount = st.slider("Test Amount Change:", 0.0, float(raw_data['amount'].max()), float(tx_row['amount']), step=5.0)
                    with col_cf2:
                        cf_vel = st.slider("Test Velocity Change:", 0, int(raw_data['velocity_last_24h'].max()), int(tx_row['velocity_last_24h']), step=1)
                    
                    
                    cf_test = tx_features.copy()
                    cf_test['amount'] = cf_amount
                    cf_test['velocity_last_24h'] = cf_vel
                    cf_proc = pipeline['preprocessor'].transform(cf_test)
                    cf_score = np.round(pipeline['calibrator'].transform(pipeline['model'].predict_proba(cf_proc)[:, 1])[0] * 100, 1)
                    
                    with col_cf3:
                        st.metric("Live Simulated Score", f"{cf_score} / 100", f"{cf_score - score_metric:.1f} variance")

        with tab2:
            st.subheader("📊 Interactive Fraud Explorer")
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                merch_filter = st.multiselect("Filter by Merchant", view_df['merchant_category'].unique())
            with c_f2:
                risk_band = st.slider("Risk Band", 0, 100, (0, 100))
                
            filtered_df = view_df[view_df['Risk Score'].between(risk_band[0], risk_band[1])]
            if merch_filter:
                filtered_df = filtered_df[filtered_df['merchant_category'].isin(merch_filter)]
                
            fig_scatter = px.scatter(filtered_df, x='transaction_hour', y='amount', color='Risk Score', 
                                     size='velocity_last_24h', hover_data=['transaction_id', 'merchant_category'])
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.subheader("🚀 Live Streaming Inference Simulation")
            if st.button("Start Live Kafka Stream Simulation"):
                stream_placeholder = st.empty()
                stream_data = view_df.sample(20).reset_index(drop=True)
                acc_data = pd.DataFrame()
                
                for _, row in stream_data.iterrows():
                    acc_data = pd.concat([acc_data, pd.DataFrame([row])])
                    with stream_placeholder.container():
                        st.write(f"**Ingesting at {time.strftime('%H:%M:%S')}...**")
                        m_col, d_col = st.columns([1, 4])
                        risk = row['Risk Score']
                        color = "red" if risk > 75 else ("orange" if risk > 50 else "green")
                        with m_col:
                            st.markdown(f"<h2 style='color:{color}'>{risk}</h2>", unsafe_allow_html=True)
                        with d_col:
                            st.dataframe(acc_data.tail(4)[['transaction_id', 'amount', 'merchant_category', 'Risk Score']], hide_index=True)
                    time.sleep(0.5)