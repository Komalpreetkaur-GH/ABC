import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import ui_config

# Set page configuration
st.set_page_config(page_title="Earnomly", layout="wide", page_icon=None)

# Custom CSS for better styling
ui_config.apply_styles()

# Title and Introduction

st.title("Earnomly")
st.markdown("""
<div class="glass-card">
    <p>This professional dashboard provides advanced socioeconomic analysis and predictive modeling for the <strong>Adult Income Dataset</strong>.</p>
</div>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
               'hours-per-week', 'native-country', 'income']
    try:
        df = pd.read_csv('adultData/adult.data', names=columns, na_values='?', skipinitialspace=True)
        # Impute missing values
        for col in ['workclass', 'occupation', 'native-country']:
            df[col] = df[col].fillna(df[col].mode()[0])
        df.drop_duplicates(inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'adultData/adult.data' not found. Please ensure the dataset is in the correct directory.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Preprocessing Helper ---
@st.cache_data
def prepare_data_for_model(df):
    le = LabelEncoder()
    df_model = df.copy()
    # Encode target
    df_model['income_encoded'] = le.fit_transform(df_model['income'])
    df_model = df_model.drop('income', axis=1)
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    
    X = df_encoded.drop('income_encoded', axis=1)
    y = df_encoded['income_encoded']
    
    return X, y, le, X.columns

@st.cache_resource
def calculate_wcss(X_scaled):
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        # Optimization: Reduced n_init to 3 for faster visual rendering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return wcss, k_range

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand" style='display:flex; align-items:center; padding: 20px; gap:12px; border:none; background:transparent;'>
            <div style='width:40px; height:40px; background:linear-gradient(135deg, #10B981, #3B82F6); border-radius:12px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:1.4rem; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);'>E</div>
            <div>
                <h2 style='margin:0; color:white !important; font-size:1.3rem !important; font-weight:800 !important;'>Earnomly</h2>
                <div style='width:20px; height:2px; background:#10B981; margin-top:2px;'></div>
            </div>
        </div>
        <div style='margin-bottom: 2rem;'></div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p class='nav-category' style='margin-left:20px; opacity:0.5; font-size:0.7rem; letter-spacing:0.1em; color:#9CA3AF;'>MAIN DASHBOARD</p>", unsafe_allow_html=True)
    options = st.radio("Navigation", 
        ["Overview", "Analytics", "Intelligence", "Clusters", "Predictor"],
        label_visibility="collapsed")

# --- 1. Data Overview ---
if "Overview" in options:
    st.markdown("<h1>Dashboard Overview</h1>", unsafe_allow_html=True)
    
    # Custom Metric Cards with Trends
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <p style='margin:0; color:var(--text-secondary); font-size:0.8rem; text-transform:uppercase;'>Total Population</p>
                <span style='color:#10B981; font-size:0.7rem; font-weight:bold;'>+12%</span>
            </div>
            <h2 style='margin:10px 0;'>{df.shape[0]:,}</h2>
            <div style='width:100%; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden;'>
                <div style='width:70%; height:100%; background:#10B981;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <p style='margin:0; color:var(--text-secondary); font-size:0.8rem; text-transform:uppercase;'>Features Analyzed</p>
                <span style='color:#3B82F6; font-size:0.7rem; font-weight:bold;'>STABLE</span>
            </div>
            <h2 style='margin:10px 0;'>{df.shape[1]}</h2>
            <div style='width:100%; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden;'>
                <div style='width:100%; height:100%; background:#3B82F6;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <p style='margin:0; color:var(--text-secondary); font-size:0.8rem; text-transform:uppercase;'>Missing Values</p>
                <span style='color:#8B5CF6; font-size:0.7rem; font-weight:bold;'>CLEAN</span>
            </div>
            <h2 style='margin:10px 0;'>0.0%</h2>
            <div style='width:100%; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden;'>
                <div style='width:100%; height:100%; background:#8B5CF6;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


    with st.container(border=True):
        st.subheader("Sample System Data")
        st.dataframe(df.head(10), use_container_width=True)

    
    with st.container(border=True):
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with st.container(border=True):
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0], use_container_width=True)
    

# --- 2. Visualizations (Plotly) ---
elif "Analytics" in options:
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("Income Distribution")
            counts = df['income'].value_counts()
            fig = px.pie(names=counts.index, values=counts.values, hole=0.6, 
                        color_discrete_sequence=['#7C3AED', '#2D323E'])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#F8FAFC'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    
        
    with col2:
        with st.container(border=True):
            st.subheader("Age vs Income")
            fig = px.box(df, x='income', y='age', color='income', 
                        color_discrete_sequence=['#A78BFA', '#7C3AED'])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#F8FAFC'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    

    with st.container(border=True):
        st.subheader("Interactive Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Purples')
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
        st.plotly_chart(fig_corr, use_container_width=True)



    with st.container(border=True):
        st.subheader("Workclass vs Income")
        fig_bar = px.histogram(df, x='workclass', color='income', barmode='group', 
                            color_discrete_sequence=['#A78BFA', '#7C3AED'])
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'))
        st.plotly_chart(fig_bar, use_container_width=True)


# --- 3. Model & Prediction ---
elif "Intelligence" in options:
    st.markdown("<h1>Machine Learning Models</h1>", unsafe_allow_html=True)
    
    X, y, le, model_columns = prepare_data_for_model(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Unified Configuration Section ---
    with st.container(border=True):
        st.subheader("Model Configuration")
        c_model, c_params, c_action = st.columns([1, 1, 1])
        
        with c_model:
            model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
        
        params = {}
        with c_params:
            if model_choice == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 200, 50, 10)
                max_depth = st.slider("Max Depth", 5, 50, 20)
                params['n_estimators'] = n_estimators
                params['max_depth'] = max_depth
            else:
                C_val = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
                params['C'] = C_val

        with c_action:
            st.write(f"**Training Samples:** {X_train.shape[0]}")
            st.write(f"**Testing Samples:** {X_test.shape[0]}")
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            train_btn = st.button("Start Training", use_container_width=True)

    # --- Training Logic & Results ---
    if train_btn:
        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                            max_depth=params['max_depth'], random_state=42)
            else:
                model = LogisticRegression(C=params['C'], max_iter=1000)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            st.success(f"Model Trained! Accuracy: {acc:.4f}")
            
            # Save model components
            st.session_state['model'] = model
            st.session_state['model_columns'] = model_columns
            st.session_state['le'] = le
            st.session_state['last_acc'] = acc
            st.session_state['model_type'] = model_choice
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
    
    # Display Results if Available
    if 'last_acc' in st.session_state:
        with st.container(border=True):
            st.subheader("Performance Overview")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Model Accuracy</h3>
                    <h2>{st.session_state['last_acc']:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Model Type</h3>
                    <h2>{st.session_state['model_type']}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Test Data Size</h3>
                    <h2>{len(st.session_state['y_test']):,}</h2>
                </div>
                """, unsafe_allow_html=True)
        
    
            
    if 'y_test' in st.session_state:
        col_metrics, col_cm = st.columns(2)
        
        with col_metrics:
            with st.container(border=True):
                st.subheader("Evaluation Metrics")
                report = classification_report(st.session_state['y_test'], st.session_state['y_pred'], output_dict=True)
                # Force specific height to align with confusion matrix
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"), use_container_width=True, height=400)
        
            
        with col_cm:
            with st.container(border=True):
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Purples',
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['<=50K', '>50K'], y=['<=50K', '>50K'])
                # Force specific height to align with metrics table
                fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), height=400)
                st.plotly_chart(fig_cm, use_container_width=True)
        

# --- 4. Unsupervised Learning ---
elif "Clusters" in options:
    st.markdown("<h1>Unsupervised Learning</h1>", unsafe_allow_html=True)
    
    # Preprocessing
    X, y, le, _ = prepare_data_for_model(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Configuration Section ---
    with st.container(border=True):
        st.subheader("Clustering Configuration")
        k_val = st.slider("Select K (Clusters)", 2, 10, 3)

    # --- Visualizations Section ---
    col_elbow, col_pca = st.columns(2)

    with col_elbow:
        with st.container(border=True):
            st.subheader("Elbow Method")
            
            with st.spinner("Calculating optimal clusters..."):
                wcss, k_range = calculate_wcss(X_scaled)
                
            fig_elbow = px.line(x=k_range, y=wcss, markers=True, 
                                labels={'x':'Number of Clusters', 'y':'WCSS'})
            fig_elbow.update_traces(line_color='#7C3AED')
            fig_elbow.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#F8FAFC'), 
                margin=dict(t=10, b=10, l=10, r=10), 
                height=450
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
    
    
    with col_pca:
        with st.container(border=True):
            st.subheader("PCA Visualization")
            
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters.astype(str)
            
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                                opacity=0.8, color_discrete_sequence=px.colors.sequential.Purples)
            fig_pca.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#F8FAFC'), 
                margin=dict(t=10, b=10, l=10, r=10),
                height=450
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
    # Display variance ratio outside the container to maintain height symmetry
    st.info(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    

# --- 5. Predict Your Income (New Feature) ---
elif "Predictor" in options:
    st.markdown("<h1>Predict Income Level</h1>", unsafe_allow_html=True)
    
    with st.container(border=True):
        if 'model' not in st.session_state:
            st.warning("Please train a model in the 'Model & Prediction' tab first.")
        else:
            st.write("Enter your details below to see what the model predicts.")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Age", 17, 90, 30)
                    workclass = st.selectbox("Workclass", df['workclass'].unique())
                    education = st.selectbox("Education", df['education'].unique())
                    marital_status = st.selectbox("Marital Status", df['marital-status'].unique())
                    occupation = st.selectbox("Occupation", df['occupation'].unique())
                    relationship = st.selectbox("Relationship", df['relationship'].unique())
                    race = st.selectbox("Race", df['race'].unique())
                
                with col2:
                    sex = st.selectbox("Sex", df['sex'].unique())
                    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
                    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
                    hours_per_week = st.number_input("Hours per Week", 1, 100, 40)
                    native_country = st.selectbox("Native Country", df['native-country'].unique())
                    
                    # Hidden/Default fields (set to mean/defaults)
                    education_num = 10 
                    fnlwgt = df['fnlwgt'].mean() 
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Predict")
                
            if submit:
                # Create a DataFrame for user input
                input_data = {
                    'age': [age], 'workclass': [workclass], 'fnlwgt': [fnlwgt],
                    'education': [education], 'education-num': [education_num],
                    'marital-status': [marital_status], 'occupation': [occupation],
                    'relationship': [relationship], 'race': [race], 'sex': [sex],
                    'capital-gain': [capital_gain], 'capital-loss': [capital_loss],
                    'hours-per-week': [hours_per_week], 'native-country': [native_country]
                }
                
                input_df = pd.DataFrame(input_data)
                
                # One-Hot Encode and Align Columns
                input_encoded = pd.get_dummies(input_df)
                
                # Reindex to match training columns
                model_columns = st.session_state['model_columns']
                input_ready = input_encoded.reindex(columns=model_columns, fill_value=0)
                
                # Predict
                model = st.session_state['model']
                prediction = model.predict(input_ready)
                le = st.session_state['le']
                result = le.inverse_transform(prediction)[0]
                
                if result == ">50K":
                    st.success(f"Prediction: {result}")
                else:
                    st.info(f"Prediction: {result}")

# --- Global Sidebar Footer ---
with st.sidebar:
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.05);">
            <p style="margin:0; font-size:0.7rem; color:#9CA3AF; letter-spacing:0.05em;">SYSTEM STATUS</p>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">
                <div style="display:flex; align-items:center; gap:6px;">
                    <div style="width:8px; height:8px; background:#10B981; border-radius:50%; box-shadow:0 0 10px #10B981;"></div>
                    <span style="font-size:0.85rem; color:#F9FAFB; font-weight:600;">System Online</span>
                </div>
                <span style="font-size:0.75rem; color:#9CA3AF;">v1.2.0</span>
            </div>
        </div>
        <div style="text-align:center; padding:15px; opacity:0.4; font-size:0.65rem; color:#9CA3AF;">
            &copy; 2024 Earnomly Lab
        </div>
    """, unsafe_allow_html=True)

