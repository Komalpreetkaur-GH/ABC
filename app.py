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

# Apply custom styles
ui_config.apply_styles()

# --- Plotly Theme Configuration ---
PLOT_COLORS = ['#6366F1', '#8B5CF6', '#A78BFA', '#C4B5FD']
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#FAFAFA', size=12),
    margin=dict(t=20, b=40, l=40, r=20),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)'),
)

# Title and Introduction
st.markdown("<h1>Earnomly</h1>", unsafe_allow_html=True)
st.markdown("""
    <p class="intro-text">
        Advanced socioeconomic analysis and predictive modeling platform for the Adult Income Dataset.
    </p>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
               'hours-per-week', 'native-country', 'income']
    try:
        df = pd.read_csv('adultData/adult.data', names=columns, na_values='?', skipinitialspace=True)
        for col in ['workclass', 'occupation', 'native-country']:
            df[col] = df[col].fillna(df[col].mode()[0])
        df.drop_duplicates(inplace=True)
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'adultData/adult.data' exists.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- Preprocessing Helper ---
@st.cache_data
def prepare_data_for_model(df):
    le = LabelEncoder()
    df_model = df.copy()
    df_model['income_encoded'] = le.fit_transform(df_model['income'])
    df_model = df_model.drop('income', axis=1)
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    X = df_encoded.drop('income_encoded', axis=1)
    y = df_encoded['income_encoded']
    return X, y, le, X.columns

@st.cache_resource
def calculate_wcss(X_scaled):
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return wcss, k_range

# --- Sidebar Navigation ---
with st.sidebar:
    ui_config.render_sidebar_brand()
    
    st.markdown("<p class='nav-category'>Navigation</p>", unsafe_allow_html=True)
    options = st.radio("Navigation", 
        ["Data Overview", "Data Analysis", "Supervised Learning", "Unsupervised Learning", "Income Predictor"],
        label_visibility="collapsed")

# --- 1. Data Overview ---
if "Data Overview" in options:
    st.markdown("<h1>Dashboard Overview</h1>", unsafe_allow_html=True)
    
    # Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(ui_config.render_metric_card(
            label="Total Population",
            value=f"{df.shape[0]:,}",
            badge_text="+12%",
            badge_type="success",
            progress=70,
            icon_name="population"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(ui_config.render_metric_card(
            label="Features Analyzed",
            value=str(df.shape[1]),
            badge_text="Stable",
            badge_type="info",
            progress=100,
            icon_name="features"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(ui_config.render_metric_card(
            label="Missing Values",
            value="0.0%",
            badge_text="Clean",
            badge_type="accent",
            progress=100,
            icon_name="clean"
        ), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h2>Sample Data</h2>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    with st.container(border=True):
        st.markdown("<h2>Statistical Summary</h2>", unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
    
    with st.container(border=True):
        st.markdown("<h2>Data Quality</h2>", unsafe_allow_html=True)
        missing = df.isnull().sum()
        if missing[missing > 0].empty:
            st.markdown("<p class='intro-text'>No missing values detected in the dataset.</p>", unsafe_allow_html=True)
        else:
            st.dataframe(missing[missing > 0], use_container_width=True)

# --- 2. Data Analysis (Plotly) ---
elif "Data Analysis" in options:
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("<h2>Income Distribution</h2>", unsafe_allow_html=True)
            counts = df['income'].value_counts()
            fig = px.pie(names=counts.index, values=counts.values, hole=0.65, 
                        color_discrete_sequence=['#6366F1', '#27272A'])
            fig.update_layout(**PLOT_LAYOUT)
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            fig.update_traces(textinfo='percent', textfont_size=12)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        with st.container(border=True):
            st.markdown("<h2>Age vs Income</h2>", unsafe_allow_html=True)
            fig = px.box(df, x='income', y='age', color='income', 
                        color_discrete_sequence=['#6366F1', '#8B5CF6'])
            fig.update_layout(**PLOT_LAYOUT)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown("<h2>Correlation Matrix</h2>", unsafe_allow_html=True)
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, text_auto='.2f', aspect="auto", 
                            color_continuous_scale=[[0, '#18181B'], [0.5, '#3F3F46'], [1, '#6366F1']])
        fig_corr.update_layout(**PLOT_LAYOUT)
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.container(border=True):
        st.markdown("<h2>Workclass Distribution</h2>", unsafe_allow_html=True)
        fig_bar = px.histogram(df, x='workclass', color='income', barmode='group', 
                            color_discrete_sequence=['#6366F1', '#8B5CF6'])
        fig_bar.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- 3. Supervised Learning ---
elif "Supervised Learning" in options:
    st.markdown("<h1>Supervised Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='intro-text'>Random Forest and Logistic Regression classifiers for income prediction.</p>", unsafe_allow_html=True)
    
    X, y, le, model_columns = prepare_data_for_model(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.container(border=True):
        st.markdown("<h2>Model Configuration</h2>", unsafe_allow_html=True)
        c_model, c_params, c_action = st.columns([1, 1, 1])
        
        with c_model:
            model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
        
        params = {}
        with c_params:
            if model_choice == "Random Forest":
                n_estimators = st.slider("Number of Trees", 50, 300, 150, 10)
                max_depth = st.slider("Max Depth", 10, 50, 25)
                params['n_estimators'] = n_estimators
                params['max_depth'] = max_depth
            else:
                C_val = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
                params['C'] = C_val

        with c_action:
            st.markdown(f"**Training Samples:** {X_train.shape[0]:,}")
            st.markdown(f"**Testing Samples:** {X_test.shape[0]:,}")
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            train_btn = st.button("Start Training", use_container_width=True)

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
            
            st.success(f"Model trained successfully. Accuracy: {acc:.4f}")
            
            st.session_state['model'] = model
            st.session_state['model_columns'] = model_columns
            st.session_state['le'] = le
            st.session_state['last_acc'] = acc
            st.session_state['model_type'] = model_choice
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
    
    if 'last_acc' in st.session_state:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(ui_config.render_metric_card(
                label="Model Accuracy",
                value=f"{st.session_state['last_acc']:.2%}",
                badge_type="success"
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(ui_config.render_metric_card(
                label="Model Type",
                value=st.session_state['model_type'],
                badge_type="info"
            ), unsafe_allow_html=True)
        with col3:
            st.markdown(ui_config.render_metric_card(
                label="Test Samples",
                value=f"{len(st.session_state['y_test']):,}",
                badge_type="accent"
            ), unsafe_allow_html=True)
    
    if 'y_test' in st.session_state:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        col_metrics, col_cm = st.columns(2)
        
        with col_metrics:
            with st.container(border=True):
                st.markdown("<h2>Classification Report</h2>", unsafe_allow_html=True)
                report = classification_report(st.session_state['y_test'], st.session_state['y_pred'], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"), use_container_width=True, height=350)
        
        with col_cm:
            with st.container(border=True):
                st.markdown("<h2>Confusion Matrix</h2>", unsafe_allow_html=True)
                cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                fig_cm = px.imshow(cm, text_auto=True, 
                                color_continuous_scale=[[0, '#18181B'], [0.5, '#3F3F46'], [1, '#6366F1']],
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['<=50K', '>50K'], y=['<=50K', '>50K'])
                fig_cm.update_layout(**PLOT_LAYOUT)
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, use_container_width=True)

# --- 4. Unsupervised Learning ---
elif "Unsupervised Learning" in options:
    st.markdown("<h1>Unsupervised Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='intro-text'>K-Means Clustering and Principal Component Analysis (PCA) for pattern discovery.</p>", unsafe_allow_html=True)
    
    X, y, le, _ = prepare_data_for_model(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    with st.container(border=True):
        st.markdown("<h2>Clustering Configuration</h2>", unsafe_allow_html=True)
        k_val = st.slider("Select K (Clusters)", 2, 10, 3)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    col_elbow, col_pca = st.columns(2)

    with col_elbow:
        with st.container(border=True):
            st.markdown("<h2>Elbow Method</h2>", unsafe_allow_html=True)
            
            with st.spinner("Calculating optimal clusters..."):
                wcss, k_range = calculate_wcss(X_scaled)
                
            fig_elbow = px.line(x=list(k_range), y=wcss, markers=True, 
                                labels={'x':'Number of Clusters', 'y':'WCSS'})
            fig_elbow.update_traces(line_color='#6366F1', marker_color='#8B5CF6')
            fig_elbow.update_layout(**PLOT_LAYOUT)
            fig_elbow.update_layout(height=400)
            st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col_pca:
        with st.container(border=True):
            st.markdown("<h2>PCA Visualization</h2>", unsafe_allow_html=True)
            
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters.astype(str)
            
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                                opacity=0.7, 
                                color_discrete_sequence=['#6366F1', '#8B5CF6', '#A78BFA', '#C4B5FD', '#DDD6FE'])
            fig_pca.update_layout(**PLOT_LAYOUT)
            fig_pca.update_layout(height=400)
            st.plotly_chart(fig_pca, use_container_width=True)
    
    var_ratio = pca.explained_variance_ratio_
    st.info(f"Explained Variance Ratio: PC1 = {var_ratio[0]:.3f}, PC2 = {var_ratio[1]:.3f}")

# --- 5. Income Predictor ---
elif "Income Predictor" in options:
    st.markdown("<h1>Income Prediction</h1>", unsafe_allow_html=True)
    
    # Education to education-num mapping (based on dataset)
    edu_num_map = {
        'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
        '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
        'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
        'Prof-school': 15, 'Doctorate': 16
    }
    
    with st.container(border=True):
        if 'model' not in st.session_state:
            st.warning("Please train a model in the Supervised Learning section first.")
        else:
            st.markdown("<p class='intro-text'>Enter your details below to predict income level.</p>", unsafe_allow_html=True)
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Age", 17, 90, 45)
                    workclass = st.selectbox("Workclass", df['workclass'].unique())
                    education = st.selectbox("Education", df['education'].unique())
                    marital_status = st.selectbox("Marital Status", df['marital-status'].unique())
                    occupation = st.selectbox("Occupation", df['occupation'].unique())
                    relationship = st.selectbox("Relationship", df['relationship'].unique())
                    race = st.selectbox("Race", df['race'].unique())
                
                with col2:
                    sex = st.selectbox("Sex", df['sex'].unique())
                    capital_gain = st.number_input("Capital Gain", 0, 100000, 5000)
                    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
                    hours_per_week = st.number_input("Hours per Week", 1, 100, 50)
                    native_country = st.selectbox("Native Country", df['native-country'].unique())
                    
                    # Dynamic education-num based on education selection
                    education_num = edu_num_map.get(education, 10)
                    fnlwgt = df['fnlwgt'].mean() 
                
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                submit = st.form_submit_button("Predict", use_container_width=True)
                
            if submit:
                input_data = {
                    'age': [age], 'workclass': [workclass], 'fnlwgt': [fnlwgt],
                    'education': [education], 'education-num': [education_num],
                    'marital-status': [marital_status], 'occupation': [occupation],
                    'relationship': [relationship], 'race': [race], 'sex': [sex],
                    'capital-gain': [capital_gain], 'capital-loss': [capital_loss],
                    'hours-per-week': [hours_per_week], 'native-country': [native_country]
                }
                
                input_df = pd.DataFrame(input_data)
                
                # Proper encoding: concatenate with original data (without income), encode, extract input row
                df_without_income = df.drop('income', axis=1)
                combined = pd.concat([df_without_income, input_df], ignore_index=True)
                combined_encoded = pd.get_dummies(combined, drop_first=True)
                
                # Extract the last row (our input)
                input_encoded = combined_encoded.iloc[[-1]]
                
                model_columns = st.session_state['model_columns']
                input_ready = input_encoded.reindex(columns=model_columns, fill_value=0)
                
                model = st.session_state['model']
                prediction = model.predict(input_ready)
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_ready)[0]
                    prob_low = proba[0] * 100
                    prob_high = proba[1] * 100
                else:
                    prob_low = prob_high = None
                
                le = st.session_state['le']
                result = le.inverse_transform(prediction)[0]
                
                # Determine styling based on result
                if result == ">50K":
                    gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%)"
                    border_color = "rgba(16, 185, 129, 0.4)"
                    text_color = "#10B981"
                    glow = "0 0 40px rgba(16, 185, 129, 0.2)"
                    icon = '<svg width="48" height="48" viewBox="0 0 24 24" fill="none"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="#10B981" stroke="#10B981" stroke-width="1.5"/></svg>'
                    label = "High Income"
                else:
                    gradient = "linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(99, 102, 241, 0.05) 100%)"
                    border_color = "rgba(99, 102, 241, 0.4)"
                    text_color = "#6366F1"
                    glow = "0 0 40px rgba(99, 102, 241, 0.2)"
                    icon = '<svg width="48" height="48" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="#6366F1" stroke-width="1.5"/><path d="M12 8V12L15 15" stroke="#6366F1" stroke-width="1.5" stroke-linecap="round"/></svg>'
                    label = "Standard Income"
                
                # Probability bar width
                bar_width = prob_high if prob_high is not None else 50
                
                st.markdown(f'''
                    <div style="margin-top: 2rem; padding: 2rem; background: {gradient}; border: 2px solid {border_color}; border-radius: 20px; box-shadow: {glow}; text-align: center;">
                        <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                            {icon}
                        </div>
                        <div style="font-size: 0.75rem; color: #71717A; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Predicted Income Level</div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 800; color: {text_color}; letter-spacing: -0.02em; margin-bottom: 0.5rem;">{result}</div>
                        <div style="font-size: 1rem; color: #FAFAFA; font-weight: 500;">{label}</div>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Probability gauge
                if prob_low is not None:
                    st.markdown(f'''
                        <div style="margin-top: 1.5rem; padding: 1.25rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                                <span style="font-size: 0.7rem; color: #71717A; text-transform: uppercase; letter-spacing: 0.08em;">Confidence Score</span>
                                <span style="font-size: 0.875rem; font-weight: 600; color: #FAFAFA;">{max(prob_low, prob_high):.1f}%</span>
                            </div>
                            <div style="position: relative; height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden;">
                                <div style="position: absolute; left: 0; top: 0; height: 100%; width: {bar_width}%; background: linear-gradient(90deg, #6366F1, #8B5CF6); border-radius: 4px; transition: width 0.6s ease;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-top: 0.75rem;">
                                <div style="text-align: left;">
                                    <div style="font-size: 0.65rem; color: #52525B; text-transform: uppercase;">Less than 50K</div>
                                    <div style="font-size: 1.125rem; font-weight: 700; color: #FAFAFA;">{prob_low:.1f}%</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 0.65rem; color: #52525B; text-transform: uppercase;">More than 50K</div>
                                    <div style="font-size: 1.125rem; font-weight: 700; color: #FAFAFA;">{prob_high:.1f}%</div>
                                </div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

# --- Global Sidebar Footer ---
with st.sidebar:
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
    st.markdown(ui_config.render_status_footer(), unsafe_allow_html=True)
