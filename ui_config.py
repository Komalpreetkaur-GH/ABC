import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Root Variables for Theme */
        :root {
            --bg-color: #0B0E14; /* Deep Navy/Black */
            --card-color: #151A25; /* Slightly lighter card bg */
            --text-primary: #FFFFFF;
            --text-secondary: #94A3B8;
            --accent-purple: #7C3AED;
            --accent-glow: rgba(124, 58, 237, 0.5);
            --accent-gradient: linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%);
            --glass-border: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Global Styling */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
            background-color: var(--bg-color);
        }

        /* App Background */
        .stApp {
            background-color: var(--bg-color);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0F1218;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        [data-testid="stSidebar"] * {
            color: var(--text-secondary) !important;
        }

        .glass-card, [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(21, 26, 37, 0.9), rgba(21, 26, 37, 0.6));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            margin-bottom: 2rem !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .glass-card:hover, [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(124, 58, 237, 0.15);
            border-color: rgba(124, 58, 237, 0.3);
        }

        /* Metric Card styling mimicking the 'Assets' cards */
        .metric-card {
            background-color: var(--card-color);
            border: var(--glass-border);
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
            margin-bottom: 24px;
        }
        
        .metric-card h3 {
            font-size: 0.9rem !important;
            color: var(--text-secondary) !important;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .metric-card h2 {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(90deg, #fff, #cecece);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        
        h1 {
            font-size: 2.5rem !important;
            background: linear-gradient(90deg, #FFFFFF, #94A3B8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Buttons - Neon Aesthetics */
        .stButton>button {
            width: 100%;
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 0 15px var(--accent-glow);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 0 25px var(--accent-glow);
            transform: translateY(-2px);
            filter: brightness(1.1);
        }

        /* Inputs */
        .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
            background-color: #12151C;
            color: var(--text-primary);
            border: 1px solid #2D323E;
            border-radius: 12px;
            height: 45px;
        }
        
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
            border-color: var(--accent-purple) !important;
            box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 20px;
            background-color: transparent;
            color: var(--text-secondary);
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--accent-purple);
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(124, 58, 237, 0.1) !important;
            color: var(--accent-purple) !important;
            border-bottom: none !important;
        }
        
        /* Plotly adjustments */
        .js-plotly-plot .plotly .main-svg {
            background: rgba(0,0,0,0) !important;
        }
        </style>
    """, unsafe_allow_html=True)
