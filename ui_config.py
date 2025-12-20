import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Root Variables for Theme */
        :root {
            --bg-color: #030712; /* Deeper background */
            --card-glass: rgba(17, 24, 39, 0.7);
            --text-primary: #F9FAFB;
            --text-secondary: #9CA3AF;
            --accent-purple: #8B5CF6;
            --accent-blue: #3B82F6;
            --glass-border: 1px solid rgba(255, 255, 255, 0.1);
            --glass-highlight: 1px solid rgba(255, 255, 255, 0.2);
        }

        /* Animated Background Blobs */
        .stApp {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.2) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.2) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(59, 130, 246, 0.15) 0px, transparent 50%);
        }

        /* Add breathing room at the bottom of the page */
        .block-container {
            padding-bottom: 100px !important;
        }

        /* The 'Liquid' effect via background blobs */
        .stApp::before, .stApp::after {
            content: "";
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            filter: blur(120px);
            opacity: 0.25;
            z-index: -1;
            animation: move 25s infinite ease-in-out alternate;
        }
        
        .stApp::after {
            background: linear-gradient(135deg, var(--accent-blue), #10B981); /* Cyan/Emerald mix */
            animation: move-reverse 30s infinite ease-in-out alternate;
            left: 40%;
            top: 40%;
            opacity: 0.2;
        }

        @keyframes move {
            0% { transform: translate(-20%, -20%) rotate(0deg) scale(1); }
            50% { transform: translate(10%, 20%) rotate(90deg) scale(1.2); }
            100% { transform: translate(30%, -10%) rotate(180deg) scale(1); }
        }
        
        @keyframes move-reverse {
            0% { transform: translate(40%, 30%) rotate(0deg) scale(1.1); }
            50% { transform: translate(-10%, -20%) rotate(-90deg) scale(1.3); }
            100% { transform: translate(-30%, 40%) rotate(-180deg) scale(1.1); }
        }

        /* Global Styling */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(3, 7, 18, 0.4) !important;
            backdrop-filter: blur(30px) saturate(200%);
            border-right: var(--glass-border);
        }
        
        /* Premium Sidebar Logo/Header Container */
        .sidebar-brand {
            padding: 1.5rem 0;
            text-align: center;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1));
            border-radius: 20px;
            border: var(--glass-border);
            margin-bottom: 2rem;
        }

        /* Custom Radio Navigation Styling */
        div[data-testid="stSidebarNav"] {
            display: none; /* Hide default nav if any */
        }

        /* Target the radio group */
        div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding: 0 10px;
        }

        /* Style each radio option as a card */
        div[role="radiogroup"] > label {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 16px !important;
            padding: 16px 24px !important;
            color: var(--text-secondary) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            margin: 4px 0 !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
        }

        div[role="radiogroup"] > label::before {
            content: "";
            display: block;
            width: 6px;
            height: 6px;
            background: var(--text-secondary);
            border-radius: 50%;
            opacity: 0.3;
            transition: all 0.3s ease;
        }

        div[role="radiogroup"] > label:hover {
            background: rgba(255, 255, 255, 0.06) !important;
            border-color: rgba(139, 92, 246, 0.4) !important;
            transform: translateX(6px);
            color: white !important;
        }

        div[role="radiogroup"] > label:hover::before {
            background: var(--accent-purple);
            opacity: 1;
            box-shadow: 0 0 8px var(--accent-purple);
        }

        /* Active State - When radio is selected */
        div[role="radiogroup"] > label[data-selected="true"] {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15)) !important;
            border-color: var(--accent-purple) !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.1) !important;
            color: white !important;
            font-weight: 600 !important;
        }

        div[role="radiogroup"] > label[data-selected="true"]::before {
            background: var(--accent-purple);
            opacity: 1;
            width: 8px;
            height: 8px;
            box-shadow: 0 0 10px var(--accent-purple);
        }

        /* Hide the actual radio circle */
        div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
            font-size: 0.9rem !important;
            margin: 0 !important;
            letter-spacing: 0.02em;
        }
        
        div[role="radiogroup"] input {
            display: none;
        }

        /* Sidebar Header Styling */
        [data-testid="stSidebar"] h2 {
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--text-secondary) !important;
            margin: 2rem 0 1rem 1.2rem !important;
            opacity: 0.8;
        }

        /* Liquid Glass Card Effect */
        .glass-card, [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--card-glass) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
            border: var(--glass-border) !important;
            border-top: var(--glass-highlight) !important; /* Top highlight for depth */
            border-radius: 28px !important;
            padding: 24px;
            box-shadow: 
                0 10px 15px -3px rgba(0, 0, 0, 0.1),
                0 4px 6px -2px rgba(0, 0, 0, 0.05),
                inset 0 0 0 1px rgba(255, 255, 255, 0.05); /* Thin inner rim */
            margin-bottom: 2rem !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .glass-card:hover, [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-4px) scale(1.002);
            background: rgba(31, 41, 55, 0.7) !important;
            box-shadow: 
                0 20px 25px -5px rgba(0, 0, 0, 0.2),
                0 10px 10px -5px rgba(0, 0, 0, 0.1);
            border-color: rgba(139, 92, 246, 0.3) !important;
        }

        /* Metric Card styling */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: var(--glass-border);
            border-radius: 22px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .metric-card:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: var(--accent-purple);
        }
        
        .metric-card h3 {
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary) !important;
            margin-bottom: 8px;
        }
        
        .metric-card h2 {
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #fff 30%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            word-wrap: break-word;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em;
        }
        
        h1 {
            font-size: 3rem !important;
            background: linear-gradient(135deg, #F9FAFB 0%, #9CA3AF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem !important;
        }

        /* Buttons */
        .stButton>button {
            background: rgba(139, 92, 246, 0.2);
            backdrop-filter: blur(10px);
            color: #DDD6FE;
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 16px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stButton>button:hover {
            background: rgba(139, 92, 246, 0.4);
            border-color: #8B5CF6;
            color: white;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
            transform: translateY(-2px);
        }

        /* Inputs */
        .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
            background-color: rgba(255, 255, 255, 0.03) !important;
            backdrop-filter: blur(5px);
            color: var(--text-primary) !important;
            border: var(--glass-border) !important;
            border-radius: 14px !important;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: var(--accent-purple) !important;
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background: rgba(255, 255, 255, 0.03);
            padding: 6px;
            border-radius: 18px;
            border: var(--glass-border);
        }
        .stTabs [data-baseweb="tab"] {
            height: 38px;
            border-radius: 14px;
            background-color: transparent;
            color: var(--text-secondary);
            font-weight: 500;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }
        
        /* Plotly adjustments */
        .js-plotly-plot .plotly .main-svg {
            background: rgba(0,0,0,0) !important;
        }
        
        /* Custom scrollbar for better aesthetics */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)

