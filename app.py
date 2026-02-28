import streamlit as st
import pandas as pd
import pickle as pk

# Page config
st.set_page_config(
    page_title="Smart Loan Eligibility Checker",
    page_icon="💰",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f4f6fb; }

    .stButton>button {
        background: linear-gradient(to right, #0a1628, #1a3a6e);
        color: white;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        border-radius: 50px;
        padding: 12px 40px;
        width: 100%;
        margin-top: 10px;
    }
     .stButton {
    width: 100% !important;
}

    .stButton>button:hover { opacity: 0.85; }

    .header-box {
        background: linear-gradient(135deg, #0a1628, #1a3a6e);
        padding: 35px 20px;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        border-bottom: 3px solid #c9a84c;
    }

    .header-box h1 { font-size: 1.8rem; margin-bottom: 8px; }
    .header-box p { opacity: 0.85; font-size: 0.95rem; }

    .stat-box {
        background: white;
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-bottom: 3px solid #c9a84c;
    }

    .stat-box h3 { color: #0a1628; margin: 0; font-size: 1.5rem; font-weight: 800; }
    .stat-box p { color: #666; margin: 5px 0 0 0; font-size: 0.85rem; }

    .tip-box {
        background: #eef1f8;
        border-left: 4px solid #1a3a6e;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 10px 0;
    }

    .result-approved {
        background: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .result-rejected {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .result-approved h2 { color: #28a745; }
    .result-rejected h2 { color: #dc3545; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #0a1628;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, #0a1628, #1a3a6e) !important;
        color: white !important;
    }
            /* Dropdown styling */
.stSelectbox > div > div {
    background-color: white !important;
    border: 2px solid #1a3a6e !important;
    border-radius: 8px !important;
    color: #0a1628 !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div:hover {
    border-color: #c9a84c !important;
}
    </style>
""", unsafe_allow_html=True)

# Load model
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Header
st.markdown("""
    <div class="header-box">
        <h1>💰 Smart Loan Eligibility Checker</h1>
        <p>Ever wondered if your loan will get approved before actually applying?<br>
        Fill in your details below and find out in seconds.</p>
    </div>
""", unsafe_allow_html=True)

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-box"><h3>98.71%</h3><p>Model Accuracy</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><h3>4,269</h3><p>Loans Trained On</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><h3>8</h3><p>Features Analyzed</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# About
st.markdown("""
    <div class="tip-box">
        <p style="margin: 0; color: #333; font-size: 0.95rem;">
             Banks look at a lot of factors before approving a loan — your income, credit score, assets and more.<br><br>
            This tool uses a <strong>Decision Tree model</strong> trained on real loan data to give you a quick idea of your eligibility.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("✏️ Fill in Your Details")

# Tabs
tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "🏦 Loan Details", "🏠 Assets & Score"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    no_of_dep = st.slider(' Number of Dependents', 0, 5)
    graduation = st.selectbox(' Education Level', ["", 'Graduated', 'Not Graduated'])
    self_emp = st.selectbox(' Are you Self Employed?', ["", 'Yes', 'No'])
    Annual_income = st.slider(' Annual Income (₹)', 0, 10000000, step=100000, format="₹%d")
    st.info("👉 Move to **Loan Details** tab to continue")

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    Loan_amount = st.slider(' Loan Amount Required (₹)', 0, 10000000, step=100000, format="₹%d")
    duration_type = st.radio(" Loan Duration Type", ["Years", "Months"])
    if duration_type == "Years":
        Loan_Dur_years = st.slider("📅 Loan Duration (Years)", 1, 30)
        Loan_Dur = Loan_Dur_years * 12
        st.caption(f"= {Loan_Dur} months total")
    else:
        Loan_Dur = st.slider("📅 Loan Duration (Months)", 6, 360, step=6)
        st.caption(f"= {Loan_Dur // 12} years and {Loan_Dur % 12} months")
    st.info("👉 Move to **Assets & Score** tab to continue")

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    cibil = st.slider('📊 CIBIL Score', 300, 900)
    Assets = st.slider('🏠 Total Assets Value (₹)', 0, 100000000, step=500000, format="₹%d")
    st.info("👉 Click **Check My Eligibility** button below when ready!")

st.markdown("---")

# Encode
if graduation == 'Graduated':
    graduation_s = 0
else:
    graduation_s = 1

if self_emp == 'No':
    self_emp_s = 0
else:
    self_emp_s = 1

# Predict button
if st.button("Check My Eligibility"):
    if graduation == "" or self_emp == "":
        st.warning("⚠️ Please fill in all the fields before checking!")
    else:
        pred_data = pd.DataFrame(
            [[no_of_dep, graduation_s, self_emp_s, Annual_income,
              Loan_amount, Loan_Dur, cibil, Assets]],
            columns=['no_of_dependents', 'education', 'self_employed',
                     'income_annum', 'loan_amount', 'loan_term',
                     'cibil_score', 'Assets']
        )
        pred_data = scaler.transform(pred_data)
        predict = model.predict(pred_data)

        if predict[0] == 1:
            st.markdown("""
                <div class="result-approved">
                    <h2>✅ Congratulations!</h2>
                    <p style="font-size: 1.1rem; color: #333;">
                        Based on your details, your loan looks like it would be <strong>Approved!</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            st.markdown("""
                <div class="tip-box" style="margin-top: 15px;">
                     This is a prediction based on your inputs. Actual approval depends on the bank's policies.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-rejected">
                    <h2>❌ Not Approved</h2>
                    <p style="font-size: 1.1rem; color: #333;">
                        Based on your details, your loan might get <strong>Rejected.</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("""
                <div class="tip-box" style="margin-top: 15px;">
                     <strong>Tips to improve your chances:</strong><br><br>
                    • Improve your <strong>CIBIL score</strong> (aim for 750+)<br>
                    • Reduce the <strong>loan amount</strong> or increase duration<br>
                    • Show higher <strong>income or assets</strong>
                </div>
            """, unsafe_allow_html=True)

st.markdown("""
    <div class="tip-box">
         <strong>How it works:</strong> Fill in all 3 tabs → Click Check My Eligibility → Get instant result!
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# Footer

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.85rem;'>
        ❗ This tool is for educational purposes only and does not guarantee actual loan approval.<br><br>
        Built by <strong>Renuka Wagh</strong>
    </div>
""", unsafe_allow_html=True)
