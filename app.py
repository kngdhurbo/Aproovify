import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        return None

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('loan_approval_dataset.csv')
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        return df
    except Exception as e:
        return None

st.set_page_config(page_title="LoanGuard: AI-Powered Credit Assessment", page_icon="🏦", layout="wide")

# Load assets
df = load_data()
model = load_model()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home / Project Info", "Exploratory Data Analysis", "Live Prediction"])

if page == "Home / Project Info":
    st.title("🏦 Approvify: Loan Approval Prediction System")
    st.write("Welcome to the Approvify Analytics & Prediction System.")
    st.markdown("""
    This application serves bank loan officers and data analysts by providing two primary functions:
    
    - **Analytics Dashboard**: An interactive visualization suite to understand historical loan trends, applicant demographics, and financial health metrics.
    - **Prediction Engine**: A machine learning interface where users can input specific applicant details and receive an instant "Approved" or "Rejected" verdict.
    
    Use the sidebar to navigate between different modules.
    """)
    
    st.markdown("---")
    st.subheader("Dataset Preview")
    st.write("Here is a quick glance at the historical loan data we use for analytics and predictions:")
    if df is not None:
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("Data not available.")

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis Dashboard")
    
    if df is not None:
        # KPI metrics
        total_applicants = len(df)
        approval_rate = (df['loan_status'] == 'Approved').mean() * 100
        avg_loan = df['loan_amount'].mean()
        avg_cibil = df['cibil_score'].mean()
        
        st.markdown("### Top-Level Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Applicants", f"{total_applicants:,}")
        col2.metric("Approval Rate", f"{approval_rate:.1f}%")
        col3.metric("Average Loan Amount", f"₹ {avg_loan:,.0f}")
        col4.metric("Average CIBIL Score", f"{avg_cibil:.0f}")
        
        st.markdown("---")
        
        row1_col1, row1_col2 = st.columns(2)
        
        # Visual 1: Donut Chart
        with row1_col1:
            st.subheader("Loan Status Distribution")
            status_counts = df['loan_status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig1 = px.pie(status_counts, values='Count', names='Status', hole=0.4,
                         color='Status', color_discrete_map={'Approved':'#2E8B57', 'Rejected':'#CD5C5C'})
            st.plotly_chart(fig1, use_container_width=True)
            
        # Visual 2: Scatter plot
        with row1_col2:
            st.subheader("Financial Health Analysis")
            fig2 = px.scatter(df, x='income_annum', y='loan_amount', color='loan_status',
                             opacity=0.6, hover_data=['cibil_score'],
                             color_discrete_map={'Approved':'#2E8B57', 'Rejected':'#CD5C5C'},
                             labels={'income_annum': 'Annual Income', 'loan_amount': 'Loan Amount'})
            st.plotly_chart(fig2, use_container_width=True)
            
        row2_col1, row2_col2 = st.columns(2)
        
        # Visual 3: Box Plot for CIBIL Score
        with row2_col1:
            st.subheader("Credit Score Impact (CIBIL)")
            fig3 = px.box(df, x='loan_status', y='cibil_score', color='loan_status',
                         color_discrete_map={'Approved':'#2E8B57', 'Rejected':'#CD5C5C'},
                         labels={'loan_status': 'Loan Status', 'cibil_score': 'CIBIL Score'})
            st.plotly_chart(fig3, use_container_width=True)
            
        # Visual 4: Bar Chart for Asset Analysis
        with row2_col2:
            st.subheader("Asset Value Analysis")
            assets_df = df.groupby('loan_status')[['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value']].mean().reset_index()
            assets_melted = pd.melt(assets_df, id_vars=['loan_status'], var_name='Asset Type', value_name='Average Value')
            assets_melted['Asset Type'] = assets_melted['Asset Type'].str.replace('_assets_value', '').str.title()
            
            fig4 = px.bar(assets_melted, x='Asset Type', y='Average Value', color='loan_status', barmode='group',
                         color_discrete_map={'Approved':'#2E8B57', 'Rejected':'#CD5C5C'},
                         labels={'Average Value': 'Avg Value (₹)', 'loan_status': 'Loan Status'})
            st.plotly_chart(fig4, use_container_width=True)
            
    else:
        st.error("Dataset not found. Please ensure 'loan_approval_dataset.csv' is available.")

elif page == "Live Prediction":
    st.title("🤖 Live Loan Application Prediction")
    st.write("Enter the applicant details below to get an instant credit assessment.")
    
    with st.form("prediction_form"):
        st.subheader("Applicant Details")
        
        col1, col2 = st.columns(2)
        with col1:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
            
        with col2:
            income = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000)
            loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, value=15000000, step=100000)
            loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=15)
            
        st.markdown("---")
        st.subheader("Financial Profile")
        
        col3, col4 = st.columns(2)
        with col3:
            cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650)
            residential_asset = st.number_input("Residential Asset Value (₹)", min_value=0, value=5000000, step=100000)
            commercial_asset = st.number_input("Commercial Asset Value (₹)", min_value=0, value=0, step=100000)
            
        with col4:
            luxury_asset = st.number_input("Luxury Asset Value (₹)", min_value=0, value=0, step=100000)
            bank_asset = st.number_input("Bank Asset Value (₹)", min_value=0, value=1000000, step=100000)
            
        submit_button = st.form_submit_button(label="Predict Status")
        
    if submit_button:
        if model is None:
            st.error("Model not found! Please ensure 'model.pkl' is present.")
        else:
            # Prepare data mapped exactly to the required format
            input_data = pd.DataFrame({
                'no_of_dependents': [dependents],
                'education': [education],
                'self_employed': [self_employed],
                'income_annum': [income],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'residential_assets_value': [residential_asset],
                'commercial_assets_value': [commercial_asset],
                'luxury_assets_value': [luxury_asset],
                'bank_asset_value': [bank_asset]
            })
            
            # Predict
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                confidence = max(proba) * 100
                
                st.markdown("---")
                st.subheader("Assessment Result")
                
                prediction_cols = st.columns([1, 1, 1])
                with prediction_cols[1]:
                    if prediction == 1:
                        st.success(f"### ✅ APPROVED")
                        st.write(f"Confidence Score: **{confidence:.1f}%**")
                    else:
                        st.error(f"### ❌ REJECTED")
                        st.write(f"Confidence Score: **{confidence:.1f}%**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
