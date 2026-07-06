import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib



# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "Employee-Attrition - Employee-Attrition.csv"
)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        os.path.join(BASE_DIR, "models", "best_model.pkl")
    )

    scaler = joblib.load(
        os.path.join(BASE_DIR, "models", "scaler.pkl")
    )

    feature_columns = joblib.load(
        os.path.join(BASE_DIR, "models", "feature_columns.pkl")
    )

    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("📊 Employee Attrition Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🤖 Predict Employee Attrition"
    ]
)

# ===================================================
# DASHBOARD PAGE
# ===================================================
if page == "🏠 Dashboard":

    st.title("📊 Employee Attrition Dashboard")

    st.markdown(
        "### Employee Insights Dashboard"
    )

    # ---------------- KPI ----------------

    total_emp = len(df)

    left_emp = df[df["Attrition"] == "Yes"].shape[0]

    stay_emp = df[df["Attrition"] == "No"].shape[0]

    attrition_rate = round((left_emp / total_emp) * 100, 2)

    avg_age = round(df["Age"].mean(), 1)

    avg_income = round(df["MonthlyIncome"].mean(), 0)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👨‍💼 Total Employees", total_emp)

    c2.metric("🚪 Employees Left", left_emp)

    c3.metric("📈 Attrition Rate", f"{attrition_rate}%")

    c4.metric("💰 Avg Monthly Income", f"₹ {avg_income:,.0f}")

    st.divider()

    # ---------------- CHARTS ----------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            df,
            names="Attrition",
            title="Employee Attrition Distribution",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        dept = df.groupby("Department")["Attrition"] \
                 .apply(lambda x: (x == "Yes").sum()) \
                 .reset_index(name="Employees Left")

        fig = px.bar(
            dept,
            x="Department",
            y="Employees Left",
            color="Department",
            title="Department-wise Attrition"
        )

        st.plotly_chart(fig, use_container_width=True)

# ===================================================
# ===================================================
# PREDICTION PAGE
# ===================================================
else:

    st.title("🤖 Employee Attrition Prediction")
    st.markdown("### Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", 18, 60, 30)

        daily_rate = st.number_input("Daily Rate", 100, 1500, 800)

        distance = st.number_input("Distance From Home", 1, 30, 5)

        education = st.selectbox("Education",[1,2,3,4,5])

        environment = st.selectbox(
            "Environment Satisfaction",
            [1,2,3,4]
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            20,
            100,
            60
        )

        job_involvement = st.selectbox(
            "Job Involvement",
            [1,2,3,4]
        )

        job_level = st.selectbox(
            "Job Level",
            [1,2,3,4,5]
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1,2,3,4]
        )

        monthly_income = st.number_input(
            "Monthly Income",
            1000,
            30000,
            5000
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            1000,
            30000,
            15000
        )

        num_companies = st.number_input(
            "Number of Companies Worked",
            0,
            10,
            2
        )

        salary_hike = st.number_input(
            "Percent Salary Hike",
            10,
            30,
            15
        )

        performance = st.selectbox(
            "Performance Rating",
            [3,4]
        )

        relationship = st.selectbox(
            "Relationship Satisfaction",
            [1,2,3,4]
        )

    with col2:

        stock = st.selectbox(
            "Stock Option Level",
            [0,1,2,3]
        )

        total_work = st.number_input(
            "Total Working Years",
            0,
            40,
            10
        )

        training = st.number_input(
            "Training Times Last Year",
            0,
            10,
            2
        )

        work_life = st.selectbox(
            "Work Life Balance",
            [1,2,3,4]
        )

        years_company = st.number_input(
            "Years At Company",
            0,
            40,
            5
        )

        years_role = st.number_input(
            "Years In Current Role",
            0,
            20,
            3
        )

        years_promotion = st.number_input(
            "Years Since Last Promotion",
            0,
            15,
            1
        )

        years_manager = st.number_input(
            "Years With Current Manager",
            0,
            20,
            3
        )

        business_travel = st.selectbox(
            "Business Travel",
            sorted(df["BusinessTravel"].unique())
        )

        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique())
        )

        education_field = st.selectbox(
            "Education Field",
            sorted(df["EducationField"].unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].unique())
        )

        job_role = st.selectbox(
            "Job Role",
            sorted(df["JobRole"].unique())
        )

        marital_status = st.selectbox(
            "Marital Status",
            sorted(df["MaritalStatus"].unique())
        )

        overtime = st.selectbox(
            "OverTime",
            sorted(df["OverTime"].unique())
        )

    st.divider()

    predict = st.button("🚀 Predict Attrition")

# ===================================================
# PREDICTION
# ===================================================

    if predict:

        # -----------------------------
        # Create Input Dictionary
        # -----------------------------
        input_data = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance,
            "Education": education,
            "EnvironmentSatisfaction": environment,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies,
            "PercentSalaryHike": salary_hike,
            "PerformanceRating": performance,
            "RelationshipSatisfaction": relationship,
            "StockOptionLevel": stock,
            "TotalWorkingYears": total_work,
            "TrainingTimesLastYear": training,
            "WorkLifeBalance": work_life,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promotion,
            "YearsWithCurrManager": years_manager,
            "BusinessTravel": business_travel,
            "Department": department,
            "EducationField": education_field,
            "Gender": gender,
            "JobRole": job_role,
            "MaritalStatus": marital_status,
            "OverTime": overtime
        }

        # -----------------------------
        # Convert to DataFrame
        # -----------------------------
        input_df = pd.DataFrame([input_data])

        # -----------------------------
        # One Hot Encoding
        # -----------------------------
        input_df = pd.get_dummies(input_df)

        # -----------------------------
        # Match Training Columns
        # -----------------------------
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # -----------------------------
        # Standardize
        # -----------------------------
        input_scaled = scaler.transform(input_df)

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("❌ Employee is likely to Leave")

            st.metric(
                "Probability of Leaving",
                f"{probability*100:.2f}%"
            )

            st.warning("""
### Suggested HR Actions

✅ Schedule a one-to-one discussion.

✅ Review workload.

✅ Improve Work-Life Balance.

✅ Discuss Career Growth.

✅ Improve Recognition & Rewards.
""")

        else:

            st.success("✅ Employee is likely to Stay")

            st.metric(
                "Probability of Staying",
                f"{(1-probability)*100:.2f}%"
            )

            st.success("""
### Suggested HR Actions

✅ Continue Employee Engagement.

✅ Reward Good Performance.

✅ Encourage Career Development.

✅ Maintain Current Satisfaction Level.
""")