import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import requests
import json

# Chart explanations
chart_explanations = {
    "Histogram": "A histogram displays the distribution of a numerical variable, showing how frequently each range of values appears.",
    "Boxplot": "A boxplot helps visualize the distribution of data and highlights outliers using quartiles.",
    "Scatter Plot": "A scatter plot shows relationships between two numerical variables, useful for identifying correlations.",
    "Line Chart": "A line chart displays trends over time or any continuous variable.",
    "Bar Chart": "A bar chart compares different categories using rectangular bars.",
    "Correlation Heatmap": "A heatmap visualizes correlations between numerical variables, with colors indicating strength of relationships.",
    "Distribution Detection": "This feature identifies the type of distribution (e.g., Normal, Skewed) for a selected numerical column."
}

# Streamlit UI
st.title("📊 Illustra : Universal Data Visualization & Machine Learning Tool")
st.write("Upload a CSV file, visualize data, and train basic machine learning models.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
mode_of_use = st.selectbox("Select Mode of Use", ["Data Visualization and model traning with AI -assiatance ", "Use compeletly autonomus AI agent "])

if uploaded_file and mode_of_use == "Data Visualization and model traning with AI -assiatance ":
    df = pd.read_csv(uploaded_file)
    csv_summary = df.describe().to_string()
    st.write("### Data Preview")

    # Creating a slider for data preview
    int_val = st.slider('Select number of rows to preview', min_value=1, max_value=df.shape[0], value=5, step=1)
    st.dataframe(df.head(int_val))

    st.write("### Null values in your dataset")
    st.write(df.isna().sum())

    # Dropdown for handling missing values
    fill_method = st.selectbox("Select a method to fill missing values", ["None", "Mean", "Median", "Mode"])
    if fill_method != "None":
        for column in df.select_dtypes(include=[np.number]).columns:
            if fill_method == "Mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif fill_method == "Median":
                df[column].fillna(df[column].median(), inplace=True)
            elif fill_method == "Mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
        st.success(f"Missing values have been replaced using {fill_method}.")
        st.write(df.isna().sum())

    # Multicollinearity Detection
    st.write("### Multicollinearity Detection")
    numerical_df = df.select_dtypes(include=['number']).dropna()
    if not numerical_df.empty:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numerical_df.columns
        vif_data["VIF"] = [variance_inflation_factor(numerical_df.values, i) for i in range(numerical_df.shape[1])]
        st.dataframe(vif_data)

        # One-Hot Encoding
        st.write("### One-Hot Encoding")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            selected_cols = st.multiselect("Select categorical columns to one-hot encode", categorical_cols)
            if st.button("Apply One-Hot Encoding") and selected_cols:
                df = pd.get_dummies(df, columns=selected_cols, drop_first=True)
                st.success(f"One-Hot Encoding applied to {', '.join(selected_cols)}.")
                st.write("Updated Data Preview:")
                st.dataframe(df.head())
        else:
            st.write("No categorical columns found for one-hot encoding.")

    # Chart selection
    chart_type = st.selectbox("Select a chart type", list(chart_explanations.keys()))
    st.write(f"**About {chart_type}:** {chart_explanations[chart_type]}")

    if chart_type == "Histogram":
        column = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)
        fig = px.histogram(df, x=column)
        st.plotly_chart(fig)

    elif chart_type == "Boxplot":
        column = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)
        fig = px.box(df, y=column)
        st.plotly_chart(fig)

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis", df.select_dtypes(include=['number']).columns)
        y_col = st.selectbox("Select Y-axis", df.select_dtypes(include=['number']).columns)
        fig = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif chart_type == "Line Chart":
        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = st.selectbox("Select Y-axis", df.select_dtypes(include=['number']).columns)
        fig = px.line(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif chart_type == "Bar Chart":
        x_col = st.selectbox("Select a categorical column", df.select_dtypes(include=['object']).columns)
        y_col = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)
        fig = px.bar(df, x=x_col, y=y_col)
        st.plotly_chart(fig)

    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    elif chart_type == "Distribution Detection":
        column = st.selectbox("Select a numerical column", df.select_dtypes(include=['number']).columns)

        # Compute skewness and kurtosis
        skewness = stats.skew(df[column].dropna())
        kurtosis = stats.kurtosis(df[column].dropna())

        # Check normality using Shapiro-Wilk test
        shapiro_test = stats.shapiro(df[column].dropna())
        p_value = shapiro_test.pvalue

        # Determine distribution type
        if p_value > 0.05:
            distribution_type = "Normal Distribution"
        elif skewness > 1 or skewness < -1:
            distribution_type = "Highly Skewed Distribution"
        else:
            distribution_type = "Moderately Skewed Distribution"

        # Display results
        st.write(f"**Distribution Analysis for '{column}':**")
        st.write(f"- **Skewness:** {skewness:.2f}")
        st.write(f"- **Kurtosis:** {kurtosis:.2f}")
        st.write(f"- **Shapiro-Wilk Test p-value:** {p_value:.4f}")
        st.write(f"- **Identified Distribution Type:** {distribution_type}")

        # Plot histogram with KDE
        fig, ax = plt.subplots()
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # DeepSeek AI Integration
    st.write("### DeepSeek AI Model")
    model_id = "deepseek/deepseek-r1-distill-llama-70b:free"
    user_input = st.text_area("Enter text for DeepSeek AI analysis")
    API_KEY = "Enter api key"
    # Define the OpenRouter API URL
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    if st.button("Analyze with DeepSeek AI"):

        if not API_KEY:
            st.error("API Key is missing. Please provide a valid OpenRouter API key.")
        elif not user_input:
            st.error("Please enter a message.")
        else:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "deepseek/deepseek-r1-distill-llama-70b:free",  
                "messages": [{"role": "system", "content":"""you are a data scientist looking to analyze dataset, if you are asked to give  a suggestion about which model to recommend you must answer from the following ml model :Linear Regression, Logistic Regression, Decision Tree Regressor,
                                                 Decision Tree Classifier, SVM Regressor, SVM Classifier ; now i want you to answer to question in short , like if asekd for ml model just nswer in one word , if sked for insghts the work as a data agent perform analytics and tell what could this data set leads to and also give the sugestion about  which charts should   \n"""+csv_summary},
                             {"role": "user", "content": user_input}]
            }

            # Send request
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

            # Display response
            if response.status_code == 200:
                response_data = response.json()
                reply = response_data["choices"][0]["message"]["content"]
                st.write("🤖 **DeepSeek AI Response:**")
                st.write(reply)
            else:
                st.error(f"Error {response.status_code}: {response.text}")



    # Machine Learning Models
    st.write("### Train a Machine Learning Model")
    model_type = st.selectbox("Select a Model", ["Linear Regression", "Logistic Regression", "Decision Tree Regressor",
                                                 "Decision Tree Classifier", "SVM Regressor", "SVM Classifier"])
    target_column = st.selectbox("Select Target Variable", df.columns)

    if st.button("Train Model"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical variables into numerical values
        categorical_cols = X.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif model_type == "SVM Regressor":
            model = SVR()
        elif model_type == "SVM Classifier":
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if model_type in ["Linear Regression", "Decision Tree Regressor", "SVM Regressor"]:
            st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
        else:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")




    st.write("📌 Tip: Choose different charts to understand your data better!")












# autonmous AI agent

if uploaded_file and mode_of_use == "Use compeletly autonomus AI agent ":
    df = pd.read_csv(uploaded_file)
    csv_summary = df.describe().to_string()
    st.write("### Data Preview")
    st.dataframe(df.head())
    def run_deepseek_agent(agent_role, prompt, api_key):
        API_URL = "https://openrouter.ai/api/v1/chat/completions" 
        # Payload for DeepSeek API
        payload = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",  
        "messages": [
            {"role": "system", "content": f"You are a {agent_role}.\n"+csv_summary},
            {"role": "user", "content": prompt}
        ]
        }

        # Headers for the API request
        headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
        }

        # Make the API request
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    # Check if the request was successful
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]  
        else:
            return f"Error {response.status_code}: {response.text}"

# Streamlit UI
    st.title("🤖 Multi-Agent AI System")

# File uploader

    

    # Multi-Agent System
    if st.button("Run Multi-Agent System"):
        API_KEY = "enter api key"  

        # Data Agent
        data_agent_response = run_deepseek_agent(
            agent_role="Data Agent",
            prompt=f"Clean and preprocess the following dataset:\n{df.head().to_string()}",
            api_key=API_KEY
        )
        st.write("🧹 **Data Agent Response:**")
        st.write(data_agent_response)

        # Model Selection Agent
        model_agent_response = run_deepseek_agent(
            agent_role="Model Selection Agent",
            prompt=f"Recommend the best machine learning model for this dataset: (in one line)\n{df.head().to_string()}",
            api_key=API_KEY
        )
        st.write("🤖 **Model Selection Agent Response:**")
        st.write(model_agent_response)

        # Visualization Agent
        visualization_agent_response = run_deepseek_agent(
            agent_role="Visualization Agent",
            prompt=f"Suggest visualizations for this dataset:(just give the names of chart and associated varibles with each chart )\n{df.head().to_string()}",
            api_key=API_KEY
        )
        st.write("📊 **Visualization Agent Response:**")
        st.write(visualization_agent_response)

        # Report Agent
        report_agent_response = run_deepseek_agent(
            agent_role="Report Agent",
            prompt=f"Summarize the findings from the dataset:(summarize in very short )\n{df.head().to_string()}",
            api_key=API_KEY
        )
        st.write("📝 **Report Agent Response:**")
        st.write(report_agent_response)
