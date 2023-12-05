from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("SHAP Analysis")

    # Upload CSV or Excel file
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        # Display uploaded data
        st.subheader("Uploaded Data")
        st.write(df.head())

        # Select target variable
        target_variable = st.selectbox("Select the target variable", df.columns)

        # Select feature variables
        feature_variables = st.multiselect("Select feature variables", df.columns)

        # Set adjustable parameters
        random_state = st.slider("Random State", 0, 100, 42)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.01)

        if st.button("Generate SHAP Plots"):
            # Check if the target variable is categorical or continuous
            if df[target_variable].dtype == 'O':  # 'O' represents object data type, which is often used for categorical variables
                # Convert the target variable to numeric labels for classification
                label_encoder = LabelEncoder()
                df[target_variable] = label_encoder.fit_transform(df[target_variable])

                # Train a simple decision tree classification model with adjustable parameters
                model = train_classification_model(df, target_variable, feature_variables, random_state, test_size)

                # Perform SHAP analysis
                shap_values, expected_value, feature_names = shap_analysis(model, df[feature_variables])

                # Display summary of classification results
                display_classification_results(model, df, target_variable, feature_variables)

            else:
                # Train a simple decision tree regression model with adjustable parameters
                model = train_regression_model(df, target_variable, feature_variables, random_state, test_size)

                # Perform SHAP analysis
                shap_values, expected_value, feature_names = shap_analysis(model, df[feature_variables])

                # Display summary of regression results
                display_regression_results(model, df, target_variable, feature_variables)

            # Plot SHAP summary plot
            st.subheader("SHAP Summary Plot")
            plot_summary(shap_values, feature_names)

            # Plot feature importances
            st.subheader("Feature Importances Plot")
            plot_feature_importances(model, feature_variables)

            # Plot SHAP dependence plots
            st.subheader("SHAP Dependence Plots")
            plot_dependence(shap_values, df[feature_variables], feature_names)

def load_data(uploaded_file):
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('xls', 'xlsx')):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("Invalid file format. Please upload a CSV or Excel file.")
        st.stop()

    return df

def train_classification_model(df, target_variable, feature_variables, random_state, test_size):
    X = df[feature_variables]
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model

def train_regression_model(df, target_variable, feature_variables, random_state, test_size):
    X = df[feature_variables]
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    return model

def shap_analysis(model, features):
    # Get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    return shap_values, explainer.expected_value, features.columns

def plot_summary(shap_values, feature_names):
    # Generate the SHAP summary plot but don't display it immediately
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)

    # Adjust the size of the plot
    plt.gcf().set_size_inches(10, 6)

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the modified plot
    st.pyplot()


def plot_dependence(shap_values, features, feature_names):
    for feature in feature_names:
        shap.dependence_plot(feature, shap_values, features, show=False)
        st.pyplot()

def display_classification_results(model, df, target_variable, feature_variables):
    X_test = df[feature_variables]
    y_test = df[target_variable]

    y_pred = model.predict(X_test)

    st.subheader("Classification Results")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Handle zero values in y_true to avoid division by zero
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return np.nan  # Handle the case where all y_true values are zero

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def display_regression_results(model, df, target_variable, feature_variables):
    X_test = df[feature_variables]
    y_test = df[target_variable]

    y_pred = model.predict(X_test)

    st.subheader("Regression Results")
    st.write("Mean Absolute Error (MAE):", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("Mean Squared Error (MSE):", round(mean_squared_error(y_test, y_pred), 2))
    st.write("Root Mean Squared Error (RMSE):", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
    st.write("Mean Absolute Percentage Error (MAPE):", round(mean_absolute_percentage_error(y_test, y_pred), 2))

def plot_feature_importances(model, feature_names):
    if isinstance(model, DecisionTreeClassifier):
        importance_type = 'classifier'
        feature_importances = model.feature_importances_
    else:
        importance_type = 'regressor'
        feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(f'Feature Importances ({importance_type})')
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

if __name__ == "__main__":
    main()