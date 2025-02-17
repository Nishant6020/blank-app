# steamlit app
from function import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,ExtraTreesRegressor,AdaBoostRegressor, VotingRegressor, StackingRegressor,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
# Streamlit App
st.title("Data Analysis & Model Building App")

st.sidebar.title("Menu")
st.sidebar.markdown("________________________")
if st.sidebar.button("clear old cache"):
    clear_data()

uploaded_file = st.sidebar.file_uploader("Upload your data file (CSV, Excel, JSON)", type=['csv', 'xlsx', 'xls', 'json'])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        # Store the DataFrame in session state
        if 'df' not in st.session_state:
            st.session_state.df = df

# External link data fetching
external_link = st.sidebar.text_input("Enter github dataset URL to fetch data:")
if external_link and st.sidebar.button("Fetch Data"):
    df = fetch_data_from_url(external_link)
    if df is not None:
        st.session_state.df = df

if 'df' in st.session_state:
# clean data download
        down = st.sidebar.toggle("Download Clean Data")
        if down:
            file_name = st.sidebar.text_input("Enter file name to save:", "cleaned_data.csv")
            if st.sidebar.button("Download Data"):
                download_clean_data(st.session_state.df, file_name)
                st.success(f"File saved as {file_name}!")
        st.sidebar.markdown("________________________")

        # Data prerview Section (Initially hidden)
        data_prerview_expander = st.sidebar.expander("Data Preview", expanded=False)
        if data_prerview_expander:
            prerview_option = st.sidebar.selectbox("Data Preview", [
                "Select Option",
                "Data Preview",
                "First 5 Rows",
                "Last 5 Rows",
                "10 Rows",
                "20 Rows",
                "50 Rows",
                "Sample Data",
            ])         
            # Handle Data prerview tasks
            if prerview_option == "Data Preview":
                data_view(st.session_state.df)
            elif prerview_option == "First 5 Rows":
                st.write(st.session_state.df.head())
            elif prerview_option == "Last 5 Rows":
                st.write(st.session_state.df.tail())
            elif prerview_option == "10 Rows":
                st.write(st.session_state.df.head(10))
            elif prerview_option == "20 Rows":
                st.write(st.session_state.df.head(20))
            elif prerview_option == "50 Rows":
                st.write(st.session_state.df.head(50))
            elif prerview_option == "Sample Data":
                sample_data(st.session_state.df)
        

        # Data Overview Section (Initially hidden)
        data_overview_expander = st.sidebar.expander("Data Overview", expanded=False)
        if data_overview_expander:
            overview_option = st.sidebar.selectbox("Data Overview", [
                "Select Option",
                "Data info",
                "Statistics Summary",
                "Show Value Counts",
                "Cross Tabulation",
                "Correlation",
                "Skewness & Kurtosis",
                "Numerical & Categorical Columns",
            ])
            
            if overview_option == "Data info":
                st.write(st.session_state.df.head(2))
                summary_info, data_info = get_data_info(st.session_state.df)
                st.write("**DataFrame Summary**")
                st.write(summary_info)
                st.write("**Detailed Column Info**")
                st.write(data_info)
                # Display rows with missing values if any
                rows_with_missing_values = st.session_state.df[st.session_state.df.isnull().any(axis=1)]
                if not rows_with_missing_values.empty:
                    st.write("**Rows with Missing Values**")
                    st.write(rows_with_missing_values)
                    if st.button("Remove Missing Values"):
                        st.session_state.df, missing_values = remove_missing_values(st.session_state.df)
                        st.success("Rows with missing values removed!")
                        st.write(missing_values)
                        st.rerun()  # Refresh the page to reflect changes
                else:
                    st.write("**No rows with missing values found.**")

                # Display duplicate rows if any
                duplicate_rows = st.session_state.df[st.session_state.df.duplicated()]
                if not duplicate_rows.empty:
                    st.write("**Duplicate Rows**")
                    st.write(duplicate_rows)
                    if st.button("Remove Duplicates"):
                        st.session_state.df = remove_duplicates(st.session_state.df)  # Reassign the cleaned DataFrame back to `df`
                        st.success("Duplicate rows removed!")
                        st.write(check_duplicates(st.session_state.df))  # Display updated DataFrame
                        st.rerun()  # Refresh the page to reflect changes
                else:
                    st.write("**No duplicate rows found.**")

            # Handle Data Overview tasks
            elif overview_option == "Statistics Summary":
                statistics(st.session_state.df)
            
            elif overview_option == "Show Value Counts":
                column = st.selectbox("Select a column to show value counts:", st.session_state.df.columns)
                show_value_counts(st.session_state.df, column)
                    
            elif overview_option == "Numerical & Categorical Columns":
                column_type = st.radio("Select column type:", ['Numerical', 'Categorical'])
                st.write(f"### {column_type} Columns Data")
                st.write(show_column_data(st.session_state.df, column_type))
                st.write(column_selection(st.session_state.df))

            elif overview_option == "Cross Tabulation":
                col1 = st.selectbox("Select the first column for cross-tabulation:", st.session_state.df.columns)
                col2 = st.selectbox("Select the second column for cross-tabulation:", st.session_state.df.columns)
                cross_tabulation(st.session_state.df, col1, col2)

            elif overview_option == "Correlation":
                st.write("### Correlation Matrix")
                corr_matrix = correlation(st.session_state.df)
                st.write(corr_matrix)
                st.write("### Heatmap")
                create_heatmap(st.session_state.df)
                st.write("### Observations")
                observations = correlation_observations(corr_matrix)
                for observation in observations:
                    st.write(observation)

            elif overview_option == "Skewness & Kurtosis":
                st.write("### Skewness & Kurtosis")
                st.write(skewness_kurtosis(st.session_state.df))
                plot_skewness_kurtosis(st.session_state.df)

        # Data Cleaning Section (Initially hidden)
        data_cleaning_expander = st.sidebar.expander("Data Cleaning", expanded=False)
        if data_cleaning_expander:
            cleaning_option = st.sidebar.selectbox("Data Cleaning", [
                "Select Option",
                "Change Data Type",
                "Fill Missing Values",
                "Drop Columns",
                "Replace Values",
                "Clean Categorical Text",
                "Encode Categorical Columns",
            ])  
            
            # Handle Data Cleaning tasks
            if cleaning_option == "Change Data Type":
                column = st.selectbox("Select a column to change data type:", st.session_state.df.columns)
                new_type = st.selectbox("Select the new data type:", ['int64', 'float64', 'object', 'datetime64[ns]', 'category'])
                
                if st.button("Change Data Type"):
                    try:
                        # Handle conversion based on the new type
                        if new_type == 'datetime64[ns]':
                            # Convert to datetime
                            st.session_state.df[column] = pd.to_datetime(st.session_state.df[column], errors='coerce')
                        elif new_type in ['float64', 'int64']:
                            # Attempt to convert to numeric, forcing errors to NaN
                            st.session_state.df[column] = pd.to_numeric(st.session_state.df[column].str.strip(), errors='coerce')
                        else:
                            # For object and category types, just change the type
                            st.session_state.df[column] = st.session_state.df[column].astype(new_type)

                        st.success(f"Data type for column '{column}' changed to {new_type}!")
                        st.write(st.session_state.df.dtypes)
                    except Exception as e:
                        st.error(f"Error changing data type: {e}")

            elif cleaning_option == "Replace Values":
                column = st.selectbox("Select a column to replace values:", st.session_state.df.columns)

                # Determine the data type of the selected column
                column_dtype = st.session_state.df[column].dtype

                replace_option = st.radio("Do you want to:", ("Select values to replace", "Select values not to replace", "Use custom text input"))

                if replace_option == "Select values to replace":
                    all_values = list(st.session_state.df[column].unique())
                    selected_values = st.multiselect("Select the values you want to replace:", all_values, [])
                    values_to_replace = selected_values
                elif replace_option == "Select values not to replace":
                    all_values = list(st.session_state.df[column].unique())
                    not_selected_values = st.multiselect("Select the values you do not want to replace:", all_values, [])
                    values_to_replace = list(set(all_values) - set(not_selected_values))
                elif replace_option == "Use custom text input":
                    if column_dtype == 'object':
                        old_value = st.text_input("Enter the value to replace:", "old_value")
                        values_to_replace = [old_value]
                    elif column_dtype in ['int64', 'float64']:
                        old_value = st.number_input("Enter the value to replace:", value=0)
                        values_to_replace = [old_value]
                    else:
                        st.warning("Unsupported data type for replacement.")
                        values_to_replace = None

                if column_dtype == 'object':
                    new_value = st.text_input("Enter the new value:", "new_value")
                elif column_dtype in ['int64', 'float64']:
                    new_value = st.number_input("Enter the new value:", value=0)
                else:
                    st.warning("Unsupported data type for replacement.")
                    new_value = None

                if st.button("Replace Values") and values_to_replace is not None and new_value is not None:
                    # Trim whitespace from the column values if dtype is object
                    if column_dtype == 'object':
                        st.session_state.df[column] = st.session_state.df[column].str.strip()

                    # Replace selected values
                    if any(val in st.session_state.df[column].values for val in values_to_replace):
                        st.session_state.df[column] = st.session_state.df[column].replace(dict.fromkeys(values_to_replace, new_value))
                        st.success("Values replaced successfully!")
                        st.write(st.session_state.df[column].head())
                    else:
                        st.warning(f"The values {values_to_replace} do not exist in the selected column.")


            elif cleaning_option == "Drop Columns":
                columns = st.multiselect("Select columns to drop:", st.session_state.df.columns)
                if st.button("Drop Columns"):
                    st.session_state.df = drop_columns(st.session_state.df, columns)
                    st.success("Selected columns dropped!")
                    st.write(st.session_state.df.head())


            elif cleaning_option == "Clean Categorical Text":
                categorical_columns = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
                columns = st.multiselect("Select categorical columns to clean:", categorical_columns)
                if st.button("Clean Text"):
                    st.session_state.df = clean_categorical_text(st.session_state.df, columns)
                    st.success("Text cleaned for selected columns!")
                    st.write(st.session_state.df.head())

            elif cleaning_option == "Encode Categorical Columns":
                categorical_columns = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
                columns = st.multiselect("Select categorical columns to encode:", categorical_columns)
                if st.button("Encode Columns"):
                    st.session_state.df = encode_categorical(st.session_state.df, columns)
                    st.success("Selected columns encoded!")
                    st.write(unique_columns(st.session_state.df))
            
            elif cleaning_option == "Fill Missing Values":
                column = st.selectbox("Select the column with missing values", st.session_state.df.columns)
                
                # Check if the selected column is numerical
                if st.session_state.df[column].dtype in ['int64', 'float64']:
                    # Show statistics of the selected column (mean, median, min, max)
                    mean_value = st.session_state.df[column].mean()
                    median_value = st.session_state.df[column].median()
                    min_value = st.session_state.df[column].min()
                    max_value = st.session_state.df[column].max()
                    
                    st.write(f"Mean: {mean_value}")
                    st.write(f"Median: {median_value}")
                    st.write(f"Min: {min_value}")
                    st.write(f"Max: {max_value}")
                    
                    method = st.selectbox(
                        "Select the method to fill missing values",
                        ['Mean', 'Median', 'Min', 'Max', 'Zero', 'Custom Value']
                    )
                    
                    if method == 'Custom Value':
                        custom_value = st.number_input(f"Enter a custom value to fill missing values in {column}", value=0)
                        method = 'Custom'
                
                # Check if the selected column is categorical
                elif st.session_state.df[column].dtype in ['object', 'category']:
                    # Show the most frequent category (mode) and its count
                    mode_value = st.session_state.df[column].mode()[0]
                    mode_count = st.session_state.df[column].value_counts()[mode_value]
                    
                    st.write(f"Most frequent in {column}: {mode_value} (Count: {mode_count})")
                    
                    method = st.selectbox(
                        "Select the method to fill missing values",
                        ['Most Frequent', 'Other', 'Custom Value']
                    )
                    
                    if method == 'Custom Value':
                        custom_value = st.text_input(f"Enter a custom value to fill missing values in {column}", value="Unknown")
                        method = 'Custom'
                else:
                    st.write("Selected column is neither numerical nor categorical.")
                    method = None  # No filling method is available
                
                # Apply the filling method
                if method is not None and st.button("Fill Missing Values"):
                    if method == 'Mean':
                        st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mean())
                    elif method == 'Median':
                        st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].median())
                    elif method == 'Min':
                        st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].min())
                    elif method == 'Max':
                        st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].max())
                    elif method == 'Zero':
                        st.session_state.df[column] = st.session_state.df[column].fillna(0)
                    elif method == 'Most Frequent':
                        st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mode()[0])
                    elif method == 'Other':
                        st.session_state.df[column] = st.session_state.df[column].fillna("Other")
                    elif method == 'Custom':
                        st.session_state.df[column] = st.session_state.df[column].fillna(custom_value)
                    
                    st.write("Missing values filled successfully!")
                    st.write(st.session_state.df[column].head())  # Show a preview of the filled column
                    
                    # Refresh the page to reflect the changes
                    st.rerun()

# visualisation
        graph_option = st.sidebar.toggle("Switch to Plotly")
        if graph_option:
            visual_expander = st.sidebar.expander("Visualization", expanded=False)       
            Visual_option = st.sidebar.selectbox("Select a task for Visualization", [
                "Select Option","Pair Plot", "Bar Plot", "Correlation Heatmap", "Scatter Plot", "Histogram","Line Chart",
                "Pie Chart","Box Plot","Count Plot","KDE Plot","Skewness & Kurtosis",
            ])
            if Visual_option == "Pair Plot":
                st.header("Pair Plot")
                create_pairplot(st.session_state.df)

            elif Visual_option == "Bar Plot":
                st.header("Bar Plot")
                create_bar_plot(st.session_state.df)

            elif Visual_option == "Correlation Heatmap":
                st.header("Correlation Heatmap")
                create_heatmap(st.session_state.df)

            elif Visual_option == "Scatter Plot":
                st.header("Scatter Plot")
                create_scatter(st.session_state.df)

            elif Visual_option == "Histogram":
                st.header("Histogram")
                st.write(create_histogram(st.session_state.df))

            elif Visual_option == "Line Chart":
                st.header("Line Chart")
                create_line_plot(st.session_state.df)

            elif Visual_option == "Pie Chart":
                st.header("Pie Chart")
                create_pie_chart(st.session_state.df)

            elif Visual_option == "Box Plot":
                st.header("Box Plot")
                create_boxplot(st.session_state.df)

            elif Visual_option == "Count Plot":
                st.header("Count Plot")
                create_count_plot(st.session_state.df)

            elif Visual_option == "KDE Plot":
                st.header("KDE Plot")
                create_kde_plot(st.session_state.df)

        else:


            Visual_option = st.sidebar.selectbox("Select a task for Visualization", [
                    "Select Option","Pair Plot", "Bar Plot", "Correlation Heatmap", "Scatter Plot", "Histogram","Line Chart",
                    "Pie Chart","Box Plot", "Count Plot", "Distribution Plot", 
                ])
            if Visual_option == "Pair Plot":
                st.header("Pair Plot")
                st.write(mat_create_pairplot(st.session_state.df))

            elif Visual_option == "Bar Plot":
                st.header("Bar Plot")
                fig = mat_create_bar_plot(st.session_state.df)
                st.pyplot(fig)

            elif Visual_option == "Correlation Heatmap":
                fig = mat_create_heatmap(st.session_state.df)
                if fig is not None:
                    st.pyplot(fig)

            elif Visual_option == "Scatter Plot":
                st.write(mat_create_scatter(st.session_state.df))

            elif Visual_option == "Histogram":
                st.header("Histogram")
                st.write(mat_create_histplot(st.session_state.df))

            elif Visual_option == "Line Chart":
                st.header("Line Chart")
                st.write(mat_create_line_plot(st.session_state.df))

            elif Visual_option == "Pie Chart":
                st.header("Pie Chart")
                st.write(mat_create_pie_chart(st.session_state.df))

            elif Visual_option == "Box Plot":
                st.header("Box Plot")
                st.write(mat_create_boxplot(st.session_state.df))

            elif Visual_option == "Count Plot":
                st.header("Count Plot")
                st.write(mat_create_count_plot(st.session_state.df))

            elif Visual_option == "Distribution Plot":
                st.header("Distribution Plot")
                st.write(mat_create_kde_plot(st.session_state.df))


# model building

        Model_Building_expander = st.sidebar.expander("Model Building", expanded=False)
        if Model_Building_expander:
            Model_Building_option = st.sidebar.selectbox("Machine Learning & Testing", [
                "Select Option","Model Building", "Test Model"
            ])  
            
            # Handle Data Cleaning tasks
            if Model_Building_option == "Model Building":
                model_type = st.selectbox("Choose Model Type", ["Regression", "Classification"])
                X_train, X_test, y_train, y_test = split_data(st.session_state.df, model_type)
    
                if X_train is not None:
                    col_trans = column_transformation(X_train)

                    if col_trans is not None:
                        st.success("Column transformation successful. Proceed to model selection.")
                        
                        if model_type == "Regression":
                            regression_model(X_train, X_test, y_train, y_test, col_trans)
                        else:
                            classification_models(X_train, X_test, y_train, y_test, col_trans)
            

            if Model_Building_option == "Test Model":
                st.title("Test Your Existing Model Here")

                # Upload model file
                st.subheader("Upload Model File")
                model_file = st.file_uploader("Upload a trained model (Pickle or Joblib)", type=["pkl", "joblib"])

                pipe = None  # Initialize model variable

                if model_file is not None:
                    try:
                        # Load the model from Pickle or Joblib
                        if model_file.name.endswith(".pkl"):
                            pipe  = pickle.load(model_file)
                        else:
                            pipe  = joblib.load(model_file)

                        # Ensure it's a valid model with a predict method
                        if not hasattr(pipe ,"predict"):
                            st.error("Invalid model! The uploaded file does not have a 'predict' method.")
                            st.stop()

                        st.success("Model uploaded successfully! Now, upload a dataset for testing.")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        pipe = None  # Reset model

                # Upload dataset file
                st.subheader("Upload Dataset File")
                dataset_file = st.file_uploader("Upload a dataset file (CSV, Excel)", type=["csv", "xlsx"])

                if dataset_file is not None and pipe is not None:
                    try:
                        # Load dataset
                        if dataset_file.name.endswith(".csv"):
                            df = pd.read_csv(dataset_file)
                        else:
                            df = pd.read_excel(dataset_file)

                        st.success("Dataset uploaded successfully!")
                        st.dataframe(df.head())

                        # Select Target Column
                        target_column = st.selectbox("Select the Target Column (y) to Remove", df.columns)
                        X = df.drop(columns=[target_column])  # Features only
                        st.write("Feature Columns:")
                        st.dataframe(X.head())

                        # Ensure correct input format
                        st.subheader("Make a Prediction")
                        input_data = {}

                        for idx, col in enumerate(X.columns):
                            if X[col].dtype == 'object':  # Categorical input
                                input_data[col] = st.selectbox(f"Select value for {col}", options=X[col].unique(), key=f"select_{idx}")
                            else:  # Numerical input
                                input_data[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()), key=f"num_{idx}")

                        # Convert input data to DataFrame
                        query = pd.DataFrame([input_data])

                        # Ensure feature count matches the model's expectation
                        if query.shape[1] != X.shape[1]:
                            st.error(f"Feature mismatch: Model expects {X.shape[1]} features but received {query.shape[1]}")
                            st.stop()

                        if st.button("Predict"):
                            try:
                                predicted = pipe.predict(query)
                                st.success(f"The predicted value is: {predicted[0]}")
                            except Exception as e:
                                st.error(f"Prediction error: {e}")

                    except Exception as e:
                        st.error(f"Error processing dataset: {e}")

st.sidebar.markdown("________________________")

# about page
if st.sidebar.button("About"):
    about()
# Contact page
if st.sidebar.button("Contact"):
    contact()
    st.write("If you have any questions or feedback, please reach out to us.")

