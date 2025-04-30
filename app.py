import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import os
import base64
from io import BytesIO
import streamlit.components.v1 as components
import Upload_Page
# Set page configuration
st.set_page_config(
    page_title="ML Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue and white theme
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #f0f8ff;
        color: #0a4c6d;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e88e5;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0d47a1;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0d47a1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #1976d2;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Success message */
    .success-message {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 10px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
    }
    
    /* Card-like containers */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1976d2;
    }
    
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e3f2fd;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: #1976d2;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1976d2 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

# Sidebar navigation
st.sidebar.title("ML Project")
st.sidebar.image("https://img.icons8.com/fluency/96/machine-learning.png", width=60)

pages = {
    1: "File Upload",
    2: "Data Visualization",
    3: "Preprocessing",
    4: "Model Selection",
    5: "Model Evaluation"
}

# Navigation function
def navigate_to(page):
    st.session_state.page = page

# Sidebar navigation buttons
for page_num, page_name in pages.items():
    if st.sidebar.button(f"{page_num}. {page_name}", key=f"nav_{page_num}"):
        navigate_to(page_num)

# Display current page
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Page:** {pages[st.session_state.page]}")

# Page 1: File Upload
if st.session_state.page == 1:
    st.title("File Upload")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload your dataset")
        st.markdown("Please upload a CSV or Excel file to begin the analysis.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                file_extension = uploaded_file.name.split(".")[-1]
                
                if file_extension.lower() == "csv":
                    data = pd.read_csv(uploaded_file)
                elif file_extension.lower() in ["xlsx", "xls"]:
                    data = pd.read_excel(uploaded_file)
                
                # Store data in session state
                st.session_state.data = data
                
                # Display success message
                st.markdown('<div class="success-message">', unsafe_allow_html=True)
                st.success(f"✅ File '{uploaded_file.name}' successfully uploaded!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display data preview
                st.markdown("### Data Preview")
                st.dataframe(data.head())
                
                # Display data information
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Dataset Shape")
                    st.markdown(f"**Rows:** {data.shape[0]}")
                    st.markdown(f"**Columns:** {data.shape[1]}")
                
                with col2:
                    st.markdown("#### Data Types")
                    dtypes_df = pd.DataFrame(data.dtypes, columns=["Data Type"])
                    dtypes_df = dtypes_df.reset_index().rename(columns={"index": "Column"})
                    st.dataframe(dtypes_df)
                
                # Next page button
                if st.button("Proceed to Data Visualization →"):
                    navigate_to(2)
                    
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Data Visualization
elif st.session_state.page == 2:
    st.title("Data Visualization")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Explore Your Data")
            st.markdown("Select columns to visualize and explore patterns in your dataset.")
            
            # Column selection
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Tabs for different visualization types
            viz_tabs = st.tabs(["Numerical Analysis", "Categorical Analysis", "Relationships"])
            
            with viz_tabs[0]:
                if len(numerical_cols) > 0:
                    st.markdown("#### Numerical Data Analysis")
                    
                    # Select column for histogram
                    num_col = st.selectbox("Select a numerical column for histogram", numerical_cols)
                    
                    # Create histogram
                    fig = px.histogram(data, x=num_col, title=f"Distribution of {num_col}", 
                                      color_discrete_sequence=['#1976d2'])
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=50, b=50, l=50, r=50),
                        xaxis_title=num_col,
                        yaxis_title="Count",
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    st.markdown("#### Summary Statistics")
                    stats_df = data[numerical_cols].describe().T
                    st.dataframe(stats_df)
                    
                    # Box plot
                    st.markdown("#### Box Plot")
                    box_col = st.selectbox("Select a numerical column for box plot", numerical_cols)
                    fig = px.box(data, y=box_col, title=f"Box Plot of {box_col}", 
                                color_discrete_sequence=['#1976d2'])
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=50, b=50, l=50, r=50),
                        yaxis_title=box_col,
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numerical columns found in the dataset.")
            
            with viz_tabs[1]:
                if len(categorical_cols) > 0:
                    st.markdown("#### Categorical Data Analysis")
                    
                    # Select column for bar chart
                    cat_col = st.selectbox("Select a categorical column", categorical_cols)
                    
                    # Create bar chart
                    value_counts = data[cat_col].value_counts().reset_index()
                    value_counts.columns = [cat_col, 'Count']
                    
                    fig = px.bar(value_counts, x=cat_col, y='Count', title=f"Distribution of {cat_col}",
                                color_discrete_sequence=['#1976d2'])
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=50, b=50, l=50, r=50),
                        xaxis_title=cat_col,
                        yaxis_title="Count",
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Pie chart
                    st.markdown("#### Pie Chart")
                    fig = px.pie(value_counts, values='Count', names=cat_col, title=f"Distribution of {cat_col}",
                                color_discrete_sequence=px.colors.sequential.Blues)
                    fig.update_layout(
                        margin=dict(t=50, b=50, l=50, r=50),
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categorical columns found in the dataset.")
            
            with viz_tabs[2]:
                if len(numerical_cols) >= 2:
                    st.markdown("#### Relationships Between Variables")
                    
                    # Scatter plot
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Select X-axis column", numerical_cols)
                    with col2:
                        y_col = st.selectbox("Select Y-axis column", numerical_cols, index=min(1, len(numerical_cols)-1))
                    
                    fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}",
                                    color_discrete_sequence=['#1976d2'])
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=50, b=50, l=50, r=50),
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation heatmap
                    st.markdown("#### Correlation Heatmap")
                    corr = data[numerical_cols].corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues',
                                   title="Correlation Matrix")
                    fig.update_layout(
                        margin=dict(t=50, b=50, l=50, r=50),
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least two numerical columns to plot relationships.")
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to File Upload"):
                    navigate_to(1)
            with col2:
                if st.button("Proceed to Preprocessing →"):
                    navigate_to(3)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please upload a dataset first.")
        if st.button("Go to File Upload"):
            navigate_to(1)

# Page 3: Preprocessing
elif st.session_state.page == 3:
    st.title("Data Preprocessing")
    
    if st.session_state.data is not None:
        data = st.session_state.data.copy()
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Preprocess Your Data")
            st.markdown("Apply various preprocessing techniques to prepare your data for machine learning.")
            
            # Tabs for different preprocessing steps
            preprocess_tabs = st.tabs(["Target Selection", "Missing Values", "Feature Scaling", "Encoding"])
            
            with preprocess_tabs[0]:
                st.markdown("#### Select Target Variable")
                st.markdown("Choose the column you want to predict (target variable).")
                
                all_columns = data.columns.tolist()
                target_column = st.selectbox("Select target column", all_columns)
                
                # Determine problem type
                if target_column:
                    if data[target_column].dtype in ['int64', 'float64']:
                        unique_values = data[target_column].nunique()
                        if unique_values <= 10:  # Arbitrary threshold for classification
                            problem_type = st.radio("Problem type", ["Classification", "Regression"], index=0)
                        else:
                            problem_type = st.radio("Problem type", ["Classification", "Regression"], index=1)
                    else:
                        problem_type = "Classification"
                    
                    st.session_state.target_column = target_column
                    st.session_state.problem_type = problem_type
                    
                    st.markdown(f"Selected target: **{target_column}**")
                    st.markdown(f"Problem type: **{problem_type}**")
            
            with preprocess_tabs[1]:
                st.markdown("#### Handle Missing Values")
                
                # Check for missing values
                missing_data = data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    st.markdown("Columns with missing values:")
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Values': missing_data.values,
                        'Percentage': (missing_data.values / len(data) * 100).round(2)
                    })
                    st.dataframe(missing_df)
                    
                    # Missing value handling strategy
                    st.markdown("#### Select Strategy for Handling Missing Values")
                    
                    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    # For numerical columns
                    if any(col in numerical_cols for col in missing_data.index):
                        st.markdown("**For Numerical Columns:**")
                        num_strategy = st.radio(
                            "Strategy for numerical columns",
                            ["Remove rows", "Mean imputation", "Median imputation", "Zero imputation"],
                            index=1
                        )
                    
                    # For categorical columns
                    if any(col in categorical_cols for col in missing_data.index):
                        st.markdown("**For Categorical Columns:**")
                        cat_strategy = st.radio(
                            "Strategy for categorical columns",
                            ["Remove rows", "Mode imputation", "Create 'Missing' category"],
                            index=1
                        )
                    
                    # Apply missing value handling
                    if st.button("Apply Missing Value Handling"):
                        # For numerical columns
                        if any(col in numerical_cols for col in missing_data.index):
                            if num_strategy == "Remove rows":
                                num_missing_cols = [col for col in missing_data.index if col in numerical_cols]
                                data = data.dropna(subset=num_missing_cols)
                            elif num_strategy == "Mean imputation":
                                for col in [col for col in missing_data.index if col in numerical_cols]:
                                    data[col].fillna(data[col].mean(), inplace=True)
                            elif num_strategy == "Median imputation":
                                for col in [col for col in missing_data.index if col in numerical_cols]:
                                    data[col].fillna(data[col].median(), inplace=True)
                            elif num_strategy == "Zero imputation":
                                for col in [col for col in missing_data.index if col in numerical_cols]:
                                    data[col].fillna(0, inplace=True)
                        
                        # For categorical columns
                        if any(col in categorical_cols for col in missing_data.index):
                            if cat_strategy == "Remove rows":
                                cat_missing_cols = [col for col in missing_data.index if col in categorical_cols]
                                data = data.dropna(subset=cat_missing_cols)
                            elif cat_strategy == "Mode imputation":
                                for col in [col for col in missing_data.index if col in categorical_cols]:
                                    data[col].fillna(data[col].mode()[0], inplace=True)
                            elif cat_strategy == "Create 'Missing' category":
                                for col in [col for col in missing_data.index if col in categorical_cols]:
                                    data[col].fillna("Missing", inplace=True)
                        
                        # Display success message
                        st.markdown('<div class="success-message">', unsafe_allow_html=True)
                        st.success("✅ Missing values handled successfully!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Update missing values count
                        new_missing = data.isnull().sum()
                        new_missing = new_missing[new_missing > 0]
                        
                        if len(new_missing) > 0:
                            st.markdown("Remaining columns with missing values:")
                            new_missing_df = pd.DataFrame({
                                'Column': new_missing.index,
                                'Missing Values': new_missing.values,
                                'Percentage': (new_missing.values / len(data) * 100).round(2)
                            })
                            st.dataframe(new_missing_df)
                        else:
                            st.markdown("No missing values remaining in the dataset.")
                else:
                    st.markdown("No missing values found in the dataset.")
            
            with preprocess_tabs[2]:
                st.markdown("#### Feature Scaling")
                st.markdown("Normalize or standardize numerical features.")
                
                numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                # Remove target column from scaling if it's numerical
                if st.session_state.target_column in numerical_cols:
                    numerical_cols.remove(st.session_state.target_column)
                
                if len(numerical_cols) > 0:
                    # Select columns for scaling
                    cols_to_scale = st.multiselect("Select columns to scale", numerical_cols, default=numerical_cols)
                    
                    if cols_to_scale:
                        # Scaling method
                        scaling_method = st.radio(
                            "Select scaling method",
                            ["StandardScaler (mean=0, std=1)", "MinMaxScaler (range 0-1)"],
                            index=0
                        )
                        
                        # Apply scaling
                        if st.button("Apply Scaling"):
                            if scaling_method == "StandardScaler (mean=0, std=1)":
                                scaler = StandardScaler()
                                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                            else:  # MinMaxScaler
                                scaler = MinMaxScaler()
                                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                            
                            # Display success message
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.success(f"✅ {scaling_method.split(' ')[0]} applied successfully!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show preview of scaled data
                            st.markdown("#### Preview of Scaled Data")
                            st.dataframe(data[cols_to_scale].head())
                else:
                    st.info("No numerical columns available for scaling.")
            
            with preprocess_tabs[3]:
                st.markdown("#### Categorical Encoding")
                st.markdown("Convert categorical variables to numerical format.")
                
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Remove target column from encoding if it's categorical
                if st.session_state.target_column in categorical_cols:
                    categorical_cols.remove(st.session_state.target_column)
                
                if len(categorical_cols) > 0:
                    # Select columns for encoding
                    cols_to_encode = st.multiselect("Select columns to encode", categorical_cols, default=categorical_cols)
                    
                    if cols_to_encode:
                        # Encoding method
                        encoding_method = st.radio(
                            "Select encoding method",
                            ["Label Encoding", "One-Hot Encoding"],
                            index=0
                        )
                        
                        # Apply encoding
                        if st.button("Apply Encoding"):
                            if encoding_method == "Label Encoding":
                                for col in cols_to_encode:
                                    le = LabelEncoder()
                                    data[col] = le.fit_transform(data[col])
                            else:  # One-Hot Encoding
                                data = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)
                            
                            # Display success message
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.success(f"✅ {encoding_method} applied successfully!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show preview of encoded data
                            st.markdown("#### Preview of Encoded Data")
                            st.dataframe(data.head())
                else:
                    st.info("No categorical columns available for encoding.")
            
            # Save preprocessed data
            if st.button("Complete Preprocessing"):
                st.session_state.preprocessed_data = data
                
                # Display success message
                st.markdown('<div class="success-message">', unsafe_allow_html=True)
                st.success("✅ Preprocessing completed successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show preview of final preprocessed data
                st.markdown("#### Final Preprocessed Data")
                st.dataframe(data.head())
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Data Visualization"):
                    navigate_to(2)
            with col2:
                if st.button("Proceed to Model Selection →"):
                    navigate_to(4)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please upload a dataset first.")
        if st.button("Go to File Upload"):
            navigate_to(1)

# Page 4: Model Selection
elif st.session_state.page == 4:
    st.title("Model Selection")
    
    if st.session_state.preprocessed_data is not None:
        data = st.session_state.preprocessed_data
        target_column = st.session_state.target_column
        problem_type = st.session_state.problem_type
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Select a Machine Learning Model")
            st.markdown(f"Choose a suitable model for your {problem_type.lower()} problem.")
            
            # Prepare features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Train-test split
            test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
            random_state = st.number_input("Random state", 0, 100, 42)
            
            # Model selection based on problem type
            if problem_type == "Classification":
                model_options = ["Logistic Regression", "Random Forest", "Support Vector Machine"]
            else:  # Regression
                model_options = ["Linear Regression", "Random Forest", "Support Vector Machine"]
            
            selected_model = st.selectbox("Select a model", model_options)
            
            # Model hyperparameters
            st.markdown("#### Model Hyperparameters")
            
            if selected_model == "Logistic Regression" or selected_model == "Linear Regression":
                C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                max_iter = st.slider("Maximum iterations", 100, 1000, 100)
                
                if problem_type == "Classification":
                    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
                else:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    
            elif selected_model == "Random Forest":
                n_estimators = st.slider("Number of trees", 10, 200, 100)
                max_depth = st.slider("Maximum depth", 1, 20, 10)
                
                if problem_type == "Classification":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                    
            elif selected_model == "Support Vector Machine":
                C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0)
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                
                if problem_type == "Classification":
                    model = SVC(C=C, kernel=kernel, random_state=random_state, probability=True)
                else:
                    model = SVR(C=C, kernel=kernel)
            
            # Train model
            if st.button("Train Model"):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Train model with progress bar
                with st.spinner(f"Training {selected_model}..."):
                    model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                st.session_state.predictions = y_pred
                st.session_state.model = model
                
                # Calculate metrics
                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    st.session_state.metrics = {
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1
                    }
                else:  # Regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.session_state.metrics = {
                        "Mean Squared Error": mse,
                        "Root Mean Squared Error": rmse,
                        "R² Score": r2
                    }
                
                # Display success message
                st.markdown('<div class="success-message">', unsafe_allow_html=True)
                st.success(f"✅ {selected_model} trained successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display metrics preview
                st.markdown("#### Model Performance Preview")
                
                metrics_cols = st.columns(len(st.session_state.metrics))
                for i, (metric_name, metric_value) in enumerate(st.session_state.metrics.items()):
                    with metrics_cols[i]:
                        st.markdown(f"<div class='metric-container'><div class='metric-value'>{metric_value:.4f}</div><div class='metric-label'>{metric_name}</div></div>", unsafe_allow_html=True)
                
                st.markdown("For detailed evaluation, proceed to the Model Evaluation page.")
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Preprocessing"):
                    navigate_to(3)
            with col2:
                if st.button("Proceed to Model Evaluation →"):
                    navigate_to(5)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please complete the preprocessing step first.")
        if st.button("Go to Preprocessing"):
            navigate_to(3)

# Page 5: Model Evaluation
elif st.session_state.page == 5:
    st.title("Model Evaluation")
    
    if st.session_state.model is not None:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        predictions = st.session_state.predictions
        metrics = st.session_state.metrics
        problem_type = st.session_state.problem_type
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Model Performance Evaluation")
            
            # Display metrics
            st.markdown("#### Performance Metrics")
            
            metrics_cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                with metrics_cols[i]:
                    st.markdown(f"<div class='metric-container'><div class='metric-value'>{metric_value:.4f}</div><div class='metric-label'>{metric_name}</div></div>", unsafe_allow_html=True)
            
            # Visualizations based on problem type
            if problem_type == "Classification":
                st.markdown("#### Classification Results Visualization")
                
                # Confusion Matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, predictions)
                
                # Get unique classes
                classes = np.unique(np.concatenate([y_test, predictions]))
                
                fig = px.imshow(cm, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=classes, y=classes,
                               text_auto=True,
                               color_continuous_scale='Blues',
                               title="Confusion Matrix")
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(t=50, b=50, l=50, r=50),
                    font=dict(family="Arial", size=12, color="#333333")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve for binary classification
                if len(classes) == 2:
                    from sklearn.metrics import roc_curve, auc
                    
                    try:
                        # Get probability predictions
                        y_prob = model.predict_proba(X_test)[:, 1]
                        
                        # Calculate ROC curve
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        # Plot ROC curve
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                        
                        fig.update_layout(
                            title='Receiver Operating Characteristic (ROC) Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=50, b=50, l=50, r=50),
                            font=dict(family="Arial", size=12, color="#333333")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("ROC curve could not be generated for this model.")
                
                # Feature importance for Random Forest
                if hasattr(model, 'feature_importances_'):
                    st.markdown("#### Feature Importance")
                    
                    feature_names = X_test.columns
                    importances = model.feature_importances_
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot feature importance
                    fig = px.bar(
                        x=importances[indices],
                        y=[feature_names[i] for i in indices],
                        orientation='h',
                        title='Feature Importance',
                        labels={'x': 'Importance', 'y': 'Feature'},
                        color=importances[indices],
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=50, b=50, l=50, r=50),
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Regression
                st.markdown("#### Regression Results Visualization")
                
                # Actual vs Predicted
                fig = px.scatter(
                    x=y_test,
                    y=predictions,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title='Actual vs Predicted Values'
                )
                
                # Add perfect prediction line
                min_val = min(min(y_test), min(predictions))
                max_val = max(max(y_test), max(predictions))
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(t=50, b=50, l=50, r=50),
                    font=dict(family="Arial", size=12, color="#333333")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals plot
                residuals = y_test - predictions
                
                fig = px.scatter(
                    x=predictions,
                    y=residuals,
                    labels={'x': 'Predicted', 'y': 'Residuals'},
                    title='Residuals Plot'
                )
                
                # Add horizontal line at y=0
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(t=50, b=50, l=50, r=50),
                    font=dict(family="Arial", size=12, color="#333333")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Residuals distribution
                fig = px.histogram(
                    residuals,
                    title='Residuals Distribution',
                    labels={'value': 'Residuals', 'count': 'Frequency'},
                    color_discrete_sequence=['#1976d2']
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(t=50, b=50, l=50, r=50),
                    font=dict(family="Arial", size=12, color="#333333")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for Random Forest
                if hasattr(model, 'feature_importances_'):
                    st.markdown("#### Feature Importance")
                    
                    feature_names = X_test.columns
                    importances = model.feature_importances_
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot feature importance
                    fig = px.bar(
                        x=importances[indices],
                        y=[feature_names[i] for i in indices],
                        orientation='h',
                        title='Feature Importance',
                        labels={'x': 'Importance', 'y': 'Feature'},
                        color=importances[indices],
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=50, b=50, l=50, r=50),
                        font=dict(family="Arial", size=12, color="#333333")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Navigation button
            if st.button("← Back to Model Selection"):
                navigate_to(4)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please train a model first.")
        if st.button("Go to Model Selection"):
            navigate_to(4)
