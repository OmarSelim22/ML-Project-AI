import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, power_transform
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sklearn.tree as tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import os
import base64
from io import BytesIO
from scipy import stats
from scipy.stats.mstats import winsorize
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
import streamlit.components.v1 as components
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title = "ML Project",
    page_icon = "🧠",
    layout = "wide",
    initial_sidebar_state = "expanded"
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
""", unsafe_allow_html = True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
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
st.sidebar.image("ML_IMG.jpg", width = 100)

pages = {
    1: "Home",
    2: "File Upload",
    3: "Data Visualization",
    4: "Preprocessing",
    5: "ML Models "
}

# Navigation function
def navigate_to(page):
    st.session_state.page = page

# Sidebar navigation buttons
for page_num, page_name in pages.items():
    if st.sidebar.button(f"{page_num}. {page_name}", key = f"nav_{page_num}"):
        navigate_to(page_num)

# Display current page
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Page:** {pages[st.session_state.page]}")

# Page 1: Home
if st.session_state.page == 1:

    st.title("Interactive Machine Learning Project")
    st.markdown("---")
    st.header("Welcome to the Interactive Machine Learning Project!")

    with st.container():
        
        st.markdown("### Overview")
        st.markdown("This interactive application provides a complete workflow for machine learning projects:")
        
        st.markdown("""
                    

        ### 🚀 Key Features:
                    
        📤 **Data Upload and Exploration**
        - Upload your CSV or Excel files
        - View data statistics and information
        - Explore data structure
        
        📊 **Interactive Data Visualization**
        - Create dynamic plots
        - Analyze relationships between variables
        - Generate correlation matrices
        
        🔧 **Data Preprocessing**
        - Handle missing values
        - Normalize/Scale features
        - Encode categorical variables
        
        🤖 **Model Training**
        - Choose from multiple algorithms
        - Split data into training and testing sets
        - Train models with customizable parameters
        
        📈 **Model Evaluation**
        - View performance metrics
        - Analyze confusion matrices
        - Visualize ROC curves
        - Examine feature importance
        

                    

        ### 🤖 Supported Models
        - Model Decision Tree
        - Model Neural Network
        - Model Naive Bayes
        - Model Linear Regression
        - Model k_means
                    



        ### 💡 Getting Started:
        1. Upload your dataset or use one of our sample datasets below
        2. Explore and visualize your data
        3. Preprocess the data as needed
        4. Train your model
        5. Evaluate the results   

        ### 📝 Note
        This dashboard is designed for educational purposes and demonstrates basic machine learning workflows.
        For production use cases, additional considerations may be needed.
             
        """)
        
        
      
        
        # Add an illustrative image
        st.image("ML_IMG.jpg", caption="Interactive Machine Learning" , use_column_width=True)
        st.info("⭐ This Project Developed by Alyaa , Nour , Omar , Zaid , Mohammed ⭐")
        st.markdown("---")


        # Next page button
        if st.button("Get Started the app →"):
            navigate_to(2)


# Page 2: File Upload
if st.session_state.page == 2:
    st.title("File Upload")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload your dataset")
        st.markdown("Please upload a CSV or Excel file to begin the analysis.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], label_visibility="collapsed", key="file_uploader")
                                                                
        # Show currently loaded file if exists
        if st.session_state.uploaded_file is not None:
            st.markdown(f"**Currently loaded file:** {st.session_state.file_name}")
            
            if st.button("Clear current file"):
                st.session_state.uploaded_file = None
                st.session_state.file_name = None
                st.session_state.data = None
                st.rerun()


        if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
            try:
                # Store file info in session state
                st.session_state.uploaded_file = uploaded_file
                st.session_state.file_name = uploaded_file.name
                
                # Read file
                file_extension = uploaded_file.name.split(".")[-1].lower()
                if file_extension == "csv":
                    data = pd.read_csv(uploaded_file)
                elif file_extension in ["xlsx", "xls"]:
                    data = pd.read_excel(uploaded_file)

                elif "data" in st.session_state:
                    st.success(f"✅ File '{uploaded_file.name}' successfully uploaded!")
                if uploaded_file is not None:
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.data = data
                elif "data" in st.session_state:
                    data = st.session_state["data"]
                    st.write("Using previously uploaded data:")
                    st.dataframe(data)

                else:
                    st.warning("Please upload a file to proceed.")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Show data preview if available
        if st.session_state.data is not None:
                st.markdown("### Data Preview")
                st.dataframe(st.session_state.data)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Dataset Shape")
                    data = st.session_state.data
                    shape_data = pd.DataFrame(data.shape, index=["Rows", "Columns"], columns=["Count"])
                    st.dataframe(shape_data)
                
                with col2:
                    st.markdown("#### Data Types")
                    st.session_state.data = data
                    dtypes_data = pd.DataFrame(data.dtypes, columns=["Data Type"])
                    dtypes_data = dtypes_data.reset_index().rename(columns={"index": "Column"})
                    st.dataframe(dtypes_data)

                st.markdown("--- ")
                st.header("Next Steps:")
                st.write("After uploading the data, you can proceed to the next pages from the sidebar to explore, preprocess, and model the data.")

             
        st.markdown('</div>', unsafe_allow_html=True)

# Page 3: Data Visualization
elif st.session_state.page == 3:
    st.title("Data Visualization")

    # Check if data is uploaded
    if st.session_state.data is not None:
        data = st.session_state.data.copy()

        # Analyze column types
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()

        # Display column type information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Numeric Columns", len(numeric_columns))
        with col2:
            st.metric("Categorical Columns", len(categorical_columns))
        with col3:
            st.metric("Datetime Columns", len(datetime_columns))

        st.write("---")  # Separator

        # Main Content Area (Settings Left, Plot Right)
        settings_col, plot_col = st.columns([1, 2])  # Ratio for layout

        # Visualization Settings (Left Column)
        with settings_col:
            st.header("Visualization Settings")

            # Define available chart types
            visualization_types = [
                "Bar Chart",
                "Histogram",
                "Line Plot",
                "Scatter Plot",
                "Box Plot",
                "Pie Chart",
            ]

            viz_type = st.selectbox(
                "Select Visualization Type",
                options=visualization_types,
                index=0  # Default to Bar Chart
            )

            # Dynamic Axis Selection based on Viz Type
            x_axis = None
            y_axis = None
            y_axis_options = ["-- None --"] + numeric_columns + categorical_columns

            if viz_type == "Bar Chart":
                x_options = categorical_columns + numeric_columns
                y_options_specific = numeric_columns
                y_axis_label = "Select Y-axis Column (Value/Aggregation)"
                x_axis_label = "Select X-axis Column (Category)"
                default_x = x_options[0] if x_options else None
                default_y = '-- None --'

            elif viz_type == "Histogram":
                x_options = numeric_columns
                y_options_specific = []
                y_axis_label = "Select Y-axis Column (N/A for Histogram)"
                x_axis_label = "Select X-axis Column (Numeric)"
                default_x = x_options[0] if x_options else None
                default_y = '-- None --'

            elif viz_type == "Line Plot":
                x_options = numeric_columns + datetime_columns
                y_options_specific = numeric_columns
                y_axis_label = "Select Y-axis Column (Value)"
                x_axis_label = "Select X-axis Column (Sequence/Time)"
                default_x = x_options[0] if x_options else None
                default_y = y_options_specific[0] if y_options_specific else '-- None --'

            elif viz_type == "Scatter Plot":
                x_options = numeric_columns
                y_options_specific = numeric_columns
                y_axis_label = "Select Y-axis Column"
                x_axis_label = "Select X-axis Column"
                default_x = x_options[0] if x_options else None
                default_y = x_options[1] if len(x_options) > 1 else '-- None --'

            elif viz_type == "Box Plot":
                x_options = ["-- None --"] + categorical_columns
                y_options_specific = numeric_columns
                y_axis_label = "Select Y-axis Column (Numeric Value)"
                x_axis_label = "Select X-axis Column (Category - Optional)"
                default_x = '-- None --'
                default_y = y_options_specific[0] if y_options_specific else '-- None --'

            elif viz_type == "Pie Chart":
                x_options = categorical_columns
                y_options_specific = numeric_columns
                y_axis_label = "Select Values Column (Numeric)"
                x_axis_label = "Select Names Column (Category)"
                default_x = x_options[0] if x_options else None
                default_y = y_options_specific[0] if y_options_specific else '-- None --'

            else:
                x_options = numeric_columns + categorical_columns + datetime_columns
                y_options_specific = numeric_columns
                y_axis_label = "Select Y-axis Column (optional)"
                x_axis_label = "Select X-axis Column"
                default_x = x_options[0] if x_options else None
                default_y = '-- None --'

            # Render Select Boxes
            x_axis = st.selectbox(x_axis_label, x_options, index=0 if default_x is None else x_options.index(default_x))
            y_axis = st.selectbox(
                y_axis_label,
                y_axis_options,
                index=0 if default_y is None else y_axis_options.index(default_y),
                disabled=(viz_type == "Histogram")
            )

            # Chart Title
            chart_title = st.text_input("Chart Title", f"{viz_type} of {x_axis}")

            # Chart Height
            chart_height = st.slider("Chart Height", min_value=300, max_value=1000, value=450)

            # Create Button
            create_chart = st.button("Create Visualization", type="primary", use_container_width=True)

        # Plotting Logic (Right Column)
        with plot_col:
            st.subheader("Visualization")

            fig = None
            if create_chart and x_axis:
                try:
                    plot_y_axis = None if y_axis == "-- None --" else y_axis

                    if viz_type == "Bar Chart":
                        if plot_y_axis:
                            grouped_data = data.groupby(x_axis, as_index=False)[plot_y_axis].mean()
                            fig = px.bar(grouped_data, x=x_axis, y=plot_y_axis)
                        else:
                            count_data = data[x_axis].value_counts().reset_index()
                            count_data.columns = [x_axis, 'Count']
                            fig = px.bar(count_data, x=x_axis, y='Count')

                    elif viz_type == "Histogram":
                        fig = px.histogram(data, x=x_axis, nbins=50)

                    elif viz_type == "Line Plot":
                        if plot_y_axis:
                            data_sorted = data.sort_values(by=x_axis)
                            fig = px.line(data_sorted, x=x_axis, y=plot_y_axis)

                    elif viz_type == "Scatter Plot":
                        if plot_y_axis:
                            fig = px.scatter(data, x=x_axis, y=plot_y_axis)

                    elif viz_type == "Box Plot":
                        if plot_y_axis:
                            if x_axis != "-- None --":
                                fig = px.box(data, x=x_axis, y=plot_y_axis)
                            else:
                                fig = px.box(data, y=plot_y_axis)

                    elif viz_type == "Pie Chart":
                        if plot_y_axis:
                            pie_data = data.groupby(x_axis, as_index=False)[plot_y_axis].sum()
                            fig = px.pie(pie_data, names=x_axis, values=plot_y_axis)

                    if fig:
                        fig.update_layout(
                            title=chart_title,
                            height=chart_height,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="sans-serif", size=12, color="#333333"),
                            margin=dict(l=40, r=40, t=50, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

            if not create_chart:
                st.info("Select visualization settings and click 'Create Visualization'.")

        # Summary Statistics Section
        st.write("---")
        st.header("Summary Statistics")

        st.subheader("Numeric Data Overview")
        if numeric_columns:
            st.dataframe(data[numeric_columns].describe().T)
        else:
            st.info("No numeric columns for descriptive statistics.")

        st.subheader("Categorical Data Overview")
        if categorical_columns:
            st.dataframe(data[categorical_columns].describe().T)
        else:
            st.info("No categorical columns for descriptive statistics.")
            # Add section for advanced statistics
    st.markdown("---")
    st.header("Advanced Statistics")

    if numeric_columns:
        # Select numeric column for advanced analysis
        num_col = st.selectbox("Select a numeric column for advanced analysis", numeric_columns)
        
        # Calculate advanced statistics
        stats = {
            "Mean": data[num_col].mean(),
            "Median": data[num_col].median(),
            "Standard Deviation": data[num_col].std(),
            "Variance": data[num_col].var(),
            "Minimum": data[num_col].min(),
            "Maximum": data[num_col].max(),
            "Range": data[num_col].max() - data[num_col].min(),
            "First Quartile (25%)": data[num_col].quantile(0.25),
            "Third Quartile (75%)": data[num_col].quantile(0.75),
            "Interquartile Range": data[num_col].quantile(0.75) - data[num_col].quantile(0.25),
            "Kurtosis": data[num_col].kurtosis(),
            "Skewness": data[num_col].skew()
        }
        
        # Display statistics as a wider table
        st.dataframe(pd.DataFrame(stats.items(), columns=['Statistic', 'Value']).set_index('Statistic'), use_container_width=True)
        
        # Create advanced distribution plot
        fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=(f"Distribution of {num_col}"),
                        vertical_spacing=0.15,
                        row_heights=[0.7, 0.3])
        
    
        # Add distribution plot
        fig.add_trace(
            go.Histogram(x=data[num_col], name="Distribution", nbinsx=30, marker_color='royalblue'),
            row=1, col=1
        )
                
        # Update layout
        fig.update_layout(
            height=500,
            template="plotly_white",
            showlegend=False,
            title_text=f"Advanced Analysis of {num_col}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please upload a big dataset first.")
        if st.button("Go to File Upload"):
            navigate_to(2)


# Page 4: Preprocessing
elif st.session_state.page == 4:

    data = st.session_state.data.copy()
    st.title("Data Preprocessing")
    # Get original data from session state
    
    if st.session_state.data is not None:
        data = st.session_state.data.copy()
        file_name = st.session_state.data.name if hasattr(st.session_state.data, 'name') else "Uploaded Data"
        with st.container():
            st.write(f"Preprocessing data from file: *{file_name}*")

            # Display preview of current data
            st.subheader("Current Data Preview")
            st.dataframe(data)
            st.markdown("#### Dataset Shape")
            data = st.session_state.data
            shape_data = pd.DataFrame(data.shape, index=["Rows", "Columns"], columns=["Count"])
            st.dataframe(shape_data)

            st.markdown("### Preprocess Your Data")
            st.markdown("Apply various preprocessing techniques to prepare your data for machine learning.")
            
            # Tabs for different preprocessing steps
            preprocess_tabs = st.tabs(["Missing Values", "Feature Scaling", "Encoding" , "Outliers","Transformation"])
            
                
            with preprocess_tabs[0]:
                st.markdown("#### Handle Missing Values")
                
                # Check for missing values
                missing_values = data.isnull().sum()
                missing_cols = missing_values[missing_values > 0].index.tolist()
                
                if not missing_cols:
                    st.success("No missing values found in the current dataset!")
                else:
                    st.warning(f"Missing values found in {len(missing_cols)} columns")
                    st.markdown("Columns with missing values:")
                    missing_data = pd.DataFrame({
                        'Column': missing_cols,
                        'Missing Values': data[missing_cols].isnull().sum().values,
                        'Percentage': (data[missing_cols].isnull().sum().values / len(data) * 100).round(2)
                    })
                    st.dataframe(missing_data)
                    
                    # Get column types
                    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    # Separate missing columns by type
                    num_missing = [col for col in missing_cols if col in numerical_cols]
                    cat_missing = [col for col in missing_cols if col in categorical_cols]
                    
                    # Always show strategy selection (remove conditionals)
                    st.markdown("### Select Handling Strategies")
                    
                    if num_missing:
                        st.markdown("**Numerical Columns Handling**")
                        num_strategy = st.radio(
                            "Strategy for numerical columns:",
                            ["Drop rows", "Mean imputation", "Median imputation", "KNN imputation", "Zero imputation"],
                            key="num_strategy"
                        )
                    
                    if cat_missing:
                        st.markdown("**Categorical Columns Handling**")
                        cat_strategy = st.radio(
                            "Strategy for categorical columns:",
                            ["Drop rows", "Mode imputation", "Create 'Missing' category"],
                            key="cat_strategy"
                        )
                    
                    if st.button("Apply Missing Value Handling"):
                        # Handle numerical columns
                        if num_missing:
                            if num_strategy == "Drop rows":
                                data = data.dropna(subset = num_missing)
                            elif num_strategy == "Mean imputation":
                                data[num_missing] = data[num_missing].fillna(data[num_missing].mean())
                            elif num_strategy == "Median imputation":
                                data[num_missing] = data[num_missing].fillna(data[num_missing].median())
                            elif num_strategy == "KNN imputation":
                                from sklearn.impute import KNNImputer
                                imputer = KNNImputer(n_neighbors=5)
                                data[num_missing] = imputer.fit_transform(data[num_missing])
                            elif num_strategy == "Zero imputation":
                                data[num_missing] = data[num_missing].fillna(0)
                        
                        # Handle categorical columns
                        if cat_missing:
                            if cat_strategy == "Drop rows":
                                data = data.dropna(subset=cat_missing)
                            elif cat_strategy == "Mode imputation":
                                for col in cat_missing:
                                    data[col] = data[col].fillna(data[col].mode()[0])
                            elif cat_strategy == "Create 'Missing' category":
                                for col in cat_missing:
                                    data[col] = data[col].fillna("Missing")
                        
                        st.success("Missing values handled successfully!")
                        st.session_state.data = data  # Update the session state                        
                        # Update missing values count
                        new_missing = data.isnull().sum()
                        new_missing = new_missing[new_missing > 0]
                        
                        if len(new_missing) > 0:
                            st.markdown("Remaining columns with missing values:")
                            new_missing_data = pd.DataFrame({
                                'Column': new_missing.index,
                                'Missing Values': new_missing.values,
                                'Percentage': (new_missing.values / len(data) * 100).round(2)
                            })
                            st.dataframe(new_missing_data)
                        else:
                            st.markdown("No missing values remaining in the dataset.")
                    else:
                        st.markdown("No missing values found in the dataset.")
            
            with preprocess_tabs[1]:
                st.markdown("#### Feature Scaling")
                st.markdown("Normalize or standardize numerical features to bring all values into a similar range.")
                
                numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if len(numerical_cols) > 0:
                    # Select columns for scaling
                    cols_to_scale = st.multiselect("Select columns to scale", numerical_cols, default=numerical_cols)
                    
                    if cols_to_scale:
                        # Scaling method
                        scaling_method = st.radio(
                            "Select scaling method",
                            ["StandardScaler (mean=0, std=1)", "MinMaxScaler (range 0-1)", "Z-score Normalization"],
                            index=0
                        )
                        
                        # Apply scaling
                        if st.button("Apply Scaling"):
                            if scaling_method == "StandardScaler (mean=0, std=1)":
                                scaler = StandardScaler()
                                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                            elif scaling_method == "MinMaxScaler (range 0-1)":
                                scaler = MinMaxScaler()
                                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                            elif scaling_method == "Z-score Normalization":
                                data[cols_to_scale] = (data[cols_to_scale] - data[cols_to_scale].mean()) / data[cols_to_scale].std()
                            
                            # Update session state with the processed data
                            st.session_state.data = data.copy()
                            
                            # Display success message
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.success(f"✅ {scaling_method.split(' ')[0]} applied successfully!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show preview of scaled data
                            st.markdown("#### Preview of Scaled Data")
                            st.dataframe(data[cols_to_scale].head())
                else:
                    st.info("No numerical columns available for scaling.")
            with preprocess_tabs[2]:
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
                            
                            # Add this line to update session state immediately:
                            st.session_state.data = data.copy()
                            # Display success message
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.success(f"✅ {encoding_method} applied successfully!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show preview of encoded data
                            st.markdown("#### Preview of Encoded Data")
                            st.dataframe(data.head())
                else:
                    st.info("No categorical columns available for encoding.")
            with preprocess_tabs[3]:
                st.markdown("#### Outlier Detection and Handling")
                st.markdown("Identify and handle outliers in the dataset.")

                # Select columns for outlier detection
                numeric_columns = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                cols_for_outliers = st.multiselect("Select columns for outlier detection", numeric_columns, default=numeric_columns)

                if cols_for_outliers:
                    # Outlier handling method
                    outlier_method = st.radio(
                        "Select outlier handling method",
                        ["Z-Score", "IQR", "Winsorization", "Clipping"],
                        index=0
                    )

                    if outlier_method == "Z-Score":
                        threshold = st.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

                    elif outlier_method == "Winsorization":
                        lower_quantile = st.slider("Lower Quantile", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
                        upper_quantile = st.slider("Upper Quantile", min_value=0.5, max_value=1.0, value=0.95, step=0.01)

                    elif outlier_method == "Clipping":
                        min_value = st.number_input("Minimum Value", value=float(st.session_state.data[cols_for_outliers].min().min()))
                        max_value = st.number_input("Maximum Value", value=float(st.session_state.data[cols_for_outliers].max().max()))

                    if st.button("Handle Outliers"):
                        try:
                            if outlier_method == "Z-Score":
                                def remove_outliers_zscore(data, columns, threshold=3.0):
                                    """
                                    Remove outliers from the dataset using the Z-score method.

                                    Parameters:
                                        data (pd.DataFrame): The dataset to process.
                                        columns (list): List of numeric columns to check for outliers.
                                        threshold (float): The Z-score threshold for identifying outliers.

                                    Returns:
                                        pd.DataFrame: The dataset with outliers removed.
                                    """
                                    for col in columns:
                                        if not pd.api.types.is_numeric_dtype(data[col]):
                                            raise ValueError(f"Z-score outlier detection can only be applied to numeric columns. Column '{col}' is not numeric.")

                                        z_scores = np.abs(stats.zscore(data[col]))
                                        data = data[z_scores < threshold]

                                    return data

                                # Apply Z-score outlier removal
                                data = remove_outliers_zscore(data, cols_for_outliers, threshold=threshold)
                                
                               
                            elif outlier_method == "IQR":
                                Q1 = data[cols_for_outliers].quantile(0.25)
                                Q3 = data[cols_for_outliers].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                outliers = ((data[cols_for_outliers] < lower_bound) | (data[cols_for_outliers] > upper_bound)).any(axis=1)
                                data = data[~outliers]
                            elif outlier_method == "Winsorization":
                                data = winsorize(data, cols_for_outliers, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
                            elif outlier_method == "Clipping":
                                # Define the clipping function
                                def clipping(data, columns, min_value, max_value):
                                    for col in columns:
                                        data[col] = data[col].clip(lower=min_value, upper=max_value)
                                    return data

                                # Apply clipping
                                data = clipping(data, cols_for_outliers, min_value=min_value, max_value=max_value)

                            # Update session state with the processed data
                            st.session_state.data = data.copy()

                            # Display success message
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.success(f"✅ Outliers handled successfully using {outlier_method}!")
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Show preview of updated data
                            st.markdown("#### Updated Data Preview")
                            st.dataframe(st.session_state.data[cols_for_outliers].head())
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
            with preprocess_tabs[4]:
                st.markdown("### Transformation")
                st.markdown("Apply transformations to numerical features to improve data distribution or create new features.")

                # Select columns for transformation
                numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                cols_to_transform = st.multiselect("Select columns for transformation", numeric_columns, default=numeric_columns)

                if cols_to_transform:
                    # Transformation method
                    transformation_method = st.radio(
                        "Select transformation method",
                        ["Log Transformation", "Power Transformation", "Polynomial Transformation"],
                        index=0
                    )

                    if transformation_method == "Power Transformation":
                        power_method = st.selectbox("Select power transformation method", ["yeo-johnson", "box-cox"])
                        if power_method == "box-cox":
                            st.warning("Box-Cox requires all values to be positive.")

                    elif transformation_method == "Polynomial Transformation":
                        degree = st.slider("Select polynomial degree", min_value=2, max_value=5, value=2)
                        interaction_only = st.checkbox("Interaction only (exclude higher-order terms)", value=False)

                    # Apply transformation
                    if st.button("Apply Transformation"):
                        try:
                            if transformation_method == "Log Transformation":
                                # Log Transformation: Useful for reducing skewness in data with positive values
                                def log_transform(data, columns):
                                    """
                                    Apply log transformation to specified columns.

                                    Parameters:
                                        data (pd.DataFrame): The dataset to process.
                                        columns (list): List of numeric columns to apply log transformation.

                                    Returns:
                                        pd.DataFrame: The dataset with log-transformed columns.
                                    """
                                    for col in columns:
                                        # Ensure all values are positive for log transformation
                                        if (data[col] <= 0).any():
                                            raise ValueError(f"Log transformation requires all values in column '{col}' to be positive.")
                                        data[col] = np.log(data[col])
                                    return data

                                # Apply log transformation
                                data = log_transform(data, cols_to_transform)

                            elif transformation_method == "Power Transformation":
                                # Power Transformation: Useful for normalizing data distribution (e.g., Yeo-Johnson or Box-Cox)
                                def apply_power_transform(data, columns, method):
                                    """
                                    Apply power transformation to specified columns.

                                    Parameters:
                                        data (pd.DataFrame): The dataset to process.
                                        columns (list): List of numeric columns to apply power transformation.
                                        method (str): The power transformation method ('yeo-johnson' or 'box-cox').

                                    Returns:
                                        pd.DataFrame: The dataset with power-transformed columns.
                                    """
                                    pt = PowerTransformer(method=method)
                                    data[columns] = pt.fit_transform(data[columns])
                                    return data

                                # Apply power transformation
                                data = apply_power_transform(data, cols_to_transform, method=power_method)

                            elif transformation_method == "Polynomial Transformation":
                                # Polynomial Transformation: Useful for creating interaction terms or higher-order features
                                def polynomial_transform(data, columns, degree=2, interaction_only=False):
                                    """
                                    Generate polynomial and interaction features for specified columns.

                                    Parameters:
                                        data (pd.DataFrame): The dataset to process.
                                        columns (list): List of numeric columns to apply polynomial transformation.
                                        degree (int): The degree of the polynomial features.
                                        interaction_only (bool): Whether to include only interaction terms.

                                    Returns:
                                        pd.DataFrame: The dataset with polynomial-transformed columns.
                                    """
                                    pf = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

                                    # Ensure all selected columns are numeric
                                    for col in columns:
                                        if not pd.api.types.is_numeric_dtype(data[col]):
                                            raise ValueError(f"Polynomial transformation can only be applied to numeric columns. Column '{col}' is not numeric.")

                                    # Generate polynomial features
                                    poly_features = pf.fit_transform(data[columns])
                                    feature_names = pf.get_feature_names_out(columns)
                                    poly_data = pd.DataFrame(poly_features, columns=feature_names, index=data.index)

                                    # Drop original columns and add transformed features
                                    data = data.drop(columns=columns, axis=1)
                                    data = pd.concat([data, poly_data], axis=1)

                                    return data

                                # Apply polynomial transformation
                                data = polynomial_transform(data, cols_to_transform, degree=degree, interaction_only=interaction_only)

                            # Update session state with the processed data
                            st.session_state.data = data.copy()

                            # Display success message
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.success(f"✅ {transformation_method} applied successfully!")
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Show preview of transformed data
                            st.markdown("#### Transformed Data Preview")
                            st.dataframe(data.head())

                        except Exception as e:
                            st.error(f"An error occurred: {e}")

                            # Update session state with the processed data
                            st.session_state.data = data.copy()

                            # Display success message
                            st.markdown('<div class="success-message">', unsafe_allow_html=True)
                            st.success(f"✅ {transformation_method} applied successfully!")
                            st.markdown('</div>', unsafe_allow_html=True)

                            # Show preview of transformed data
                            st.markdown("#### Transformed Data Preview")
                            st.dataframe(data.head())
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

            # Save preprocessed data
            if st.button("Complete Preprocessing"):
                st.session_state.preprocessed_data = st.session_state.data.copy()
            # Display preprocessed data if it exists in session state
            if st.session_state.preprocessed_data is not None:
                st.markdown("### Processed Data Preview")
                st.dataframe(st.session_state.preprocessed_data)
                # Display success message
                st.markdown('<div class="success-message">', unsafe_allow_html=True)
                st.success("✅ Preprocessing completed successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show preview of final preprocessed data
                st.markdown("#### Final Preprocessed Data")
                st.dataframe(st.session_state.preprocessed_data)
            # Placeholder for next steps
            st.markdown("--- ")
            st.header("Next Step:")
            st.write("After completing preprocessing, you can proceed to the Model Selection page to train machine learning models.")

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No dataset uploaded. Please upload a dataset first.")
        if st.button("Go to File Upload"):
            navigate_to(2)
# Page 5: ML Models
elif st.session_state.page == 5:
    st.title("📊 Machine Learning Models")

    # Check if preprocessed data is available
    if st.session_state.preprocessed_data is not None:
        data = st.session_state.preprocessed_data.copy()
        st.subheader("📄 Preprocessed Data:")
        st.dataframe(data)

        # Model selection in the main page
        st.subheader("🔍 Select Machine Learning Model:")
        model_option = st.selectbox("Choose the model you want to apply:", 
            ["Decision Tree", "Neural Network", "Naive Bayes", "Linear Regression", "k-means"])

        # Select target column
        st.subheader("🎯 Select Target Column:")
        target_column = st.selectbox("Choose the target column", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        st.subheader("🧪 Features and Target:")
        st.write("✅ Features (X):")
        st.dataframe(X.head())
        st.write("🎯 Target (y):")
        st.dataframe(y.head())

        if X.empty or y.empty:
            st.error("❌ Features or Target are empty. Please check your data.")
        else:
            # Model-specific logic
            if model_option == "Decision Tree":
                st.subheader("🌳 Decision Tree")
                if st.button("Train Your Model"):
                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Initialize the Decision Tree Classifier
                    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
                    
                    # Train the model
                    clf.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = clf.predict(X_test)
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"🎯 **Accuracy:** {accuracy:.4f}")
                    
                    # Display classification report
                    st.write("📋 **Classification Report:**")
                    st.text(classification_report(y_test, y_pred))
                    
                    # Display confusion matrix
                    st.write("📊 **Confusion Matrix:**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)
                    
                    # Visualize the Decision Tree
                    st.write("🌲 **Decision Tree Visualization:**")
                    fig, ax = plt.subplots(figsize=(20, 10))
                    tree.plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_], filled=True)
                    st.pyplot(fig)

            elif model_option == "Neural Network":
                st.subheader("🌐 Neural Network")
                if st.button("Train Your Model"):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                    # Train the model
                    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                
                    # Metrics
                    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                    accuracy = accuracy_score(y_test, y_pred)
                
                    # Show metrics
                    st.subheader("📊 Model Performance:")
                    st.write("🎯 **Accuracy:**", round(accuracy, 4))
                    st.write("📋 **Classification Report:**")
                    st.text(classification_report(y_test, y_pred))
                
                    # Confusion matrix
                    st.subheader("📉 Confusion Matrix:")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)
                   
            elif model_option == "Naive Bayes":
                st.subheader("📊 Naive Bayes")
                if st.button("Train Your Model"):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train the Naive Bayes model
                    clf = GaussianNB()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    # Calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)

                    # Display the results
                    st.subheader("📊 Model Performance:")
                    st.write("🎯 **Accuracy:**", round(accuracy, 4))
                    st.write("📋 **Classification Report:**")
                    st.text(classification_report(y_test, y_pred))

                    # Confusion matrix
                    st.subheader("📉 Confusion Matrix:")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)

            elif model_option == "Linear Regression":
                st.subheader("📈 Linear Regression")
                if st.button("Train Your Model"):
                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Initialize and train the linear regression model
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    # Predict on the test set
                    y_pred = model.predict(X_test)

                    # Calculate R² and Mean Squared Error
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)

                    # Show metrics
                    st.subheader("📊 Model Performance:")
                    st.write(f"🎯 **R² (Coefficient of Determination):** {r2:.4f}")
                    st.write(f"📍 **Mean Squared Error (MSE):** {mse:.4f}")

                    # Plotting predictions vs true values
                    st.subheader("📈 Predictions vs True Values")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, y_pred, color='blue', alpha=0.5)
                    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
                    ax.set_xlabel('True Values')
                    ax.set_ylabel('Predictions')
                    ax.set_title('Linear Regression: Predictions vs True Values')
                    st.pyplot(fig)

            elif model_option == "k-means":
                st.subheader("📊 K-Means Clustering")

                # Normalize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(X)

                # Select number of clusters
                st.subheader("🔢 Select Number of Clusters")
                n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3, key="kmeans_n_clusters")

                # Apply KMeans
                if st.button("Train Your Model", key="train_kmeans"):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(scaled_data)
                    labels = kmeans.labels_
                    data['Cluster'] = labels

                    # Show SSE
                    sse = kmeans.inertia_
                    st.subheader("📉 Clustering Performance")
                    st.write(f"📏 **Sum of Squared Errors (SSE):** {sse:.2f}")

                    # Show clustered data
                    st.subheader("📊 Clustered Data")
                    st.dataframe(data.head())

                    # Plot 2D 
                    st.subheader("📈 Cluster Visualization")
                    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

                    fig2d = plt.figure(figsize=(8, 5))
                    # Ensure x_axis and y_axis are defined for the scatterplot
                    if len(numeric_cols) >= 2:
                        x_axis = numeric_cols[0]
                        y_axis = numeric_cols[1]
                        sns.scatterplot(x=data[x_axis], y=data[y_axis], hue=data['Cluster'], palette="tab10")
                    else:
                        st.error("Not enough numeric columns to create a 2D scatterplot.")
                    plt.title("K-Means Clusters (2D)")
                    st.pyplot(fig2d)

    else:
        st.warning("⚠️ Please complete the preprocessing step first.")
        if st.button("Go to Preprocessing Page"):
            navigate_to(4)
            # Finish Button
    if st.button("Finish", key="finish_button"):
                st.balloons()
                st.success("🎉 You have successfully trained your model! Thank you for using the Interactive Machine Learning Project. 🚀")
                st.info("Feel free to explore other features or retrain your model with different configurations.")