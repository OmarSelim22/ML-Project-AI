import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


st.set_page_config(page_title="ML Classifier App", layout="wide")
st.title("📊 Machine Learning Models")


model_option = st.sidebar.selectbox("Select the model you want to apply: ", 
    ["Decision Tree", "Neural Network", "Naive Bayes", "Linear Regression", "k-means"])

if model_option == "Decision Tree":
    # Upload CSV file
    uploaded_file = st.file_uploader("📂 Upload a CSV file containing your data", type=["csv"])

    if uploaded_file is not None:
        # Read the data
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Loaded Data:")
        st.dataframe(df.head())
          
        # User selects sample percentage
        st.subheader("📊 Select Sample Percentage:")
        sample_percentage = st.selectbox("Choose the sample percentage", [10, 25, 50, 75, 100])
            
        # Take sample based on the selected percentage
        if sample_percentage != 100:
            # Sample data randomly, preserving the order as it appears in the dataset
            sample_df = df.sample(frac=sample_percentage / 100, random_state=42)
            sample_df = sample_df.sort_index()  # Sort by the original index to preserve order
            st.write(f"You have sampled {sample_percentage}% of the data randomly:")
            st.dataframe(sample_df)
        else:
            # If user selects 100%, show the full dataset
            sample_df = df
            st.write("You have selected 100% of the data:")
            st.dataframe(sample_df)
        
        # Preprocessing options
        st.subheader("🧼 Preprocessing Options")
    
        # Handle missing values
        missing_option = st.radio("How to handle missing values?", ("Drop rows with missing values", "Fill with mean"))
        if missing_option == "Drop rows with missing values":
            sample_df = sample_df.dropna()
        else:
            if sample_df.select_dtypes(include=['int64', 'float64']).shape[1] > 0:
                sample_df = sample_df.fillna(sample_df.mean(numeric_only=True))
            else:
                st.warning("⚠️ No numeric columns found to fill missing values. Missing values will not be filled.")
    
        # Check for missing values after preprocessing
        if sample_df.isnull().any().any():
            st.warning("⚠️ There are still missing values in the dataset.")
        else:
            st.success("No missing values in the dataset.")
    
        # Encode all categorical (object type) columns
        label_encoders = {}
        for col in sample_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            sample_df[col] = le.fit_transform(sample_df[col])
            label_encoders[col] = le
        
        # Select target column (outside the loop!)
        st.subheader("🎯 Select the Target Column:")
        target_column = st.selectbox("Choose the target column", sample_df.columns, key="target_column_selectbox")
    
        # Identify features (X) by excluding the target column
        X = sample_df.drop(columns=[target_column])  # Drop the target column
        y = sample_df[target_column]
    
        # Display X and y
        st.subheader("🧪 Features (X) and Target (y):")
        st.write("✅ Features (X):")
        st.dataframe(X.head())
        st.write("🎯 Target (y):")
        st.dataframe(y.head())
    
        # Check if X_train and y_train contain any data
        if X.empty or y.empty:
            st.error("Error: Features (X) or Target (y) are empty. Please check your data.")
        else:
            # Train the model using the sample data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if X_train.empty or y_train.empty:
                st.error("Error: Training data is empty after splitting. Please check your preprocessing steps.")
            else:
                clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
    
                # Calculate accuracy
                accuracy = np.sum(y_test == y_pred) / len(y_test)
                
                # Calculate precision
                cm = confusion_matrix(y_test, y_pred)
                precision = cm.diagonal() / cm.sum(axis=0)  # Precision for each class
                precision_avg = np.nanmean(precision)  # Average precision
                
                # Calculate recall
                recall = cm.diagonal() / cm.sum(axis=1)  # Recall for each class
                recall_avg = np.nanmean(recall)  # Average recall
                
                # Calculate F1-score
                f_measure = 2 * (precision * recall) / (precision + recall)  # F1 for each class
                f_measure_avg = np.nanmean(f_measure)  # Average F1-score
                
                # Display the results
                st.subheader("📊 Model Performance:")
                st.write("🎯 **Accuracy of the model**: ", accuracy)
                st.write("🔑 **Precision**: ", precision_avg)
                st.write("📍 **Recall**: ", recall_avg)
                st.write("📏 **F-measure**: ", f_measure_avg)
                
                # Confusion matrix with labels
                cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                class_names = clf.classes_
    
                cm_df = pd.DataFrame(cm,
                                     index=[f"Actual: {name}" for name in class_names],
                                     columns=[f"Predicted: {name}" for name in class_names])
                cm_df.index.name = "ACTUAL CLASS"
                cm_df.columns.name = "PREDICTED CLASS"
    
                st.write("📉 Confusion Matrix (with numeric class labels):")
                st.dataframe(cm_df)
                        
                # Visualize the decision tree based on the sample
                st.subheader("🌳 Decision Tree Visualization:")
                fig, ax = plt.subplots(figsize=(20, 10))
                tree.plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_], filled=True)
                st.pyplot(fig)



elif model_option == "Neural Network":
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Loaded Data:")
        st.dataframe(df.head())
    
        # Sample percentage selection
        st.subheader("📊 Select Sample Percentage:")
        sample_percentage = st.selectbox("Choose sample percentage", [10, 25, 50, 75, 100])
        if sample_percentage != 100:
            sample_df = df.sample(frac=sample_percentage / 100, random_state=42).sort_index()
        else:
            sample_df = df.copy()
    
        # Handle missing values
        st.subheader("🧼 Missing Value Handling:")
        missing_option = st.radio("How to handle missing values?", ("Drop rows", "Fill with mean"))
        if missing_option == "Drop rows":
            sample_df = sample_df.dropna()
        else:
            sample_df = sample_df.fillna(sample_df.mean(numeric_only=True))
    
        # Encode categorical columns
        label_encoders = {}
        for col in sample_df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            sample_df[col] = le.fit_transform(sample_df[col])
            label_encoders[col] = le
    
        # Select target column
        st.subheader("🎯 Select Target Column:")
        target_column = st.selectbox("Choose the target column", sample_df.columns)
    
        # Separate features and target
        X = sample_df.drop(columns=[target_column])
        y = sample_df[target_column]
    
        st.subheader("🧪 Features and Target:")
        st.write("✅ Features (X):")
        st.dataframe(X.head())
        st.write("🎯 Target (y):")
        st.dataframe(y.head())
    
        if X.empty or y.empty:
            st.error("❌ Features or Target are empty. Please check your data.")
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
            # Train the model
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
    
            # Metrics
            cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    
            with np.errstate(divide='ignore', invalid='ignore'):
                precision = np.diag(cm) / np.sum(cm, axis=0)
                recall = np.diag(cm) / np.sum(cm, axis=1)
                f_measure = 2 * (precision * recall) / (precision + recall)
    
            precision_avg = np.nanmean(precision)
            recall_avg = np.nanmean(recall)
            f_measure_avg = np.nanmean(f_measure)
            accuracy = np.mean(y_pred == y_test)
    
            # Show metrics
            st.subheader("📊 Model Performance:")
            st.write("🎯 **Accuracy:**", round(accuracy, 4))
            st.write("🔑 **Precision:**", round(precision_avg, 4))
            st.write("📍 **Recall:**", round(recall_avg, 4))
            st.write("📏 **F1 Score:**", round(f_measure_avg, 4))
    
            # Confusion matrix
            class_names = clf.classes_
            cm_df = pd.DataFrame(cm,
                                 index=[f"Actual: {cls}" for cls in class_names],
                                 columns=[f"Predicted: {cls}" for cls in class_names])
            st.subheader("📉 Confusion Matrix:")
            st.dataframe(cm_df)


elif model_option == "Naive Bayes":
    # File uploader
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Loaded Data:")
        st.dataframe(df.head())
    
        # Sample percentage selection
        st.subheader("📊 Select Sample Percentage:")
        sample_percentage = st.selectbox("Choose sample percentage", [10, 25, 50, 75, 100])
        if sample_percentage != 100:
            sample_df = df.sample(frac=sample_percentage / 100, random_state=42).sort_index()
        else:
            sample_df = df.copy()
    
        # Handle missing values
        st.subheader("🧼 Missing Value Handling:")
        missing_option = st.radio("How to handle missing values?", ("Drop rows", "Fill with mean"))
        if missing_option == "Drop rows":
            sample_df = sample_df.dropna()
        else:
            sample_df = sample_df.fillna(sample_df.mean(numeric_only=True))
    
        # Encode categorical columns
        label_encoders = {}
        for col in sample_df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            sample_df[col] = le.fit_transform(sample_df[col])
            label_encoders[col] = le
    
        # Select target column
        st.subheader("🎯 Select Target Column:")
        target_column = st.selectbox("Choose the target column", sample_df.columns)
    
        # Separate features and target
        X = sample_df.drop(columns=[target_column])
        y = sample_df[target_column]
    
        st.subheader("🧪 Features and Target:")
        st.write("✅ Features (X):")
        st.dataframe(X.head())
        st.write("🎯 Target (y):")
        st.dataframe(y.head())
    
        if X.empty or y.empty:
            st.error("❌ Features or Target are empty. Please check your data.")
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
            # Train the Naive Bayes model
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
    
            # Calculate accuracy
            accuracy = np.sum(y_test == y_pred) / len(y_test)
            
            # Calculate precision
            cm = confusion_matrix(y_test, y_pred)
            precision = cm.diagonal() / cm.sum(axis=0)  # Precision for each class
            precision_avg = np.nanmean(precision)  # Average precision
            
            # Calculate recall
            recall = cm.diagonal() / cm.sum(axis=1)  # Recall for each class
            recall_avg = np.nanmean(recall)  # Average recall
            
            # Calculate F1-score
            f_measure = 2 * (precision * recall) / (precision + recall)  # F1 for each class
            f_measure_avg = np.nanmean(f_measure)  # Average F1-score
            
            # Display the results
            st.subheader("📊 Model Performance:")
            st.write("🎯 **Accuracy of the model**: ", round(accuracy, 4))
            st.write("🔑 **Precision**: ", round(precision_avg, 4))
            st.write("📍 **Recall**: ", round(recall_avg, 4))
            st.write("📏 **F-measure**: ", round(f_measure_avg, 4))
                
    
            # Confusion matrix
            class_names = clf.classes_
            cm_df = pd.DataFrame(cm,
                                 index=[f"Actual: {cls}" for cls in class_names],
                                 columns=[f"Predicted: {cls}" for cls in class_names])
            st.subheader("📉 Confusion Matrix:")
            st.dataframe(cm_df)



elif model_option == "Linear Regression":
# File uploader
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Loaded Data:")
        st.dataframe(df.head())
    
        # Sample percentage selection
        st.subheader("📊 Select Sample Percentage:")
        sample_percentage = st.selectbox("Choose sample percentage", [10, 25, 50, 75, 100])
        if sample_percentage != 100:
            sample_df = df.sample(frac=sample_percentage / 100, random_state=42).sort_index()
        else:
            sample_df = df.copy()
    
        # Handle missing values
        st.subheader("🧼 Missing Value Handling:")
        missing_option = st.radio("How to handle missing values?", ("Drop rows", "Fill with mean"))
        if missing_option == "Drop rows":
            sample_df = sample_df.dropna()
        else:
            sample_df = sample_df.fillna(sample_df.mean(numeric_only=True))
    
        # Select target column
        st.subheader("🎯 Select Target Column:")
        target_column = st.selectbox("Choose the target column", sample_df.columns)
    
        # Select features (X) and target (y)
        X = sample_df.drop(columns=[target_column])
        y = sample_df[target_column]
    
        # If target column is categorical, encode it
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            y = pd.Series(y)  # Convert numpy array to pandas Series for displaying
    
        st.subheader("🧪 Features and Target:")
        st.write("✅ Features (X):")
        st.dataframe(X.head())
        st.write("🎯 Target (y):")
        st.dataframe(y.head())  # Now it will work since y is a pandas Series
    
        if X.empty or y.empty:
            st.error("❌ Features or Target are empty. Please check your data.")
        else:
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
    
            SSE = np.sum((y_test - y_pred) ** 2)
            variance = np.var(y_test)
            std_dev = np.std(y_test)
        
            # Show metrics
            st.subheader("📊 Model Performance:")
            st.write(f"🎯 **R² (Coefficient of Determination):** {r2:.4f}")
            st.write(f"📍 **Mean Squared Error (MSE):** {mse:.4f}")
    
            # Print the results
            st.subheader("📊 Model Metrics:")
            st.write(f"🎯 **Sum of Squared Error (SSE)**: {SSE}")
            st.write(f"📍 **Variance**: {variance}")
            st.write(f"📏 **Standard Deviation (STD)**: {std_dev}")
    
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
    # Upload CSV file
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Loaded Data")
        st.dataframe(df.head())
    
        # Sample selection
        st.subheader("📊 Select Sample Percentage:")
        sample_percentage = st.selectbox("Choose sample percentage", [10, 25, 50, 75, 100])
        if sample_percentage != 100:
            df = df.sample(frac=sample_percentage / 100, random_state=42).sort_index()
            
        # Handle missing values
        df = df.dropna()
    
        # Encode categorical data
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
    
        # Select number of clusters
        st.subheader("🔢 Select Number of Clusters")
        n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
    
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        labels = kmeans.labels_
        df['Cluster'] = labels
    
        # Show SSE
        sse = kmeans.inertia_
        st.subheader("📉 Clustering Performance")
        st.write(f"📏 **Sum of Squared Errors (SSE):** {sse:.2f}")
    
        # Show clustered data
        st.subheader("📊 Clustered Data")
        st.dataframe(df.head())
    
        # Plot 2D or 3D if possible
        st.subheader("📈 Cluster Visualization")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X-Axis", numeric_cols, index=0)
            y_axis = st.selectbox("Y-Axis", numeric_cols, index=1)
            plot_type = st.radio("Plot type", ["2D", "3D"])
    
            if plot_type == "2D":
                fig2d = plt.figure(figsize=(8, 5))
                sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df['Cluster'], palette="tab10")
                plt.title("K-Means Clusters (2D)")
                st.pyplot(fig2d)
    
            elif plot_type == "3D" and len(numeric_cols) >= 3:
                z_axis = st.selectbox("Z-Axis", numeric_cols, index=2)
                fig3d = plt.figure(figsize=(8, 6))
                ax = fig3d.add_subplot(111, projection='3d')
                ax.scatter(df[x_axis], df[y_axis], df[z_axis], c=df['Cluster'], cmap='tab10')
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_zlabel(z_axis)
                ax.set_title("K-Means Clusters (3D)")
                st.pyplot(fig3d)
        else:
            st.warning("Need at least 2 numeric columns to plot clusters.")






