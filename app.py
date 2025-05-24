import pandas as pd
import streamlit as st
import os
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model, get_config as reg_get_config
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, save_model as cls_save_model, get_config as cls_get_config
from pycaret.clustering import setup as clu_setup, create_model as clu_create_model, pull as clu_pull, save_model as clu_save_model, get_config as clu_get_config
import pickle

# Sidebar setup
st.sidebar.image("https://fekrait.com/uploads/topics/16719193475376.jpg")
st.sidebar.title('Auto Machine Learning')
choice = st.sidebar.radio('Navigation', ['Upload', 'Profiling', 'ML', 'Download' , 'Product Recommendation'])
st.sidebar.info('Build an automated ML Pipeline using Streamlit, ydata_profiling, and PyCaret.')

# Load dataset if it exists
@st.cache_data
def load_data(file):
    return pd.read_csv(file, index_col=None)

# Upload dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv"])
    if file:
        df = load_data(file)
        df.to_csv('dataset.csv', index=None)  
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)

# Data profiling
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

    if choice == 'Profiling':
        st.title('Automated Exploratory Data Analysis')
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)

    # Machine Learning setup
    if choice == "ML":
        st.title('Machine Learning Modeling')

        # Feature selection
        feature_options = df.columns.tolist()
      
        # Model type selection
        chosen_model_type = st.selectbox('Choose the Model Type', ['Regression', 'Classification', 'Clustering'])

        if chosen_model_type in ['Regression', 'Classification']:
            selected_features = st.multiselect('Select Features for the Model', feature_options, default=feature_options[:-1])
            chosen_target = st.selectbox('Choose the Target Column', [col for col in feature_options if col not in selected_features])
        else:  # Clustering
            selected_features = st.multiselect('Select Features for Clustering', feature_options, default=feature_options)
            chosen_target = None  # No target for clustering

        remove_outliers = st.checkbox('Remove Outliers', value=True)

        # Clustering-specific options
        if chosen_model_type == 'Clustering':
            num_clusters = st.slider('Select Number of Clusters', min_value=2, max_value=10, value=4)

        if st.button('Run Modeling'):
            with st.spinner('Training the model...'):
                try:
                    # Prepare data for PyCaret
                    if chosen_model_type == 'Regression':
                        reg_setup(df[selected_features + [chosen_target]], target=chosen_target, remove_outliers=remove_outliers)
                        setup_df = reg_pull()
                        st.subheader("Setup Data")
                        st.dataframe(setup_df)

                        # Extract and save preprocessed data
                        preprocessed_X = reg_get_config('X')
                        preprocessed_y = reg_get_config('y')
                        preprocessed_df = pd.concat([preprocessed_X, preprocessed_y], axis=1)
                        preprocessed_df.to_csv('cleaned_dataset.csv', index=False)
                        st.session_state.cleaned_data_available = True

                        best_model = reg_compare_models()
                        compare_df = reg_pull()
                        st.subheader("Model Comparison")
                        st.dataframe(compare_df)

                        # Save the best regression model
                        model_file = 'best_regression_model'
                        reg_save_model(best_model, model_file)
                        st.success(f"Best regression model saved as '{model_file}.pkl'.")
                        st.session_state.best_model_type = 'regression'

                    elif chosen_model_type == 'Classification':
                        cls_setup(df[selected_features + [chosen_target]], target=chosen_target, remove_outliers=remove_outliers)
                        setup_df = cls_pull()
                        st.subheader("Setup Data")
                        st.dataframe(setup_df)

                        # Extract and save preprocessed data
                        preprocessed_X = cls_get_config('X')
                        preprocessed_y = cls_get_config('y')
                        preprocessed_df = pd.concat([preprocessed_X, preprocessed_y], axis=1)
                        preprocessed_df.to_csv('cleaned_dataset.csv', index=False)
                        st.session_state.cleaned_data_available = True

                        best_model = cls_compare_models()
                        compare_df = cls_pull()
                        st.subheader("Model Comparison")
                        st.dataframe(compare_df)

                        # Save the best classification model
                        model_file = 'best_classification_model'
                        cls_save_model(best_model, model_file)
                        st.success(f"Best classification model saved as '{model_file}.pkl'.")
                        st.session_state.best_model_type = 'classification'

                    elif chosen_model_type == 'Clustering':
                        clu_setup(df[selected_features], remove_outliers=remove_outliers)
                        setup_df = clu_pull()
                        st.subheader("Setup Data")
                        st.dataframe(setup_df)

                        # Extract and save preprocessed data
                        preprocessed_df = clu_get_config('X')
                        preprocessed_df.to_csv('cleaned_dataset.csv', index=False)
                        st.session_state.cleaned_data_available = True

                        # Create and evaluate clustering model (e.g., K-Means)
                        best_model = clu_create_model('kmeans', num_clusters=num_clusters)
                        st.subheader("Clustering Results")
                        st.write("K-Means clustering model created with {} clusters.".format(num_clusters))

                        # Save the best clustering model
                        model_file = 'best_clustering_model'
                        clu_save_model(best_model, model_file)
                        st.success(f"Best clustering model saved as '{model_file}.pkl'.")
                        st.session_state.best_model_type = 'clustering'

                except Exception as e:
                    st.error(f"An error occurred during modeling: {str(e)}")

# Download section
if choice == 'Download':
    st.title('Download Results')
    if hasattr(st.session_state, 'best_model_type'):
        model_file = f'best_{st.session_state.best_model_type}_model'
        if os.path.exists(model_file + '.pkl'):
            with open(model_file + '.pkl', 'rb') as f:
                st.download_button('Download Best Model', f, file_name=f"{model_file}.pkl")
        else:
            st.warning('No trained model found. Please train a model first.')
    else:
        st.warning('No trained model found. Please train a model first.')

    if hasattr(st.session_state, 'cleaned_data_available') and st.session_state.cleaned_data_available:
        if os.path.exists('cleaned_dataset.csv'):
            with open('cleaned_dataset.csv', 'rb') as f:
                st.download_button('Download Cleaned Dataset', f, file_name='cleaned_dataset.csv', mime='text/csv')
        else:
            st.warning('No cleaned dataset found. Please run modeling first.')
    else:
        st.warning('No cleaned dataset found. Please run modeling first.')



## Prodcut Recommendation Module:
if choice == 'Product Recommendation':
        # --- Load model ---
    @st.cache_resource
    def load_model():
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    @st.cache_data
    def load_data():
        product_pivot = pd.read_csv('product_pivot.csv', index_col=0)
        reviews_above_25 = pd.read_csv('reviews_above_25.csv')
        # load other necessary data (like reviews_above_25, product_sparse)
        with open('product_sparse.pkl', 'rb') as f:
            product_sparse = pickle.load(f) 
        return product_pivot, reviews_above_25, product_sparse

    model = load_model()
    product_pivot, reviews_above_25, product_sparse = load_data()

    st.title("Customer-based Product Recommendation App")

    # Pick customer ID from the index of product_pivot
    customer_ids = product_pivot.index.tolist()
    selected_customer = st.selectbox("Choose a customer ID:", customer_ids)
    n_neighbors = st.slider('Number of similar customers to consider', 1, 10, 3)
    n_recommend = st.slider('Number of recommendations', 1, 10, 3)

    # Function for recommendations (uses your logic)
    def recommend_products(customer_id, n_neighbors=2, n_recommendations=3, product_pivot=None, product_sparse=None, reviews_above_25=None):
        if customer_id not in product_pivot.index:
            raise ValueError("Customer not found.")
        customer_vector = product_pivot.loc[[customer_id]]
        distances, indices = model.kneighbors(product_sparse[customer_id], n_neighbors=n_neighbors+1)
        

        similar_customers = product_pivot.index[indices.flatten()[1:]]  # Exclude self
        # Get products rated highly by similar customers
        print(reviews_above_25.head())
        similar_ratings = reviews_above_25[reviews_above_25['customer_id'].isin(similar_customers)]
        high_rated = similar_ratings[similar_ratings['rating'] >= 4]
        seen_products = set(reviews_above_25[reviews_above_25['customer_id'] == customer_id]['name'])
        candidate_products = high_rated[~high_rated['name'].isin(seen_products)]
        recommendations = (candidate_products
                        .groupby('name')
                        .size()
                        .sort_values(ascending=False)
                        .head(n_recommendations)
                        .index.tolist())
        return recommendations

    if st.button("Show Recommendations"):
        try:
            recommended_products = recommend_products(
                customer_id=selected_customer,
                n_neighbors=n_neighbors,
                n_recommendations=n_recommend,
                product_pivot=product_pivot,
                product_sparse=product_sparse,
                reviews_above_25=reviews_above_25
            )
            st.subheader("Recommended Products:")
            if recommended_products:
                st.table(pd.DataFrame(recommended_products, columns=["Product Name"]))
            else:
                st.info("No new recommendations found for this customer.")
        except Exception as e:
            st.error(f"Could not generate recommendations: {e}")

    with st.expander("Show Customer Ratings Table"):
        st.dataframe(product_pivot.fillna(""))