import streamlit as st
import pandas as pd
import pickle

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