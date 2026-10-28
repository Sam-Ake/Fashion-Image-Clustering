import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fashion Image Clustering Explorer",
    page_icon="👕",
    layout="wide",
)

st.title("👕 End-to-End Fashion Image Clustering & Explorer")
st.markdown(
    "Explore how unsupervised learning groups thousands of diverse fashion items using **ResNet50 feature embeddings**, **PCA**, and **KMeans Clustering**."
)


# Load saved artifacts with caching for performance
@st.cache_resource
def load_models():
    pca = joblib.load("pca_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return pca, kmeans


@st.cache_data
def load_data():
    if os.path.exists("clustered_fashion_sample.csv"):
        return pd.read_csv("clustered_fashion_sample.csv")
    return None


pca, kmeans = load_models()
df = load_data()

if df is None:
    st.error(
        "⚠️ Clustered dataset CSV not found! Please ensure 'clustered_fashion_sample.csv' is in your project directory."
    )
else:
    # Sidebar navigation
    st.sidebar.header("Navigation")
    option = st.sidebar.selectbox(
        "Choose View", ["Cluster Explorer", "Dataset Overview"]
    )

    if option == "Cluster Explorer":
        st.subheader("🔍 Explore Fashion Clusters")
        st.markdown(
            "Select a cluster ID below to inspect sample items grouped together automatically by computer vision."
        )

        cluster_ids = sorted(df["cluster"].unique())
        selected_cluster = st.selectbox(
            "Select Cluster ID",
            cluster_ids,
            format_func=lambda x: f"Cluster {x} ({len(df[df['cluster'] == x])} items)",
        )

        # Filter items for selected cluster
        cluster_items = df[df["cluster"] == selected_cluster]

        # Display sample grid
        num_samples = min(10, len(cluster_items))
        sample_subset = cluster_items.sample(
            num_samples, random_state=42
        ).reset_index(drop=True)

        cols = st.columns(5)
        for i, row in sample_subset.iterrows():
            img_id = row["id"]
            image_path = os.path.join("images_sample", f"{image_id}.jpg")
            

           with cols[i % 5]:
    img_path = os.path.join("images_sample", f"{int(img_id)}.jpg")

    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(
            img,
            use_container_width=True,
            caption=f"{row['masterCategory']}\n{row['subCategory']}",
        )
    else:
        st.info(f"ID {img_id} (Sample mode)")

    elif option == "Dataset Overview":
        st.subheader("📊 Project Dataset & Pipeline Summary")
        st.metric(
            label="Total Processed Subset Items", value=f"{len(df):,}"
        )
        st.metric(label="Embedding Dimensions (ResNet50)", value="2,048")
        st.metric(label="PCA Reduced Dimensions", value="50")
        st.metric(label="KMeans Clusters", value=f"{df['cluster'].nunique()}")

        st.markdown("### Sample Data Preview")
        st.dataframe(df[["id", "masterCategory", "subCategory", "cluster"]].head(10))