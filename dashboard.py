import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Dashboard E-Commerce", page_icon="🛒", layout="wide")

@st.cache_data
def load_data():
    orders = pd.read_csv("orders_dataset.csv")
    order_items = pd.read_csv("order_items_dataset.csv")
    products = pd.read_csv("products_dataset.csv")
    translations = pd.read_csv("product_category_name_translation.csv")
    reviews = pd.read_csv("order_reviews_dataset.csv")
    
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    reviews['review_comment_title'] = reviews['review_comment_title'].fillna("Tanpa Judul")
    reviews['review_comment_message'] = reviews['review_comment_message'].fillna("Tidak ada komentar")
    
    return orders, order_items, products, translations, reviews

orders_df, order_items_df, products_df, translations_df, reviews_df = load_data()

# Q1: Produk terlaris per bulan 2017
orders_2017 = orders_df[orders_df['order_purchase_timestamp'].dt.year == 2017].copy()
orders_2017['month_num'] = orders_2017['order_purchase_timestamp'].dt.month
orders_2017['month'] = orders_2017['order_purchase_timestamp'].dt.month_name()

items_2017 = pd.merge(orders_2017, order_items_df, on='order_id')
items_2017 = pd.merge(items_2017, products_df, on='product_id')
items_2017 = pd.merge(items_2017, translations_df, on='product_category_name', how='left')

monthly_sales = items_2017.groupby(['month_num', 'month', 'product_category_name_english']).size().reset_index(name='sales_count')
idx_max = monthly_sales.groupby('month_num')['sales_count'].idxmax()
top_per_month_2017 = monthly_sales.loc[idx_max].reset_index(drop=True)

# Q2: Kategori ulasan buruk (skor 1-2)
bad_reviews = reviews_df[reviews_df['review_score'] <= 2]
bad_reviews_items = pd.merge(bad_reviews, order_items_df, on='order_id')
bad_reviews_items = pd.merge(bad_reviews_items, products_df, on='product_id')
bad_reviews_items = pd.merge(bad_reviews_items, translations_df, on='product_category_name', how='left')

top_bad_categories = bad_reviews_items.groupby('product_category_name_english').size().reset_index(name='bad_review_count')
top_bad_categories = top_bad_categories.sort_values(by='bad_review_count', ascending=False).head(5).reset_index(drop=True)

# UI Dashboard
st.title("🛒 Dashboard Kinerja Produk E-Commerce")
st.write("Visualisasi interaktif analisis produk terlaris bulanan (2017) dan kategori produk penyumbang ulasan buruk tertinggi (2016-2018).")

with st.sidebar:
    st.header("Informasi Proyek")
    st.write("Analisis Data E-Commerce Public Dataset")
    st.metric(label="Total Kategori Teranalisis", value=len(top_bad_categories))

st.subheader("1. Kategori Produk Terlaris Setiap Bulan (2017)")
fig1, ax1 = plt.subplots(figsize=(12, 5))
sns.barplot(
    x="month", 
    y="sales_count", 
    hue="product_category_name_english", 
    data=top_per_month_2017, 
    dodge=False, 
    palette="Set2", 
    ax=ax1
)
ax1.set_xlabel("Bulan")
ax1.set_ylabel("Jumlah Terjual")
ax1.legend(title="Kategori Produk", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig1)

st.markdown("---")

st.subheader("2. Top 5 Kategori Produk Ulasan Buruk (Skor 1-2)")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.barplot(
    x="bad_review_count", 
    y="product_category_name_english", 
    data=top_bad_categories, 
    palette="Reds_r", 
    ax=ax2
)
ax2.set_xlabel("Jumlah Ulasan Buruk")
ax2.set_ylabel("Kategori Produk")

for index, row in enumerate(top_bad_categories['bad_review_count']):
    ax2.text(row + 5, index, str(row), color='black', va="center", fontweight='bold')

st.pyplot(fig2)

st.info(" **Key Action Item:** Segera audit Quality Control (QC) dan standar pengemasan pada kategori `bed_bath_table` dan `furniture_decor` untuk menekan tingginya komplain ulasan buruk.")