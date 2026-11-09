import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Konfigurasi Halaman
st.set_page_config(page_title="Dashboard E-Commerce", page_icon="🛒", layout="wide")

@st.cache_data
def load_data():
    orders = pd.read_csv("orders_dataset.csv")
    order_items = pd.read_csv("order_items_dataset.csv")
    products = pd.read_csv("products_dataset.csv")
    translations = pd.read_csv("product_category_name_translation.csv")
    reviews = pd.read_csv("order_reviews_dataset.csv")
    customers = pd.read_csv("customers_dataset.csv")
    payments = pd.read_csv("order_payments_dataset.csv")
    
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    reviews['review_comment_title'] = reviews['review_comment_title'].fillna("Tanpa Judul")
    reviews['review_comment_message'] = reviews['review_comment_message'].fillna("Tidak ada komentar")
    
    return orders, order_items, products, translations, reviews, customers, payments

orders_df, order_items_df, products_df, translations_df, reviews_df, customers_df, payments_df = load_data()

# --- BAGIAN SIDEBAR (INTERAKTIVITAS) ---
with st.sidebar:
    st.header(" Filter Data & Konfigurasi")
    
    # Widget 1: Filter jumlah top kategori ulasan buruk
    top_n_bad = st.slider("Jumlah Kategori Ulasan Buruk:", min_value=3, max_value=10, value=5)
    
    st.markdown("---")
    st.write("🛒 **Dashboard E-Commerce**")
    st.write("Oleh: Muhammad Zulham Nur Zacky")
    st.write("ID: DS296B6Y1289")

# --- UI UTAMA & TABS ---
st.title(" Dashboard Kinerja Produk E-Commerce")
st.write("Visualisasi interaktif analisis produk terlaris, evaluasi ulasan pelanggan, dan profil transaksi.")
tab1, tab2, tab3 = st.tabs([" Tren Penjualan 2017", " Evaluasi Kategori (Ulasan)", " Analisis Lanjutan (RFM)"])

with tab1:
    st.subheader("Kategori Produk Terlaris Setiap Bulan (2017)")
    
    # Pre-processing Q1
    orders_2017 = orders_df[orders_df['order_purchase_timestamp'].dt.year == 2017].copy()
    orders_2017['month_num'] = orders_2017['order_purchase_timestamp'].dt.month
    orders_2017['month'] = orders_2017['order_purchase_timestamp'].dt.month_name()

    items_2017 = pd.merge(orders_2017, order_items_df, on='order_id')
    items_2017 = pd.merge(items_2017, products_df, on='product_id')
    items_2017 = pd.merge(items_2017, translations_df, on='product_category_name', how='left')

    monthly_sales = items_2017.groupby(['month_num', 'month', 'product_category_name_english']).size().reset_index(name='sales_count')
    idx_max = monthly_sales.groupby('month_num')['sales_count'].idxmax()
    top_per_month_2017 = monthly_sales.loc[idx_max].reset_index(drop=True)

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

with tab2:
    st.subheader(f"Top {top_n_bad} Kategori Produk Ulasan Buruk (Skor 1-2)")
    
    # Pre-processing Q2
    bad_reviews = reviews_df[reviews_df['review_score'] <= 2]
    bad_reviews_items = pd.merge(bad_reviews, order_items_df, on='order_id')
    bad_reviews_items = pd.merge(bad_reviews_items, products_df, on='product_id')
    bad_reviews_items = pd.merge(bad_reviews_items, translations_df, on='product_category_name', how='left')

    top_bad_categories = bad_reviews_items.groupby('product_category_name_english').size().reset_index(name='bad_review_count')
    
    top_bad_categories = top_bad_categories.sort_values(by='bad_review_count', ascending=False).head(top_n_bad).reset_index(drop=True)

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
    st.info("**Action Item:** Segera audit standar pengemasan pada kategori penyumbang komplain teratas untuk menekan ulasan buruk.")

with tab3:
    st.subheader("RFM Analysis (Pelanggan Top Berdasarkan Pengeluaran)")
    
    # Integrasi perhitungan RFM ke dashboard
    rfm_base = pd.merge(orders_df, customers_df, on='customer_id')
    rfm_base = pd.merge(rfm_base, payments_df, on='order_id')
    recent_date = rfm_base['order_purchase_timestamp'].max()

    rfm_df = rfm_base.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (recent_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).reset_index()

    rfm_df.columns = ['customer_unique_id', 'Recency (Days)', 'Frequency (Orders)', 'Monetary (Value)']
    
    top_rfm = rfm_df.sort_values(by='Monetary (Value)', ascending=False).head(10)
    
    st.dataframe(top_rfm, use_container_width=True)
    st.caption("Menampilkan 10 pelanggan dengan Total Pengeluaran (Monetary) tertinggi.")
