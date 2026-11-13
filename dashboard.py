from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Olist E-Commerce Dashboard", page_icon="🛒", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
GRAY, BLUE, ORANGE, DARK = "#B7BDC6", "#1F5A94", "#E07A1F", "#2B2F36"
ID_MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
SEGMENTS = ["Loyal Customers", "Dormant Repeat Buyers", "Promising High-Value", "New Customers",
            "At Risk High-Value", "Hibernating"]
MIN_RFM_CUSTOMERS = 50

def month_label(ym: str, full: bool = False) -> str:
    y, m = ym.split("-")
    return f"{ID_MONTHS[int(m) - 1]} {y if full else y[2:]}"

def pretty_cat(name: str) -> str:
    return name.replace("_", " ").title()

def fmt_brl(x: float) -> str:
    return f"R$ {x / 1e6:.2f} jt" if abs(x) >= 5e5 else f"R$ {x / 1e3:,.0f} rb"

def style_fig(fig: go.Figure, height: int = 420, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=20, t=30, b=10),
        font=dict(color=DARK, size=12), showlegend=legend, hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def show(fig: go.Figure) -> None:

    st.plotly_chart(fig, theme=None)

@st.cache_data(show_spinner="Memuat data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "main_data.csv", parse_dates=["order_purchase_timestamp"])
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["is_late"] = df["is_late"].astype(bool)
    df["item_total"] = df["price"] + df["freight_value"]
    df["same_state"] = df["customer_state"] == df["seller_state"]
    return df

@st.cache_data
def load_geo() -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / "geo_state.csv")

@st.cache_data(show_spinner=False)
def get_filtered(start: str, end: str, states: tuple) -> pd.DataFrame:
    df = load_data()
    mask = (df["year_month"] >= start) & (df["year_month"] <= end)
    if states:
        mask &= df["customer_state"].isin(states)
    return df[mask]

@st.cache_data(show_spinner=False)
def get_orders(start: str, end: str, states: tuple) -> pd.DataFrame:
    df = get_filtered(start, end, states)
    return (
        df.groupby("order_id")
        .agg(customer_unique_id=("customer_unique_id", "first"), customer_state=("customer_state", "first"),
             year_month=("year_month", "first"), purchase_ts=("order_purchase_timestamp", "first"),
             revenue=("price", "sum"), delivery_days=("delivery_days", "first"),
             is_late=("is_late", "first"), review_score=("review_score", "first"),
             same_state=("same_state", "all"))
        .reset_index()
    )

@st.cache_data(show_spinner=False)
def get_rfm(start: str, end: str, states: tuple) -> pd.DataFrame:
    df = get_filtered(start, end, states)
    snapshot = df["order_purchase_timestamp"].max().normalize() + pd.DateOffset(days=1)
    rfm = (
        df.groupby("customer_unique_id")
        .agg(last_purchase=("order_purchase_timestamp", "max"), frequency=("order_id", "nunique"),
             monetary=("item_total", "sum"))
        .reset_index()
    )
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    if len(rfm) < MIN_RFM_CUSTOMERS:
        return rfm.assign(segment="n/a")
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["f_score"] = pd.cut(rfm["frequency"], bins=[0, 1, 2, np.inf], labels=[1, 2, 3]).astype(int)
    conditions = [
        (rfm["f_score"] >= 2) & (rfm["r_score"] >= 3),
        (rfm["f_score"] >= 2),
        (rfm["r_score"] >= 4) & (rfm["m_score"] >= 4),
        (rfm["r_score"] >= 4),
        (rfm["r_score"] <= 2) & (rfm["m_score"] >= 4),
        (rfm["r_score"] <= 2),
    ]
    rfm["segment"] = np.select(conditions, SEGMENTS, default="Need Attention")
    return rfm

@st.cache_data(show_spinner=False)
def rfm_csv(start: str, end: str, states: tuple) -> bytes:
    cols = ["customer_unique_id", "recency", "frequency", "monetary", "segment"]
    return get_rfm(start, end, states)[cols].to_csv(index=False).encode("utf-8")

df_all = load_data()
geo_state = load_geo()
months = sorted(df_all["year_month"].unique())

with st.sidebar:
    st.title("🛒 Olist E-Commerce")
    st.caption("Brazilian E-Commerce Public Dataset | pesanan berstatus *delivered*")
    st.subheader("Filter")
    start_m, end_m = st.select_slider(
        "Periode (bulan pembelian)", options=months, value=(months[0], months[-1]),
        format_func=lambda m: month_label(m, full=True),
    )
    sel_states = st.multiselect(
        "State pelanggan", sorted(df_all["customer_state"].unique()), default=[],
        help="Kosongkan untuk menampilkan semua state.",
    )
    with st.expander("Definisi metrik"):
        st.markdown(
            "- **Revenue**: jumlah harga produk (tanpa ongkir), dalam Real Brasil (R$)\n"
            "- **Pesanan terlambat**: tanggal diterima > tanggal estimasi\n"
            "- **Review score**: skor 1-5 per pesanan\n"
            "- **RFM Monetary**: harga + ongkir"
        )
    st.caption("Filter berlaku untuk seluruh tab.")

states_key = tuple(sorted(sel_states))
df = get_filtered(start_m, end_m, states_key)
orders = get_orders(start_m, end_m, states_key)

if df.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Ubah periode atau state.")
    st.stop()

st.title("Dashboard Analisis E-Commerce Olist")
scope = "semua state" if not sel_states else ", ".join(sel_states)
st.caption(f"Periode **{month_label(start_m, True)} - {month_label(end_m, True)}** | pelanggan: **{scope}**")

total_rev = df["price"].sum()
n_orders = len(orders)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", fmt_brl(total_rev))
k2.metric("Jumlah Pesanan", f"{n_orders:,}")
k3.metric("Rata-rata Nilai Pesanan", f"R$ {total_rev / n_orders:,.0f}")
k4.metric("Rata-rata Review", f"{orders['review_score'].mean():.2f} / 5")
k5.metric("Pesanan Terlambat", f"{orders['is_late'].mean() * 100:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [" Tren Penjualan", " Kategori Produk", " Pengiriman & Kepuasan", " Geospasial", " Segmentasi RFM"]
)

with tab1:
    st.subheader("Bagaimana tren pesanan dan revenue bulanan, dan kapan puncaknya?")
    metric_choice = st.radio("Metrik", ["Revenue (R$ juta)", "Jumlah pesanan"], horizontal=True)

    monthly = (
        orders.groupby("year_month").agg(orders=("order_id", "nunique"), revenue=("revenue", "sum")).reset_index()
    )
    col = "revenue" if metric_choice.startswith("Revenue") else "orders"
    scale = 1e6 if col == "revenue" else 1
    monthly["value"] = monthly[col] / scale
    monthly["mom"] = monthly[col].pct_change() * 100
    labels = [month_label(m) for m in monthly["year_month"]]
    peak_i = int(monthly["value"].idxmax())

    fig = go.Figure(go.Bar(
        x=labels, y=monthly["value"], marker_color=[BLUE if i == peak_i else GRAY for i in range(len(monthly))],
        hovertemplate="%{x}<br>" + metric_choice + ": %{y:,.2f}<extra></extra>",
    ))
    fig.add_annotation(x=labels[peak_i], y=monthly["value"].iloc[peak_i], text=f"Puncak: {labels[peak_i]}",
                       showarrow=True, arrowhead=2, ay=-35, font=dict(color=DARK))
    fig.update_yaxes(title=metric_choice, rangemode="tozero")
    show(style_fig(fig, 360))

    st.markdown("**Pertumbuhan Month-over-Month (%)**")
    mom = monthly.dropna(subset=["mom"])
    if mom.empty:
        st.info("Pilih minimal dua bulan untuk melihat pertumbuhan MoM.")
    else:
        fig = go.Figure(go.Bar(
            x=[month_label(m) for m in mom["year_month"]], y=mom["mom"],
            marker_color=[BLUE if v >= 0 else ORANGE for v in mom["mom"]],
            text=[f"{v:+.0f}%" for v in mom["mom"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>MoM: %{y:+.1f}%<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=DARK, line_width=1)
        fig.update_yaxes(title="MoM (%)")
        show(style_fig(fig, 300))

    peak = monthly.iloc[peak_i]
    st.info(
        f"**Puncak:** {month_label(peak['year_month'], True)} dengan {fmt_brl(peak['revenue'])} dan "
        f"{int(peak['orders']):,} pesanan. Pada data lengkap, puncak jatuh di November 2017 (bertepatan dengan Black Friday, "
        "24 Nov 2017) lalu penjualan mendatar sepanjang 2018. Persiapkan stok & kapasitas kurir sebelum periode serupa."
    )

with tab2:
    st.subheader("Kategori mana yang menyumbang revenue terbesar, dan mana yang review-nya di bawah rata-rata platform?")
    c_a, c_b = st.columns([1, 1])
    top_n = c_a.slider("Jumlah kategori teratas", 5, 20, 10)
    min_rev = c_b.slider("Minimal jumlah ulasan agar skor ditampilkan", 0, 500, 30, step=10,
                         help="Skor rata-rata pada kategori dengan sedikit ulasan tidak stabil.")

    cat = df.groupby("category").agg(revenue=("price", "sum"), items=("price", "size"))
    rev_review = (
        df.drop_duplicates(["order_id", "category"]).groupby("category")["review_score"]
        .agg(avg_review="mean", n_reviews="count")
    )
    cat = cat.join(rev_review)
    cat["share"] = cat["revenue"] / cat["revenue"].sum() * 100
    platform_avg = orders["review_score"].mean()
    top = cat.sort_values("revenue", ascending=False).head(top_n).iloc[::-1].copy()
    top["ok_review"] = top["n_reviews"] >= min_rev
    below = (top["avg_review"] < platform_avg) & top["ok_review"]
    colors = np.where(below, ORANGE, BLUE).tolist()
    ylabels = [pretty_cat(c) for c in top.index]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.58, 0.42], horizontal_spacing=0.03,
                        subplot_titles=("Revenue (R$ juta) & kontribusi", f"Review score vs rata-rata platform ({platform_avg:.2f})"))
    fig.add_trace(go.Bar(
        x=top["revenue"] / 1e6, y=ylabels, orientation="h", marker_color=colors,
        text=[f"R$ {r / 1e6:.2f} jt ({s:.1f}%)" for r, s in zip(top["revenue"], top["share"])],
        textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>Revenue: R$ %{x:.2f} jt<extra></extra>",
    ), row=1, col=1)

    xs, ys = [], []
    shown = top[top["ok_review"]]
    for lbl, v in zip([pretty_cat(c) for c in shown.index], shown["avg_review"]):
        xs += [platform_avg, v, None]
        ys += [lbl, lbl, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GRAY, width=2), hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=shown["avg_review"], y=[pretty_cat(c) for c in shown.index], mode="markers+text",
        marker=dict(size=12, color=np.where(below[shown.index], ORANGE, BLUE).tolist()),
        text=[f"{v:.2f}" for v in shown["avg_review"]], textposition="middle right",
        customdata=shown["n_reviews"], hovertemplate="%{y}<br>Review: %{x:.2f}<br>n ulasan: %{customdata:,}<extra></extra>",
    ), row=1, col=2)
    fig.add_vline(x=platform_avg, line_dash="dash", line_color=DARK, row=1, col=2)
    xmin = min(3.5, float(shown["avg_review"].min()) - 0.1) if len(shown) else 3.5
    fig.update_xaxes(range=[xmin, 4.6], title="Rata-rata review (sumbu dipotong)", row=1, col=2)
    fig.update_xaxes(range=[0, top["revenue"].max() / 1e6 * 1.45], title="Revenue (R$ juta)", row=1, col=1)
    show(style_fig(fig, max(420, 36 * top_n + 120)))
    st.caption("🟧 oranye = review di bawah rata-rata platform | 🟦 biru = di atas / sama. "
               "Sumbu panel kanan tidak dimulai dari 0 (menyoroti selisih terhadap rata-rata).")

    n_below = int(below.sum())
    top_cats = list(top.index[::-1])
    msg = (f"{top_n} kategori teratas menyumbang **{top['share'].sum():.1f}%** revenue. "
           f"**{n_below}** di antaranya berulasan di bawah rata-rata platform")
    if n_below:
        worst = top[below].sort_values("revenue", ascending=False)
        msg += f" (kontributor revenue terbesar di antaranya: **{pretty_cat(worst.index[0])}**, {worst['avg_review'].iloc[0]:.2f})."
    st.info(msg + " Prioritaskan audit kualitas pada kategori berwarna oranye.")

with tab3:
    st.subheader("Berapa persen pesanan terlambat per state, dan seberapa besar dampaknya pada review?")
    c_a, c_b = st.columns(2)
    min_orders = c_a.slider("Minimal jumlah pesanan per state", 50, 1000, 250, step=50,
                            help="State dengan pesanan sedikit dikecualikan agar persentase tidak menyesatkan.")
    top_states = c_b.slider("Jumlah state ditampilkan", 5, 15, 10)

    late_overall = orders["is_late"].mean() * 100
    state_stats = orders.groupby("customer_state").agg(
        orders=("order_id", "size"), late_orders=("is_late", "sum"), late_rate=("is_late", "mean"),
        avg_review=("review_score", "mean"), median_days=("delivery_days", "median"),
    )
    state_stats["late_rate"] *= 100
    elig = state_stats[state_stats["orders"] >= min_orders].sort_values("late_rate", ascending=False).head(top_states)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Tingkat keterlambatan per state**")
        if elig.empty:
            st.info("Tidak ada state yang memenuhi batas minimal pesanan. Turunkan batas pada slider.")
        else:
            e = elig.iloc[::-1]
            flags = np.arange(len(e)) >= len(e) - min(5, len(e))
            fig = go.Figure(go.Bar(
                x=e["late_rate"], y=e.index, orientation="h", marker_color=np.where(flags, ORANGE, GRAY).tolist(),
                text=[f"{r:.1f}% (n={n:,})" for r, n in zip(e["late_rate"], e["orders"])], textposition="outside",
                cliponaxis=False, hovertemplate="%{y}<br>Terlambat: %{x:.1f}%<extra></extra>",
            ))
            fig.add_vline(x=late_overall, line_dash="dash", line_color=DARK,
                          annotation_text=f"Rata-rata {late_overall:.1f}%", annotation_position="bottom right")
            fig.update_xaxes(title="Pesanan terlambat (%)", range=[0, e["late_rate"].max() * 1.35])
            show(style_fig(fig, max(360, 34 * len(e) + 100)))
    with right:
        st.markdown("**Dampak ke review score**")
        grp = orders.groupby("is_late")["review_score"].agg(["mean", "count"]).reindex([False, True])
        names = ["Tepat waktu", "Terlambat"]
        fig = go.Figure(go.Bar(
            x=names, y=grp["mean"].fillna(0), marker_color=[BLUE, ORANGE],
            text=[f"{v:.2f}" if pd.notna(v) else "-" for v in grp["mean"]], textposition="outside",
            customdata=grp["count"].fillna(0), hovertemplate="%{x}<br>Review: %{y:.2f}<br>n ulasan: %{customdata:,}<extra></extra>",
        ))
        fig.update_yaxes(range=[0, 5.4], title="Rata-rata review (skala 1-5)")
        show(style_fig(fig, 360))

    if grp["mean"].notna().all():
        st.info(f"Pesanan terlambat menerima review rata-rata **{grp.loc[True, 'mean']:.2f}** vs **{grp.loc[False, 'mean']:.2f}** "
                f"pada pesanan tepat waktu (selisih **{grp.loc[False, 'mean'] - grp.loc[True, 'mean']:.2f} poin**). "
                "Hubungan ini bersifat asosiasi, bukan bukti sebab-akibat penuh.")

    st.markdown("---")
    st.markdown("#### Analisis lanjutan: Binning durasi pengiriman")
    bin_labels = ["≤ 7 hari", "> 7-14 hari", "> 14-21 hari", "> 21-30 hari", "> 30 hari"]
    ob = orders.assign(bin=pd.cut(orders["delivery_days"], [0, 7, 14, 21, 30, np.inf], labels=bin_labels))
    bs = ob.groupby("bin", observed=True).agg(
        orders=("order_id", "size"), avg_review=("review_score", "mean"),
        pct_neg=("review_score", lambda s: (s.dropna() <= 2).mean() * 100),
    )
    bs["pct_orders"] = bs["orders"] / bs["orders"].sum() * 100
    bad = (bs["pct_neg"] >= 20).to_numpy()
    xt = [f"{i}<br>({p:.0f}% pesanan)" for i, p in zip(bs.index, bs["pct_orders"])]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Rata-rata review score", "Ulasan negatif (skor 1-2), %"))
    fig.add_trace(go.Bar(x=xt, y=bs["avg_review"], marker_color=np.where(bad, ORANGE, BLUE).tolist(),
                         text=[f"{v:.2f}" for v in bs["avg_review"]], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=xt, y=bs["pct_neg"], marker_color=np.where(bad, ORANGE, GRAY).tolist(),
                         text=[f"{v:.0f}%" for v in bs["pct_neg"]], textposition="outside"), row=1, col=2)
    fig.update_yaxes(range=[0, 5.6], row=1, col=1)
    fig.update_yaxes(range=[0, max(10, bs["pct_neg"].max() * 1.25)], row=1, col=2)
    show(style_fig(fig, 380))
    st.caption("🟧 oranye = kelompok dengan ≥ 20% ulasan negatif. Kepuasan turun tajam setelah ±21 hari.")

with tab4:
    st.subheader("Geospatial Analysis: sebaran permintaan, penawaran, dan dampaknya ke pengiriman")
    state_agg = orders.groupby("customer_state").agg(
        revenue=("revenue", "sum"), orders=("order_id", "size"), customers=("customer_unique_id", "nunique"),
        late_rate=("is_late", "mean"), avg_review=("review_score", "mean"), median_days=("delivery_days", "median"),
    ).reset_index()
    state_agg["late_rate"] *= 100
    geo_df = state_agg.merge(geo_state, on="customer_state", how="left").dropna(subset=["lat", "lng"])
    geo_df["revenue_jt"] = geo_df["revenue"] / 1e6

    st.markdown("**Peta state pelanggan:** ukuran bubble = revenue, warna = % pesanan terlambat")
    fig = px.scatter_map(
        geo_df, lat="lat", lon="lng", size="revenue_jt", color="late_rate", hover_name="customer_state",
        hover_data={"lat": False, "lng": False, "revenue_jt": ":.2f", "orders": ":,", "late_rate": ":.1f",
                    "avg_review": ":.2f", "median_days": ":.1f"},
        labels={"revenue_jt": "Revenue (R$ jt)", "orders": "Pesanan", "late_rate": "% terlambat",
                "avg_review": "Review", "median_days": "Median hari"},
        color_continuous_scale=[[0, "#D6E0EE"], [0.5, "#F2B77A"], [1, "#B04E00"]],
        size_max=50, zoom=3, center={"lat": -14.5, "lon": -52}, map_style="carto-positron",
    )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title="% terlambat"))
    st.plotly_chart(fig, theme=None)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Pangsa revenue: state pembeli vs state penjual** (8 pembeli terbesar)")
        buyer = df.groupby("customer_state")["price"].sum()
        seller = df.groupby("seller_state")["price"].sum()
        share = pd.DataFrame({"Pembeli": buyer / buyer.sum() * 100, "Penjual": seller / seller.sum() * 100}).fillna(0)
        sd = share.sort_values("Pembeli", ascending=False).head(8).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=sd.index, x=sd["Penjual"], orientation="h", name="Penjual", marker_color=GRAY,
                             text=[f"{v:.1f}%" for v in sd["Penjual"]], textposition="outside"))
        fig.add_trace(go.Bar(y=sd.index, x=sd["Pembeli"], orientation="h", name="Pembeli", marker_color=BLUE,
                             text=[f"{v:.1f}%" for v in sd["Pembeli"]], textposition="outside"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="% dari total revenue", range=[0, max(sd.max()) * 1.25])
        show(style_fig(fig, 400, legend=True))
        if sel_states:
            st.caption("Catatan: karena filter state aktif, pangsa pembeli hanya mencakup state terpilih.")
    with g2:
        st.markdown("**Satu state vs lintas state**")
        gd = orders.groupby("same_state").agg(
            median_days=("delivery_days", "median"), late_rate=("is_late", "mean"), avg_review=("review_score", "mean"),
        ).reindex([True, False])
        gd["late_rate"] *= 100
        names = ["Satu state", "Lintas state"]
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Median hari", "% terlambat", "Review"))
        for i, (c, fmt) in enumerate([("median_days", "{:.1f}"), ("late_rate", "{:.1f}%"), ("avg_review", "{:.2f}")], 1):
            fig.add_trace(go.Bar(x=names, y=gd[c].fillna(0), marker_color=[BLUE, ORANGE],
                                 text=[fmt.format(v) if pd.notna(v) else "-" for v in gd[c]], textposition="outside"),
                          row=1, col=i)
            fig.update_yaxes(range=[0, max(gd[c].max() * 1.3, 1)], row=1, col=i)
        show(style_fig(fig, 400))

    top3 = state_agg.sort_values("revenue", ascending=False).head(3)
    st.info(f"Tiga state dengan revenue terbesar pada filter ini: **{', '.join(top3['customer_state'])}** "
            f"({top3['revenue'].sum() / state_agg['revenue'].sum() * 100:.1f}% revenue). Penjual terkonsentrasi di SP sehingga "
            "banyak pesanan harus dikirim lintas state, yang lebih lama dan lebih sering terlambat.")

with tab5:
    st.subheader("RFM Analysis: segmentasi pelanggan berdasarkan perilaku pembelian")
    rfm = get_rfm(start_m, end_m, states_key)
    if len(rfm) < MIN_RFM_CUSTOMERS:
        st.info(f"Jumlah pelanggan pada filter ini ({len(rfm)}) terlalu sedikit untuk segmentasi RFM "
                f"(minimal {MIN_RFM_CUSTOMERS}). Perluas periode atau state.")
    else:
        one_time = (rfm["frequency"] == 1).mean() * 100
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Pelanggan unik", f"{len(rfm):,}")
        r2.metric("Hanya membeli 1x", f"{one_time:.1f}%")
        r3.metric("Rata-rata recency", f"{rfm['recency'].mean():.0f} hari")
        r4.metric("Rata-rata monetary", f"R$ {rfm['monetary'].mean():,.0f}")

        prof = rfm.groupby("segment").agg(
            customers=("customer_unique_id", "size"), revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"), avg_monetary=("monetary", "mean"), avg_frequency=("frequency", "mean"),
        )
        prof["customer_pct"] = prof["customers"] / prof["customers"].sum() * 100
        prof["revenue_pct"] = prof["revenue"] / prof["revenue"].sum() * 100
        prof = prof.sort_values("revenue_pct")

        fig = go.Figure()
        fig.add_trace(go.Bar(y=prof.index, x=prof["customer_pct"], orientation="h", name="% jumlah pelanggan",
                             marker_color=GRAY, text=[f"{v:.1f}%" for v in prof["customer_pct"]], textposition="outside"))
        fig.add_trace(go.Bar(y=prof.index, x=prof["revenue_pct"], orientation="h", name="% total pengeluaran",
                             marker_color=BLUE, text=[f"{v:.1f}%" for v in prof["revenue_pct"]], textposition="outside"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="Persentase (%)", range=[0, max(prof["customer_pct"].max(), prof["revenue_pct"].max()) * 1.2])
        show(style_fig(fig, 460, legend=True))

        hv = prof.reindex(["Promising High-Value", "At Risk High-Value"]).dropna()
        if not hv.empty:
            st.info(f"Segmen high-value (*Promising* + *At Risk*) = **{hv['customer_pct'].sum():.1f}%** pelanggan namun "
                    f"**{hv['revenue_pct'].sum():.1f}%** pengeluaran. Sebanyak **{one_time:.1f}%** pelanggan **hanya membeli sekali**, "
                    "sehingga reaktivasi *At Risk* dan konversi *Promising* menjadi pelanggan loyal adalah prioritas retensi.")

        with st.expander("Profil tiap segmen & aturan segmentasi"):
            show_prof = prof.sort_values("revenue_pct", ascending=False).rename(columns={
                "customers": "Pelanggan", "revenue": "Total pengeluaran (R$)", "avg_recency": "Rata-rata recency (hari)",
                "avg_monetary": "Rata-rata monetary (R$)", "avg_frequency": "Rata-rata frekuensi",
                "customer_pct": "% pelanggan", "revenue_pct": "% pengeluaran"})
            st.dataframe(show_prof.round(1))
            st.markdown(
                "R & M = kuantil 1-5 (skor tinggi = lebih baik); F = binning manual (1, 2, ≥3 pesanan).\n\n"
                "| Segmen | Aturan |\n|---|---|\n"
                "| Loyal Customers | F ≥ 2 dan R ≥ 3 |\n| Dormant Repeat Buyers | F ≥ 2 dan R ≤ 2 |\n"
                "| Promising High-Value | F = 1, R ≥ 4, M ≥ 4 |\n| New Customers | F = 1, R ≥ 4, M ≤ 3 |\n"
                "| At Risk High-Value | F = 1, R ≤ 2, M ≥ 4 |\n| Hibernating | F = 1, R ≤ 2, M ≤ 3 |\n"
                "| Need Attention | Sisanya (F = 1, R = 3) |"
            )
        st.download_button("⬇️ Unduh daftar segmen pelanggan (CSV)", rfm_csv(start_m, end_m, states_key),
                           file_name="rfm_segments.csv", mime="text/csv")

st.divider()
st.caption("Sumber data: Olist Brazilian E-Commerce Public Dataset | Proyek Analisis Data | Dibangun dengan Streamlit & Plotly")
