
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Kepuasan Pegawai V2", layout="wide")
st.title("📊 Dashboard Analisis Kepuasan Layanan Kepegawaian (Advanced)")
st.markdown("Dashboard interaktif untuk analisis kepuasan dan rekomendasi kebijakan")

# ==========================================================
# LOAD DATA
# ==========================================================
uploaded_file = st.file_uploader("Upload File Excel Survei", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    indikator = df.iloc[:, 1:6].apply(pd.to_numeric, errors="coerce")

    # ==========================================================
    # KPI KEPUASAN (IKM)
    # ==========================================================
    mean_scores = indikator.mean()
    ikm = (mean_scores.mean() / 5) * 100

    def kategori_ikm(x):
        if x >= 81: return "Sangat Baik"
        elif x >= 66: return "Baik"
        elif x >= 51: return "Cukup"
        else: return "Kurang"

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Indeks Kepuasan (IKM)", f"{ikm:.2f}%")
    col2.metric("🏷️ Kategori", kategori_ikm(ikm))
    col3.metric("👥 Jumlah Responden", len(df))

    st.divider()

    # ==========================================================
    # DISTRIBUSI SKOR
    # ==========================================================
    st.header("📊 Distribusi Skor Kepuasan")

    fig_dist, ax_dist = plt.subplots()
    indikator.mean().plot(kind="bar", ax=ax_dist)
    ax_dist.set_ylim(0,5)
    ax_dist.set_ylabel("Rata-rata Skor")
    ax_dist.set_title("Rata-rata Skor Tiap Indikator")
    st.pyplot(fig_dist)

    st.divider()

    # ==========================================================
    # ANALISIS GAP
    # ==========================================================
    st.header("📉 Analisis GAP")

    gap_scores = 5 - mean_scores
    prioritas_gap = gap_scores.idxmax()

    fig_gap, ax_gap = plt.subplots()
    ax_gap.bar(gap_scores.index, gap_scores.values)
    ax_gap.set_title("GAP per Indikator")
    ax_gap.set_ylabel("Nilai GAP")

    for i, v in enumerate(gap_scores.values):
        ax_gap.text(i, v + 0.05, f"{v:.2f}", ha="center")

    st.pyplot(fig_gap)
    st.warning(f"Prioritas utama perbaikan: {prioritas_gap}")

    st.divider()

    # ==========================================================
    # KORELASI
    # ==========================================================
    st.header("🔗 Korelasi Antar Indikator")

    corr = indikator.corr()

    fig_corr, ax_corr = plt.subplots()
    im = ax_corr.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im)
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=45)
    ax_corr.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax_corr.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")

    st.pyplot(fig_corr)

    st.divider()

    # ==========================================================
    # REGRESI
    # ==========================================================
    st.header("📈 Regresi Linear Berganda")

    X = sm.add_constant(indikator.iloc[:, 0:4])
    y = indikator.iloc[:, 4]

    model = sm.OLS(y, X, missing="drop").fit()

    coef = model.params[1:]
    r2 = model.rsquared

    fig_reg, ax_reg = plt.subplots()
    ax_reg.bar(coef.index, coef.values)
    ax_reg.axhline(0, linestyle="--")
    ax_reg.set_title("Koefisien Regresi")
    st.pyplot(fig_reg)

    st.info(f"Nilai R²: {r2:.2f}")
    st.success(f"Faktor paling dominan: {coef.abs().idxmax()}")

    st.divider()

    # ==========================================================
    # CLUSTERING
    # ==========================================================
    st.header("📌 Segmentasi Kepuasan")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_label = kmeans.fit_predict(X_scaled)

    indikator_cluster = indikator.copy()
    indikator_cluster["Cluster"] = cluster_label

    cluster_mean = indikator_cluster.groupby("Cluster").mean()
    st.dataframe(cluster_mean)

else:
    st.info("Silakan upload file Excel terlebih dahulu.")
