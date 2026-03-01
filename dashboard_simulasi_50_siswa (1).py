
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# DASHBOARD ANALISIS 50 SISWA - 20 SOAL
# ===============================

# Baca file Excel (pastikan file ini satu folder dengan script)
file_path = "data_simulasi_50_siswa_20_soal.xlsx"
df = pd.read_excel(file_path)

# Hitung total skor tiap siswa
df["Total_Skor"] = df.sum(axis=1)

# Hitung rata-rata skor per soal
rata_per_soal = df.iloc[:, :20].mean()

# Hitung rata-rata total siswa
rata_total = df["Total_Skor"].mean()

print("===== HASIL ANALISIS =====")
print("Rata-rata Total Skor Siswa :", round(rata_total, 2))
print()
print("Rata-rata per Soal:")
print(rata_per_soal.round(2))

# ===============================
# VISUALISASI
# ===============================

plt.figure()
plt.plot(rata_per_soal.index, rata_per_soal.values)
plt.xticks(rotation=45)
plt.xlabel("Soal")
plt.ylabel("Rata-rata Skor")
plt.title("Rata-rata Skor per Soal (50 Siswa)")
plt.tight_layout()
plt.show()
