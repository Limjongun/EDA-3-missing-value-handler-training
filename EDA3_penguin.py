# ============================================================
# EDA PENGUINS DATASET
# Fokus: Exploratory Data Analysis + Missing Value Repair
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. LOAD DATASET
# ============================================================

# Load dataset penguins dari seaborn
# Dataset ini berisi informasi spesies penguin, pulau asal,
# ukuran paruh, panjang sirip, berat badan, dan jenis kelamin.
df = sns.load_dataset('penguins')

# Menampilkan 5 data pertama
print("Head dataset:")
print(df.head())

# Menampilkan 5 data terakhir
print("\nTail dataset:")
print(df.tail())


# ============================================================
# 2. DATA OVERVIEW
# ============================================================

# Melihat jumlah baris dan kolom
print("\nShape dataset:")
print(df.shape)

# Melihat informasi dataset:
# - nama kolom
# - tipe data
# - jumlah data non-null
# - penggunaan memory
print("\nInfo dataset:")
print(df.info())

# Melihat statistik deskriptif untuk kolom numerik
# Berisi count, mean, std, min, Q1, median, Q3, dan max
print("\nStatistik deskriptif:")
print(df.describe())

# Melihat semua nama kolom dalam dataset
print("\nKolom dataset:")
print(df.columns)

# Insight awal:
# Dataset penguins memiliki 344 baris dan 7 kolom.
# Kolom numerik:
# - bill_length_mm
# - bill_depth_mm
# - flipper_length_mm
# - body_mass_g
#
# Kolom kategorikal:
# - species
# - island
# - sex
#
# Dari df.info(), terlihat bahwa beberapa kolom memiliki jumlah non-null
# yang berbeda. Artinya, dataset memiliki missing value.


# ============================================================
# 3. CEK MISSING VALUE
# ============================================================

# Mengecek jumlah missing value per kolom
print("\nMissing value per kolom:")
print(df.isnull().sum())

# Menghitung persentase missing value per kolom
missing_value_percent = df.isnull().sum() / len(df) * 100

print("\nPersentase missing value:")
print(missing_value_percent)

# Insight:
# - Kolom sex memiliki missing value paling banyak.
# - Kolom species dan island tidak memiliki missing value.
# - Kolom numerik seperti bill_length_mm, bill_depth_mm,
#   flipper_length_mm, dan body_mass_g memiliki beberapa missing value.
#
# Missing value belum langsung diperbaiki di awal,
# karena dalam EDA kita perlu memahami kondisi data terlebih dahulu.


# ============================================================
# 4. CEK DATA DUPLIKAT
# ============================================================

# Mengecek jumlah data duplikat
print("\nJumlah data duplikat:")
print(df.duplicated().sum())

# Menampilkan baris yang terdeteksi sebagai duplikat
duplicated_rows = df[df.duplicated()]

print("\nData duplikat:")
print(duplicated_rows)

# Insight:
# Jika hasilnya 0, berarti tidak ada baris duplikat.
# Jika ada duplikat, perlu dicek apakah benar data ganda
# atau hanya kebetulan nilainya sama.


# ============================================================
# 5. ANALISIS FITUR KATEGORIKAL
# ============================================================

# Fitur kategorikal adalah fitur berbentuk kategori/label.
#
# species = jenis penguin
# island  = pulau asal penguin
# sex     = jenis kelamin penguin

categorical_column = ['species', 'island', 'sex']

for column in categorical_column:
    print(f"\nValue count untuk {column}:")
    
    # dropna=False digunakan agar missing value juga ikut dihitung
    print(df[column].value_counts(dropna=False))

    sns.countplot(data=df, x=column)
    plt.title(f"Distribusi {column}")
    plt.xlabel(column)
    plt.ylabel("Jumlah")
    plt.show()

# Insight:
# - Species terbanyak adalah Adelie.
# - Island dengan jumlah data terbanyak adalah Biscoe.
# - Jumlah penguin male dan female cukup seimbang,
#   tetapi terdapat beberapa data sex yang masih missing/NaN.
#
# Analisis kategorikal berguna untuk memahami komposisi data
# sebelum masuk ke analisis numerik atau modeling.


# ============================================================
# 6. ANALISIS FITUR NUMERIK
# ============================================================

# Fitur numerik pada dataset penguins:
#
# bill_length_mm    = panjang paruh
# bill_depth_mm     = kedalaman/ketebalan paruh
# flipper_length_mm = panjang sirip
# body_mass_g       = berat badan penguin

numeric_column = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

for column in numeric_column:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribusi {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Insight:
# - Histogram digunakan untuk melihat bentuk distribusi data.
# - Beberapa fitur terlihat memiliki pola distribusi yang cukup jelas.
# - bill_length_mm banyak berada pada range sekitar 40 sampai 50 mm.
# - bill_depth_mm memiliki variasi yang cukup lebar.
# - flipper_length_mm dan body_mass_g kemungkinan memiliki hubungan,
#   karena penguin dengan sirip lebih panjang biasanya lebih besar/berat.
#
# Catatan:
# Distribusi pada dataset penguins bisa terlihat seperti gabungan beberapa pola,
# karena data berasal dari beberapa species yang berbeda.


# ============================================================
# 7. CORRELATION HEATMAP
# ============================================================

# Korelasi hanya dihitung pada kolom numerik.
# Maka kita ambil hanya kolom bertipe int64 dan float64.
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Menghitung korelasi antar fitur numerik
correlation = numeric_df.corr()

print("\nCorrelation matrix:")
print(correlation)

# Membuat heatmap korelasi
plt.figure(figsize=(8, 5))
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.title("Correlation Heatmap Penguins Dataset")
plt.show()

# Cara membaca korelasi:
#  1.00  = hubungan positif sangat kuat
#  0.00  = hampir tidak ada hubungan linear
# -1.00  = hubungan negatif sangat kuat
#
# Insight:
# - body_mass_g dan flipper_length_mm memiliki korelasi positif kuat.
#   Artinya, semakin panjang flipper, berat badan penguin cenderung semakin besar.
# - Korelasi membantu menemukan hubungan linear antar fitur numerik.
# - Namun korelasi bukan berarti sebab-akibat mutlak.


# ============================================================
# 8. VISUALISASI BOXPLOT SEBELUM MISSING VALUE REPAIR
# ============================================================

# Boxplot digunakan untuk melihat:
# - median
# - Q1 dan Q3
# - sebaran data
# - kandidat outlier

for column in numeric_column:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=column)
    plt.title(f"Boxplot Sebelum Missing Value Repair: {column}")
    plt.xlabel(column)
    plt.show()

# Insight:
# Boxplot sebelum repair berguna untuk melihat kondisi awal data.
# Jika ada outlier, median lebih aman digunakan untuk imputasi
# dibanding mean/rata-rata.


# ============================================================
# 9. MISSING VALUE REPAIR
# ============================================================

# Membuat salinan dataframe agar data asli tetap aman
df_clean = df.copy()

# Strategi missing value repair:
#
# 1. Kolom numerik diisi dengan median.
#    Median dipilih karena lebih tahan terhadap outlier.
#
# 2. Kolom kategorikal sex diisi dengan mode.
#    Mode adalah nilai yang paling sering muncul,
#    sehingga cocok untuk data kategorikal.


# ============================================================
# 9A. IMPUTASI KOLOM NUMERIK DENGAN MEDIAN
# ============================================================

for column in numeric_column:
    median_value = df_clean[column].median()

    print(f"\nMedian untuk {column}: {median_value}")

    # Mengisi missing value pada kolom numerik dengan median
    df_clean[column] = df_clean[column].fillna(median_value)

# Penjelasan:
# fillna(median_value) mengganti nilai NaN dengan nilai median.
# Contoh:
# Jika bill_length_mm kosong, maka akan diisi dengan median bill_length_mm.


# ============================================================
# 9B. IMPUTASI KOLOM KATEGORIKAL DENGAN MODE
# ============================================================

# Mengambil nilai mode dari kolom sex
sex_mode = df_clean['sex'].mode()[0]

print("\nMode untuk sex:")
print(sex_mode)

# Mengisi missing value pada kolom sex dengan mode
df_clean['sex'] = df_clean['sex'].fillna(sex_mode)

# Penjelasan:
# Karena sex adalah kolom kategorikal,
# maka tidak bisa diisi dengan mean atau median.
# Oleh karena itu, kita gunakan mode/nilai yang paling sering muncul.


# ============================================================
# 10. CEK ULANG MISSING VALUE SETELAH REPAIR
# ============================================================

print("\nMissing value sebelum repair:")
print(df.isnull().sum())

print("\nMissing value sesudah repair:")
print(df_clean.isnull().sum())

# Insight:
# Jika semua kolom bernilai 0 setelah repair,
# berarti missing value berhasil ditangani.


# ============================================================
# 11. MEMBANDINGKAN DATA SEBELUM DAN SESUDAH REPAIR
# ============================================================

print("\nShape sebelum repair:")
print(df.shape)

print("\nShape sesudah repair:")
print(df_clean.shape)

print("\nDescribe sebelum repair:")
print(df.describe())

print("\nDescribe sesudah repair:")
print(df_clean.describe())

# Insight:
# Karena kita menggunakan fillna/imputation,
# jumlah baris dataset tetap sama.
#0
# Ini berbeda dengan dropna(),
# karena dropna() akan menghapus baris yang mengandung missing value.


# ============================================================
# 12. VISUALISASI BOXPLOT SESUDAH MISSING VALUE REPAIR
# ============================================================

for column in numeric_column:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_clean, x=column)
    plt.title(f"Boxplot Sesudah Missing Value Repair: {column}")
    plt.xlabel(column)
    plt.show()

# Insight:
# Setelah missing value repair, boxplot bisa dibandingkan dengan sebelum repair.
# Jika missing value sedikit, bentuk distribusi biasanya tidak berubah banyak.
# Jika missing value banyak, hasil imputasi bisa memengaruhi distribusi.


# ============================================================
# 13. DETEKSI OUTLIER MENGGUNAKAN IQR
# ============================================================

# IQR adalah Interquartile Range.
#
# Rumus:
# IQR = Q3 - Q1
#
# Lower fence = Q1 - 1.5 * IQR
# Upper fence = Q3 + 1.5 * IQR
#
# Data dianggap outlier jika:
# data < lower_fence
# atau
# data > upper_fence

def detect_outlier_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = data[
        (data[column] < lower_fence) |
        (data[column] > upper_fence)
    ]

    print(f"\nOutlier detection untuk kolom: {column}")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower fence:", lower_fence)
    print("Upper fence:", upper_fence)
    print("Jumlah outlier:", len(outliers))

    return lower_fence, upper_fence, outliers


# Menjalankan deteksi outlier untuk setiap kolom numerik
# Sebaiknya menggunakan df_clean karena missing value sudah diperbaiki
for column in numeric_column:
    lower, upper, outliers = detect_outlier_iqr(df_clean, column)
    print(outliers)

# Insight:
# Jika jumlah outlier 0, berarti tidak ada data yang melewati batas IQR.
# Namun jika ada, outlier belum tentu data salah.
# Pada data biologis seperti penguin, nilai ekstrem bisa saja valid
# karena perbedaan ukuran antar species.


# ============================================================
# 14. FINAL INSIGHT
# ============================================================

print("""
FINAL INSIGHT PENGUINS EDA:

1. Dataset penguins memiliki kombinasi fitur numerik dan kategorikal.
   Fitur numerik utama adalah bill_length_mm, bill_depth_mm,
   flipper_length_mm, dan body_mass_g.
   Fitur kategorikal utama adalah species, island, dan sex.

2. Dataset memiliki missing value pada beberapa kolom.
   Kolom sex memiliki missing value paling banyak,
   sedangkan species dan island tidak memiliki missing value.

3. Tidak terdapat data duplikat pada dataset.

4. Species terbanyak adalah Adelie.
   Island dengan jumlah data paling banyak adalah Biscoe.
   Kolom sex memiliki kategori male, female, dan beberapa missing value.

5. Distribusi fitur numerik menunjukkan variasi ukuran tubuh penguin.
   Perbedaan distribusi ini kemungkinan dipengaruhi oleh perbedaan species.

6. Heatmap menunjukkan bahwa flipper_length_mm dan body_mass_g
   memiliki korelasi positif kuat.
   Artinya, penguin dengan flipper lebih panjang cenderung memiliki
   body mass yang lebih besar.

7. Missing value repair dilakukan di akhir EDA.
   Kolom numerik diisi menggunakan median,
   sedangkan kolom kategorikal sex diisi menggunakan mode.

8. Setelah missing value repair, seluruh missing value berhasil diisi.
   Jumlah baris dataset tetap sama karena kita menggunakan imputasi,
   bukan menghapus data dengan dropna().

9. Outlier dicek menggunakan metode IQR.
   Jika tidak ada outlier, berarti data numerik masih berada dalam batas wajar
   berdasarkan Q1, Q3, dan IQR.

10. Dataset penguins cocok untuk latihan EDA karena memiliki:
    - data numerik
    - data kategorikal
    - missing value
    - korelasi antar fitur
    - distribusi berdasarkan species
    - proses missing value repair
""")