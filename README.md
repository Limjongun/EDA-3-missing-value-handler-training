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
