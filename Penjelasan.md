# Penjelasan Proyek: Simulasi Pemrosesan Sinyal "CascadeFilter"

Dokumen ini berisi penjelasan menyeluruh mengenai proyek simulasi CascadeFilter, mencakup arsitektur kode, cara kerja, instruksi untuk menjalankan program, serta analisis hasil output yang akan diperoleh.

---

## 1. Ringkasan dan Tujuan Proyek
Proyek ini mengimplementasikan algoritma pemrosesan sinyal digital di mana sebuah sinyal dilewatkan melalui serangkaian filter matriks berantai. Tujuan utamanya adalah mencari cara paling efisien untuk mengalikan matriks-matriks transformasi ini menjadi satu matriks gabungan sebelum diaplikasikan pada sinyal.

Dua teknik utama yang digunakan untuk mencapai efisiensi tertinggi adalah:
1. **Dynamic Programming (DP)**: Digunakan untuk menentukan urutan perkalian (pengelompokan tanda kurung) matriks yang menghasilkan jumlah komputasi paling minimal.
2. **Paralelisme (OpenMP)**: Digunakan untuk mempercepat proses komputasi perkalian matriks itu sendiri dengan memanfaatkan arsitektur multi-core pada prosesor modern.

## 2. Arsitektur dan Modul Kode
Proyek ini ditulis dalam bahasa C dan dibagi menjadi beberapa modul agar rapi dan mudah di-maintain:
*   `src/main.c`: File utama (driver) yang mengatur alur logika dari awal hingga akhir, menghasilkan data acak, menjalankan berbagai metode algoritma, dan melakukan pengukuran waktu (benchmarking).
*   `src/dp_order.c`: Berisi algoritma **Bottom-Up Dynamic Programming** untuk mencari urutan *parenthesization* (pengelompokan) matriks paling optimal (mencari nilai operasi paling kecil).
*   `src/mult_sequential.c`: Implementasi perkalian matriks standar $O(N^3)$ (secara berurutan / *naive*). Digunakan sebagai patokan (baseline) untuk memastikan hasil komputasi yang dioptimalkan benar.
*   `src/mult_parallel.c`: Mengimplementasikan pemrosesan paralel (*Data Parallelism*). Sebelum dikalikan, matriks kanan di-transposisi terlebih dahulu untuk menghindari *cache-miss* di memori RAM, selanjutnya loop $i$ dan $j$ diparalelkan menggunakan perintah `#pragma omp parallel for collapse(2)`.
*   `src/mult_parallel_task.c`: Mengimplementasikan pemrosesan paralel berbasis tugas (*Task Parallelism*). Program membagi eksekusi pohon perkalian matriks (kiri dan kanan) untuk dikerjakan secara bersamaan oleh thread yang berbeda.
*   `Makefile`: Skrip bantu untuk melakukan proses build dan kompilasi otomatis dengan gcc compiler dan optimasi `-O3`.

---

## 3. Cara Menjalankan Program

### Prasyarat (Requirements)
Sistem operasi Anda harus berbasis Linux/Unix atau macOS dan memiliki aplikasi berikut:
*   `gcc` compiler (dengan dukungan OpenMP).
*   `make` utilitas build.

### Langkah 1: Kompilasi Kode
Buka terminal dan navigasikan ke dalam direktori proyek ini, kemudian jalankan perintah:
```bash
make
```
Perintah ini akan membaca `Makefile`, membuat folder `obj/`, mengompilasi semua file `.c` di folder `src/`, menautkan perpustakaan matematika (`-lm`) dan OpenMP (`-fopenmp`), kemudian menghasilkan satu file *executable* bernama `cascade_filter`.

### Langkah 2: Menjalankan Program
Terdapat dua cara untuk menjalankan program: 

**Cara A: Menggunakan dimensi *Default* (Skenario Kecil)**
Sangat berguna untuk pengecekan cepat, program telah menyediakan dimensi *default* (4 matriks dengan ukuran 10x150, 150x20, 20x80, dan 80x10).
```bash
./cascade_filter
```

**Cara B: Menggunakan dimensi Kustom (Custom Input)**
Anda dapat menguji dengan ukuran atau jumlah matriks yang berbeda. Format perintahnya adalah:
`./cascade_filter <jumlah_matriks> <dimensi_0> <dimensi_1> ... <dimensi_N>`

Contoh, untuk 3 buah matriks dengan dimensi (100x50), (50x200), (200x10):
```bash
./cascade_filter 3 100 50 200 10
```

### Langkah 3: Menjalankan Pengujian Ekstensif (Skrip Bash)
Proyek ini juga dilengkapi dengan skrip otomatis untuk menguji performa pada berbagai ukuran matriks dan jumlah thread:
```bash
bash ./scripts/benchmark.sh
```

### Langkah 4: Membersihkan File Kompilasi
Untuk menghapus file *executable* dan folder *object* bekas kompilasi, jalankan:
```bash
make clean
```

---

## 4. Penjelasan dan Analisis Hasil Output (Skenario Manual)

Saat Anda menjalankan program dengan skenario matriks sedang (misalnya 6 matriks acak), Anda akan melihat analisis output terminal berikut ini:

### A. Pengurangan Operasi melalui DP
```text
DP Operations Reduction: 48.55%
```
*Kenapa & Bagaimana?* Perkalian matriks memiliki sifat asosiatif. Algoritma *Dynamic Programming* mencari posisi tanda kurung pengelompokan yang paling optimal sebelum mengalikan angkanya. DP sukses mendeteksi urutan yang memotong kebutuhan komputasi matematika CPU sebesar 48,55%. Efeknya, waktu eksekusi sekuensial turun drastis tanpa memerlukan *multi-core*.

### B. Validasi / Verifikasi Kebenaran
```text
Verification SUCCESS: All versions output match within epsilon 1e-06.
```
*Kenapa & Bagaimana?* Sebelum mengukur waktu eksekusi, program mengecek setiap elemen desimal (*float*) dalam matriks hasil pendekatan DP dan Paralel. Memastikan tidak ada *bug* ataupun masalah *race condition* akibat pemrosesan paralel (karena toleransi kesalahan berada di bawah 0.000001).

### C. Analisis Metrik Kinerja (Speedup)
```text
Metrik Speedup:
- DP-only Speedup      : 2.29 x
- Parallel-only Speedup: 3.43 x
- Total Speedup        : 7.84 x
- Parallel Efficiency  : 0.43 (threads=8)
```
*Kenapa & Bagaimana?*
*   **DP-only Speedup (2.29x)**: Berkat pengurangan 48% operasi matematika, kecepatan proses langsung naik 2 kali lipat murni hanya dengan mengubah urutan perkalian.
*   **Parallel-only Speedup (3.43x)**: Karena perkalian matriks berifat *embarrassingly parallel*, loop perkalian disebar ke 8 utas (thread). Kecepatannya hanya naik 3,4 kali (bukan 8 kali) karena adanya batas kecepatan transfer memori RAM (*memory bandwidth*) dan *overhead* manajemen thread.
*   **Total Speedup (7.84x)**: Akumulasi dari DP yang menyederhanakan matematika dikali dengan CPU memprosesnya keroyokan (2.29 * 3.43 ≈ 7.84). Kecepatan total melesat hampir 8 kali lipat!

### D. Pembuktian CascadeFilter
```text
Runtime - Apply one-by-one: 0.001006 s
Runtime - Apply combined  : 0.000023 s
```
*Kenapa & Bagaimana?* Mengalikan vektor sinyal dengan 1 buah matriks gabungan (hasil perkalian paralel tadi) jauh lebih instan dibandingkan harus melewatkan sinyal ke tiap-tiap filter matriks satu-per-satu. Tidak ada proses penulisan *cache memory* yang berulang-ulang, cocok untuk audio DSP waktu-nyata (*real-time*).

---

## 5. Analisis Lanjutan: Hasil Benchmark Ekstensif (Skrip `benchmark.sh`)

Skrip `benchmark.sh` melakukan ujian berlapis yang membongkar karakteristik sejati dari program Anda:

### 1. Skalabilitas Ekstrem pada Data Raksasa (Skenario 10 Matriks)
Pada saat diuji menggunakan ukuran matriks beresolusi hingga `2048x2048`, hasil kehebatan algoritma dan OpenMP baru benar-benar terlihat:
*   **Pengaruh Mengerikan DP**: Waktu awal eksekusi program tanpa DP membutuhkan waktu raksasa sekitar **40 Detik**. Hanya dengan algoritma DP (tanpa paralel), waktunya hancur lebur menjadi **~1,1 detik** (*Speedup DP-only:* ~36x). Ini membuktikan struktur efisiensi algoritmik (*Big-O Notation*) jauh mendominasi kecepatan perangkat keras.
*   **Skalabilitas Paralel Sempurna**: Ketika *thread* OpenMP dinaikkan, performa metode *Parallel Data* meningkat dengan sangat efisien:
    *   1 utas: 0.70 detik
    *   2 utas: 0.35 detik
    *   4 utas: 0.21 detik
    *   8 utas: 0.11 detik
*   **Total Speedup = 366.71x**: Ketika 8 core bekerja bahu-membahu dengan algoritma DP yang optimal, program berjalan nyaris **367 kali lebih cepat** (41 detik vs 0.11 detik).

### 2. Kenapa Efisiensi Paralel melebihi 100%?
Pada uji coba ini, efisiensi paralelnya melampaui angka 1.0 (misalnya 1.56). Mengapa *speedup* 1 utas OpenMP bisa 1,5 kali lebih cepat dari metode sekuensial naif? 
Hal ini terjadi karena kode *mult_parallel.c* melakukan trik **Transposisi Matriks (Transpose)** pada matriks sisi kanan sebelum dikalikan. Trik ini memutar akses memori kolom menjadi akses memori baris (*contiguous memory*), sehingga meminimalkan lompatan *cache miss* di memori RAM dan mempercepat waktu eksekusi meskipun tidak menggunakan skema paralelisasi penuh.

### 3. Fenomena Kegagalan *Task Parallelism*
Sepanjang *benchmark* raksasa ini, eksekusi `Parallel Task + DP` selalu berjalan konstan di angka **~0.6 hingga ~0.7 detik** berapapun jumlah *thread* yang diberikan (entah 1, 4, maupun 8). 
**Alasannya:** Paralelisme berbasis pohon tugas (*Task Tree*) bergantung pada asumsi bahwa bentuk pohon bersifat seimbang (*balanced*). Sayangnya, urutan kurung hasil optimasi matriks DP seringkali menghasilkan wujud "rantai lurus" linier, di mana setiap matriks bergantung pada hasil operasi cabang sebelumnya secara ketat. Akibatnya utas-utas tidak bisa mengeksekusi operasi tersebut bersama-sama dan sistem hanya membuang-buang memori tambahan untuk mengatur pembuatan tugas OpenMP yang sia-sia. Hal ini menjadikan **Data Parallel** sebagai pemenang absolut dalam proyek ini.
