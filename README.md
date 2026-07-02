# CascadeFilter

**CascadeFilter** adalah simulasi sistem pemrosesan sinyal yang mengimplementasikan **Matrix Chain Multiplication (MCM) Optimization** dan komputasi paralel menggunakan **OpenMP**. Proyek ini mendemonstrasikan bagaimana optimasi algoritma yang dipadukan dengan teknik paralelisasi dapat memberikan peningkatan performa (_speedup_) ekstrem (hingga lebih dari 1000x lipat) pada sistem _filter bank_ bertingkat (_multi-stage filter bank_).

## 🚀 Fitur Utama

- **Matrix Chain Order Dynamic Programming (DP):** Menentukan urutan perkalian matriks yang paling efisien untuk meminimalkan jumlah perkalian skalar, memangkas operasi matematika secara dramatis (hingga >90%).
- **Cache-Optimized Data Parallelism:** Menghindari masalah _cache miss_ dalam perkalian matriks dengan melakukan transposisi pada matriks kedua (`B`) terlebih dahulu, yang kemudian diparalelkan menggunakan `OpenMP for loop collapse`.
- **Tree-based Task Parallelism:** Variasi opsional eksekusi paralel pada level _task_ yang mengeksekusi cabang matriks sebelah kiri dan kanan dari pohon urutan komputasi secara bersamaan (`omp task`).
- **Verifikasi Numerik Otomatis:** Sistem yang terotomatisasi penuh memastikan _output_ yang dihasilkan dari seluruh versi paralel dan sekuensial adalah sama dan sesuai dalam batas toleransi kesalahan *floating point* (*relative epsilon tolerance*).
- **Benchmark & Analytics:** Tiga skenario benchmark lengkap dengan pemisahan matriks pengukuran _speedup_ untuk algoritma DP saja, performa Paralel saja, dan total efisiensi _threading_.

## 📁 Struktur Direktori

```text
cascadefilter/
├── Makefile                 # Konfigurasi kompilasi GNU Make
├── README.md                # Dokumentasi Proyek (file ini)
├── include/
│   ├── dp_order.h           # Header Dynamic Programming MCM
│   ├── matrix.h             # Header representasi Vector & Matrix array 1D
│   └── verify.h             # Header komparasi dan verifikasi Epsilon
├── scripts/
│   ├── benchmark.sh         # Script shell orkestrasi 3 skenario tes
│   ├── generate_input.py    # Generator input berdimensi pseudo-random
│   └── plot_results.py      # Script pembuat grafik mock benchmark
└── src/
    ├── apply_filter.c       # Penerapan matriks filter gabungan ke sinyal
    ├── dp_order.c           # Implementasi MCM DP Algorithm
    ├── main.c               # Titik masuk program (benchmark & reporting)
    ├── matrix.c             # Manajemen memori alokasi matriks (1D-contiguous)
    ├── mult_parallel.c      # Implementasi Data-Parallelism + Transpose dengan OpenMP
    ├── mult_parallel_task.c # Implementasi Task-Parallelism dengan OpenMP
    ├── mult_sequential.c    # Implementasi Baseline & Sequential DP
    ├── signal_gen.c         # Pembangkit matriks random untuk topologi filter bank
    └── verify.c             # Logika validasi epsilon
```

## 🛠 Panduan Instalasi & Eksekusi

### Kebutuhan Sistem
- Kompilator yang mendukung spesifikasi C11 (misalnya, `GCC` atau `Clang`).
- Dukungan library `OpenMP`.
- `Make` utility.

### Kompilasi Proyek
Masuk ke direktori proyek dan jalankan `make`:
```bash
make clean
make
```

### Eksekusi Manual Terarah
Anda bisa memberikan argumen eksekusi berformat `./cascade_filter <jumlah_filter> <dim_0> <dim_1> ... <dim_N>`.
Contoh untuk 5 matriks filter:
```bash
./cascade_filter 5 256 512 512 128 128 64
```
_Jika dijalankan tanpa argumen, program akan memakai konfigurasi verifikasi Small Scenario._

### Eksekusi Seluruh Skenario Benchmark (Otomatis)
Jalankan script test benchmarking yang kami sediakan untuk menguji proyek secara terukur:
```bash
./scripts/benchmark.sh
```

## 📊 Detail Implementasi Teknikal

Program ini menyajikan 3 strategi utama perkalian sekumpulan matriks filter:

1. **Baseline / Naive Sequential (`mult_sequential.c`)**
   Pendekatan O(N³) standar yang mengalikan rentetan filter matriks secara sekuensial (_left-to-right_) dari filter ujung kiri ke kanan tanpa optimasi cache maupun urutan.

2. **Sequential + DP (`dp_order.c` & `mult_sequential.c`)**
   Pendekatan optimal secara algoritmik. Pertama, *Dynamic Programming* membangun matriks biaya komputasi yang menentukan lokasi penyekatan perkalian. Setelah urutan (*binary tree*) perkalian diperoleh, program memanggil algoritma perkalian sekuensial biasa untuk tiap nodenya.

3. **Data Parallel + Cache Transpose (`mult_parallel.c`)**
   Implementasi perkalian matriks yang optimal secara infrastruktur *hardware*. Sebelum mengalikan `A * B`, matriks `B` ditranspos ke matriks baru `BT`. Hal ini bertujuan menjadikan iterasi *loop* komputasi memiliki *contiguous memory access* (membaca baris demi baris alih-alih kolom memori yang saling berjauhan). Loopnya diparalelkan secara hibrida menggunakan instruksi _pragma_ OpenMP `#pragma omp parallel for collapse(2)`.

## 📈 Metrik Performa (Hasil Benchmark)

Pengujian pada Skenario **Large** (Topologi `10` filter bank berukuran makro; ukuran `1024x2048`, `2048x2048` dll.) dengan `8 OpenMP threads` menunjukkan hasil:

- **Baseline Naive Time**: ~47.08 Detik
- **Optimasi Algoritmik (Sequential + DP) Time**: ~0.28 Detik *(DP Speedup = ~166x)*
- **Optimasi Hardware (Parallel Data + DP) Time**: ~0.033 Detik *(Parallel-only Speedup = ~8.49x)*
- **Total Speedup Keseluruhan (Baseline vs Parallel+DP)**: **~1409x Lebih Cepat**

Dalam Skenario *Large*, optimasi Dynamic Programming sendiri menekan hampir **~91.4%** iterasi komputasi mentah. Data performa lengkap terekam saat Anda mengeksekusi `./scripts/benchmark.sh`.
