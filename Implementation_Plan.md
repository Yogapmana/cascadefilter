# Implementation Plan — CascadeFilter (v2, Gabungan)
## Simulasi Pemrosesan Sinyal dengan Matrix Chain Multiplication Optimization & Parallel Computing

**Mata Kuliah:** Algoritma & Pengolahan Paralel — ATA 2025/2026
**Studi Kasus:** 2 (CascadeFilter)
**Timeline:** ~10 hari, dikumpulkan minggu ke-13

Versi ini menggabungkan draft v1 (timeline, pemisahan metrik) dengan draft alternatif (model matematis vektor, cache-transpose optimization). Tujuh gap yang ditemukan saat perbandingan kedua draft **sudah diputuskan** di bawah — bukan dibiarkan terbuka untuk didiskusikan ulang.

---

## 0. Keputusan yang Sudah Dikunci

| # | Isu | Keputusan | Alasan |
|---|---|---|---|
| 1 | Kontradiksi topologi filter (gain/echo = persegi vs model d_{i-1}×d_i) | **Filter direpresentasikan sebagai filter bank multi-stage**, bukan gain/echo/EQ standar. Lihat Bagian 1.2. | Gain dan delay/echo filter itu memang persegi (K×K) — kalau dipakai sebagai contoh literal, DP jadi tidak relevan (semua urutan sama biayanya). Filter bank secara natural mengubah dimensi representasi tiap stage, sehingga variasi biaya antar urutan menjadi nyata dan DP punya fungsi. |
| 2 | Metrik speedup ambigu (T_seq baseline atau DP-seq?) | **Wajib dipisah 3 metrik**: speedup DP-only, speedup paralel-only, speedup total. Lihat Bagian 7. | Kalau cuma satu angka speedup total, tidak bisa dibuktikan DP itu sendiri berkontribusi berapa persen — rawan disalahartikan sebagai speedup paralel murni. |
| 3 | Hybrid task+data parallelism dijanjikan tapi kode contoh cuma data-parallel | **Diimplementasikan dua varian eksplisit**: `multiply_parallel_data()` (data-parallel, wajib) dan `execute_tree_parallel()` (task-parallel via `omp task`, opsional/bonus). Tidak boleh diklaim "hybrid" kalau cuma salah satu yang jalan. | Klaim di laporan harus match implementasi aktual — sesuatu yang disebut tapi tidak dibuktikan kodenya adalah celah pertama yang diserang saat tanya jawab. |
| 4 | Representasi matriks pakai `vector<vector<double>>` | **Ganti ke array 1D kontigu** (`double* data`, indexing manual `data[i*cols+j]`). | Nested vector punya alokasi heap per-baris terpisah — bertentangan dengan tujuan optimasi cache (transposisi) yang dijanjikan. Array 1D memberi locality yang konsisten dengan klaim optimasi. |
| 5 | MPI tidak disebut sama sekali | **Eksplisit: OpenMP wajib, MPI tidak dikerjakan.** Dicatat sebagai keputusan sadar di laporan, bukan kelalaian. | Soal bilang "OpenMP *atau* MPI" — memilih satu itu sah, tapi harus terlihat sebagai keputusan, bukan sesuatu yang lupa. |
| 6 | Dua peran (Dev Paralel & Analis) sama-sama pegang `matrix_op.cpp` | **File dipecah**: `mult_sequential.c`, `mult_parallel.c` terpisah, masing-masing pemilik jelas. | Dua orang edit file sama secara paralel adalah sumber merge conflict yang bisa dihindari sejak desain struktur folder. |
| 7 | Tidak ada timeline/checklist di draft alternatif | **Timeline 10 hari + deliverables checklist dipertahankan dari v1**, ditambah checkpoint verifikasi wajib. | Spesifikasi teknis tanpa rencana kerja bukan implementation plan, itu cuma desain algoritma. |

---

## 1. Model Matematis & Representasi

### 1.1 Definisi Formal

- **Vektor sinyal audio X**: berukuran K×1, domain waktu, sample audio mentah.
- **Matriks transformasi filter M_i**: berukuran `d_{i-1} × d_i`, merepresentasikan satu stage filter bank.
- **Matriks gabungan**: `M_gabungan = M_1 × M_2 × ... × M_n`
- **Output akhir**: `Y = M_gabungan × X`

### 1.2 Justifikasi Topologi — Filter Bank Multi-Stage

**Ini bukan gain/delay/EQ standar** (yang secara alami persegi K×K dan akan membuat DP percuma). Model yang dipakai:

```
Stage 0 (input)     : K = 256   sample per frame
Stage 1 (analysis)  : 512       — dekomposisi ke sub-band frekuensi (naik)
Stage 2 (analysis)  : 512       — pemrosesan per sub-band
Stage 3 (synthesis) : 128       — kombinasi sub-band (turun)
Stage 4 (synthesis) : 128
Stage 5 (output)    : 64        — representasi akhir terkompresi
```

Filter M_i menghubungkan `dim[i-1]` ke `dim[i]`. Ini konsisten dengan filter bank riil di DSP (misalnya QMF/wavelet filter bank) yang memang mengubah jumlah channel/bin di tiap tahap dekomposisi-rekombinasi. **Koefisien filter tetap diinisialisasi acak namun terstruktur** (nilai dominan di sekitar diagonal untuk mensimulasikan lokalitas frekuensi), tapi dimensinya mengikuti topologi di atas — bukan persegi.

---

## 2. Arsitektur & Aliran Data

```
[ Generator: topologi filter bank + vektor X ]
               │
               ▼
   [ DP: Matrix Chain Order ] ──► s[i][j] (tabel split optimal)
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
[Baseline] [Seq+DP]  [Parallel+DP]
 naif        s[i][j]   s[i][j] + OpenMP
 kiri→kanan  sequential + cache transpose
     │         │         │
     └─────────┼─────────┘
               ▼
      [ Verifikasi: epsilon 1e-6 ]
               ▼
      [ Y = M_gabungan × X, ukur runtime apply ]
               ▼
      [ Benchmark: 3 skenario × thread count ]
```

---

## 3. Modul DP (Matrix Chain Order)

```c
// dp_order.h / dp_order.c
typedef struct { int **m; int **s; int n; } DPResult;

DPResult compute_dp_order(int *dims, int n);
// m[i][j] = biaya minimum chain i..j
// s[i][j] = titik split optimal

Matrix* execute_optimal_order(Matrix **matrices, int **s, int i, int j,
                                Matrix* (*multiply_fn)(Matrix*, Matrix*));
```

`multiply_fn` sebagai function pointer — satu implementasi DP dipakai ulang oleh versi sequential dan parallel, hanya beda fungsi kali-matriksnya. Ini mencegah duplikasi logika DP di banyak file (sumber inkonsistensi paling umum).

**Kompleksitas:** O(n³) waktu, O(n²) memori. DP **tidak diparalelkan** — state `m[i][j]` bergantung pada `m[i][k]` dan `m[k+1][j]` yang harus selesai lebih dulu (dependency chain). Ini fakta algoritmik yang harus dinyatakan eksplisit di laporan, bukan disembunyikan seolah keterbatasan implementasi.

---

## 4. Modul Perkalian Matriks

### 4.1 Representasi Matriks (Array 1D Kontigu)

```c
typedef struct {
    int rows, cols;
    double *data;  // data[i*cols + j], bukan double**
} Matrix;

Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix *m);
```

### 4.2 Baseline & Sequential+DP

```c
Matrix* multiply_sequential(Matrix *A, Matrix *B) {
    Matrix *C = matrix_create(A->rows, B->cols);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < B->cols; j++) {
            double sum = 0;
            for (int k = 0; k < A->cols; k++)
                sum += A->data[i*A->cols+k] * B->data[k*B->cols+j];
            C->data[i*C->cols+j] = sum;
        }
    return C;
}
```

### 4.3 Parallel: Data-Parallelism + Cache Transpose (wajib)

```c
Matrix* transpose(Matrix *B) {
    Matrix *BT = matrix_create(B->cols, B->rows);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < B->rows; i++)
        for (int j = 0; j < B->cols; j++)
            BT->data[j*BT->cols+i] = B->data[i*B->cols+j];
    return BT;
}

Matrix* multiply_parallel_data(Matrix *A, Matrix *B) {
    Matrix *BT = transpose(B);        // akses kolom B jadi akses baris BT
    Matrix *C = matrix_create(A->rows, B->cols);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < B->cols; j++) {
            double sum = 0;
            for (int k = 0; k < A->cols; k++)
                sum += A->data[i*A->cols+k] * BT->data[j*BT->cols+k]; // kontigu
            C->data[i*C->cols+j] = sum;
        }
    matrix_free(BT);
    return C;
}
```

`schedule(static)` dipakai karena beban kerja tiap iterasi seragam (tidak ada branching). Transpose sendiri juga diparalelkan — kalau tidak, dia jadi bottleneck sequential yang mengurangi manfaat paralelisasi keseluruhan, terutama untuk matriks besar.

### 4.4 Parallel: Task-Parallelism di Level Pohon (opsional/bonus)

```c
Matrix* execute_tree_parallel(Matrix **mats, int **s, int i, int j) {
    if (i == j) return mats[i];
    Matrix *left, *right;
    #pragma omp task shared(left)
    left = execute_tree_parallel(mats, s, i, s[i][j]);
    #pragma omp task shared(right)
    right = execute_tree_parallel(mats, s, s[i][j]+1, j);
    #pragma omp taskwait
    return multiply_parallel_data(left, right);
}
```

Dipanggil dari dalam `#pragma omp parallel` + `#pragma omp single`. **Ini terpisah dari 4.3, bukan pengganti** — kalau dikerjakan, laporan harus membandingkan tiga varian paralel (data-only, task-only, hybrid), bukan cuma klaim "hybrid" tanpa data pembanding.

---

## 5. Aplikasi ke Sinyal & Verifikasi

### 5.1 Verifikasi Numerik (wajib, terintegrasi ke pipeline — bukan manual)

```c
int verify_equal(Matrix *A, Matrix *B, double epsilon) {
    if (A->rows != B->rows || A->cols != B->cols) return 0;
    for (int idx = 0; idx < A->rows * A->cols; idx++)
        if (fabs(A->data[idx] - B->data[idx]) > epsilon) return 0;
    return 1;
}
```

Dipanggil otomatis setiap run, epsilon = 1e-6 (toleransi akumulasi floating point akibat urutan operasi berbeda antar versi). Kalau gagal, program **exit dengan error**, tidak lanjut mencatat waktu eksekusi seolah valid.

### 5.2 Apply Filter ke Sinyal — Bukti Value Proposition

```c
// Y = M_gabungan × X
Vector* apply_filter(Matrix *M_combined, Vector *X);
```

Ukur dan bandingkan:
- (a) menerapkan filter satu-per-satu ke X (n perkalian matriks-vektor)
- (b) menerapkan M_gabungan langsung ke X (1 perkalian matriks-vektor)

Ini bagian yang menjawab **kenapa CascadeAudio butuh sistem ini**: bukan cuma "kita bisa menghitung matriks gabungan lebih cepat," tapi "sistem produksi memproses tiap sinyal masuk lebih cepat karena filter sudah digabung di awal." Tanpa bagian ini, laporan cuma benchmark abstrak tanpa justifikasi bisnis.

---

## 6. Spesifikasi Tiga Versi Program

| Spesifikasi | Baseline | Sequential+DP | Parallel+DP |
|---|---|---|---|
| Urutan perkalian | Kiri→kanan, naif | Optimal (DP) | Optimal (DP) |
| Eksekusi | Single thread | Single thread | OpenMP multi-thread |
| Optimasi memori | Tidak ada | Tidak ada | Transpose B (cache locality) |
| Representasi matriks | Array 1D | Array 1D | Array 1D |
| Tujuan | Tolok ukur dasar | Isolasi kontribusi DP | Isolasi kontribusi paralel + cache |

---

## 7. Metrik & Skenario Benchmark

### 7.1 Metrik (wajib dipisah, tidak boleh cuma satu angka speedup)

| Metrik | Formula | Mengisolasi |
|---|---|---|
| Reduksi operasi DP | `(ops_naif - ops_dp) / ops_naif × 100%` | Manfaat algoritmik murni |
| Speedup DP-only | `T_baseline / T_sequential_dp` | Kontribusi DP saja |
| Speedup paralel-only | `T_sequential_dp / T_parallel_dp` | Kontribusi paralelisasi + cache saja |
| Speedup total | `T_baseline / T_parallel_dp` | Angka headline, gabungan keduanya |
| Efisiensi | `Speedup_paralel / jumlah_thread` | Kualitas scaling terhadap core |

Kalau reduksi operasi DP kecil (<20%), itu sinyal topologi filter bank kurang bervariasi — harus diperbaiki sebelum lanjut benchmark besar-besaran, bukan dilaporkan apa adanya seolah hasil final.

### 7.2 Skenario Dataset (angka konkret, reproducible)

**Skenario Kecil — Verifikasi Fungsionalitas:**
- 4 filter, `p = [10, 150, 20, 80, 10]`
- Tujuan: cek kebenaran DP + verifikasi output identik antar versi (harus lolos sebelum lanjut)

**Skenario Sedang — Kontribusi DP:**
- 6 filter, `p = [256, 512, 512, 128, 128, 64]` (mengikuti topologi Bagian 1.2)
- Tujuan: ukur selisih baseline vs sequential+DP secara jelas

**Skenario Besar — Skalabilitas Paralel:**
- 10-16 filter, dimensi diperbesar proporsional dari topologi Bagian 1.2 (misal ×4)
- Thread: 1, 2, 4, 8
- Tujuan: speedup dan efisiensi paralel, termasuk titik di mana overhead thread mulai dominan (matriks kecil → speedup bisa turun, itu temuan valid, bukan bug)

---

## 8. Struktur Proyek

```
cascadefilter/
├── Makefile
├── README.md
├── include/
│   ├── matrix.h            # struct Matrix (array 1D), alokasi/dealokasi
│   ├── dp_order.h
│   └── verify.h
├── src/
│   ├── main.c               # orkestrasi: generate → DP → 3 versi → verifikasi → benchmark
│   ├── signal_gen.c         # generate vektor X + topologi filter bank
│   ├── dp_order.c           # DP matrix chain order (satu sumber kebenaran)
│   ├── mult_sequential.c    # baseline + sequential+DP (owner: Analis)
│   ├── mult_parallel.c      # OpenMP data-parallel + transpose (owner: Dev Paralel)
│   ├── mult_parallel_task.c # OpenMP task-parallel, opsional (owner: Dev Paralel)
│   ├── verify.c             # validasi epsilon, gate wajib tiap run
│   └── apply_filter.c       # Y = M_gabungan × X, ukur runtime
├── scripts/
│   ├── generate_input.py
│   ├── benchmark.sh
│   └── plot_results.py
├── data/
├── results/
└── docs/
```

File `mult_sequential.c` dan `mult_parallel.c` **terpisah** — dua peran tidak lagi edit file yang sama secara bersamaan.

---

## 9. Makefile

```makefile
CC = gcc
CFLAGS = -O3 -Wall -std=c11 -fopenmp
INCLUDES = -Iinclude
TARGET = cascade_filter
SRC_DIR = src
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
```

---

## 10. Timeline 10 Hari

| Hari | Fokus | PIC |
|---|---|---|
| 1 | Kunci topologi (sudah selesai di Bagian 1.2), setup repo, rencana kerja | Semua + Manajer |
| 2–3 | `dp_order.c` + unit test manual (n=4); `mult_sequential.c`; `signal_gen.c` | Arsitek DP, Analis |
| 4–5 | `mult_parallel.c` (data+transpose); integrasi DP+sequential dan DP+parallel; **checkpoint verifikasi wajib** | Dev Paralel, semua |
| 6 | `apply_filter.c`; `mult_parallel_task.c` jika waktu memungkinkan (bonus) | Analis, Dev Paralel |
| 7–8 | Benchmarking 3 skenario × 4 thread count, catat CSV | Analis |
| 9 | `plot_results.py`; tulis Bagian Hasil & Analisis laporan berdasarkan data aktual | Manajer, Analis |
| 10 | Review kode silang, finalisasi README/Makefile/laporan | Semua |

**Checkpoint hari 5 adalah gate keras**: kalau verifikasi belum lolos (output 3 versi belum identik dalam epsilon), jangan mulai benchmarking. Geser: potong `mult_parallel_task.c` (bonus) dulu, prioritas ke tiga versi wajib yang benar.

---

## 11. Deliverables Checklist

**Kode:**
- [ ] `mult_sequential.c` (baseline + sequential+DP)
- [ ] `mult_parallel.c` (data-parallel + transpose)
- [ ] `mult_parallel_task.c` — opsional
- [ ] `dp_order.c` — satu sumber kebenaran, dipakai via function pointer
- [ ] `verify.c` — gate otomatis tiap run
- [ ] `apply_filter.c`
- [ ] Makefile, README.md

**Laporan:**
- [ ] Justifikasi topologi filter bank (Bagian 1.2) — wajib ada, ini yang menutup celah terbesar
- [ ] DP: rekurensi + contoh angka dari skenario sedang (bukan generik)
- [ ] Paralel: data-parallel + cache-transpose, dengan bukti before/after cache (opsional: hitung cache miss kalau ada tooling `perf`)
- [ ] Tiga metrik speedup terpisah (DP-only, paralel-only, total) + efisiensi
- [ ] Bagian apply-to-signal: bukti value proposition bisnis
- [ ] Kesimpulan dikaitkan balik ke kebutuhan CascadeAudio
- [ ] Pembagian tugas tiap anggota (wajib sesuai instruksi soal)

---

## 12. Risiko yang Masih Berlaku

| Risiko | Mitigasi |
|---|---|
| Topologi filter bank ternyata masih kurang variatif → reduksi DP kecil | Uji reduksi operasi di hari 3, sebelum lanjut ke implementasi paralel |
| Checkpoint verifikasi hari 5 gagal | Potong task-parallel (bonus), jangan potong verifikasi |
| Transpose jadi bottleneck sequential kalau tidak diparalelkan | Sudah ditangani di 4.3 — transpose ikut `parallel for` |
| Overhead OpenMP dominan di matriks kecil | Dokumentasikan sebagai temuan valid, bukan disembunyikan |
| Klaim "hybrid parallelism" tanpa data pembanding | Kalau `mult_parallel_task.c` tidak selesai, jangan sebut hybrid di laporan — sebut data-parallel saja |
