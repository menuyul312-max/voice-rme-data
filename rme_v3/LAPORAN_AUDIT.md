# Audit dataset dan perbaikan notebook RME

Tanggal: 3 November 2026. Sasaran: transkripsi/dikte medis Bahasa Indonesia sebagai draft RME RS/klinik. Integrasi SIMRS tidak termasuk.

## Kesimpulan

**Dataset bermanfaat sebagai bahan pengembangan, tetapi belum dapat dianggap gold klinis atau bukti kesiapan enterprise.** Semua 32.604 file bisa didekode, namun terdapat duplikasi, konflik transkrip pada audio identik, ketidaksesuaian durasi, dan perubahan isi oleh proses pembersihan LLM. Dua notebook v3 menggunakan pemeriksaan konservatif dan evaluasi yang dapat ditelusuri. Peningkatan akurasi model baru belum dapat diklaim sebelum training dan benchmark GPU dijalankan.

## Cakupan audit nyata

Arsip Drive yang diperiksa: `gold_dataset.tar.gz`, 4.185.817.291 byte, SHA-256:
`1213c8df5ca041dbf7dcfb119cd2d9112d970ad0849cff2193f02d1dd65c68d6`.

Ketiga manifest berisi total **32.604 baris**, cocok dengan **32.604 file audio**; seluruh file didekode ke PCM float32 dengan libsndfile 1.2.2. Durasi diukur dari sampel dan sample rate, bukan diambil begitu saja dari manifest. Total **107,60 jam**, mono seluruhnya; 25.407 file 16 kHz, 2.877 file 24 kHz, 4.320 file 44,1 kHz. Rentang durasi aktual 1,08–26,121 detik. Tidak ada decode error, file >30 detik, atau file mendekati silent pada ambang RMS <1e-5.

| Sumber menurut manifest | Klip | Jam aktual |
|---|---:|---:|
| Level_2_Clinical_Education_OSCE | 7,063 | 25.74 |
| Level_3_Real_Hospital_Clinic_Dialog | 17,325 | 63.18 |
| fleurs_id_test | 204 | 0.59 |
| fleurs_id_train | 714 | 2.08 |
| fleurs_id_validation | 101 | 0.29 |
| mrqorib_medisco_test | 720 | 1.51 |
| mrqorib_medisco_train | 3,600 | 7.68 |
| synthetic_clinical_v3 | 2,877 | 6.54 |

Nama kategori “Real Hospital” berasal dari metadata dataset; audit ini tidak membuktikan setiap rekaman merupakan konsultasi/dikte klinis asli.

## Temuan prioritas

### 1. Pembersihan LLM mengubah label di luar tanda baca

24.345 klip (**74,67%**) memiliki perubahan leksikal antara `raw_text` tanpa tag bahasa dan `clean_text`, setelah normalisasi yang mempertahankan angka/desimal/negasi. Perubahan dapat mencakup perbaikan yang benar, perubahan gaya, atau kesalahan; **angka ini bukan tingkat error label**.

Pemeriksaan konservatif menemukan 3.929 klip dengan perubahan token angka, 266 dengan perubahan token satuan, dan 1.031 dengan perubahan token negasi. Gabungannya adalah **4.898 klip unik**. Penulisan angka menjadi kata atau mg menjadi miligram juga dapat memicu flag; label harus didengar ulang, bukan otomatis ditolak sebagai salah.

Contoh terukur `test/fleurs_train_0570.wav`: `raw_text` menyebut `6.500 pengunjuk rasa`, sedangkan `clean_text` menyebut `enam ratus lima puluh pengunjuk rasa`. `raw_text_before_llm` juga berbeda (`enam lima ratus`). Tanpa mendengarkan dan adjudikasi, tidak ada dasar memilih salah satunya sebagai gold.

**Perbaikan:** v3 memakai label sumber `raw_text` tanpa tag bahasa seperti `<ID-ID>`, tidak memakai parafrasa LLM sebagai target verbatim. Semua selisih angka/unit/negasi dikarantina. Label sumber pun masih dapat salah—pemilihan ini mencegah penambahan perubahan LLM, bukan menjadikan caption otomatis gold. Referensi benchmark perlu dikoreksi melalui review buta.

### 2. Audio duplikat dan satu konflik klinis nyata antarreferensi

Ada **1.535 file tambahan duplikat PCM**: 1.531 pasangan lintas Level 2–Level 3 dan 4 pasangan sintetis. Duplikasi lintas kategori memberi bobot berlebih pada rekaman yang sama; group_id yang berbeda tidak cukup mendeteksinya.

Satu pasangan identik memiliki label berbeda:
- `train/SYN_0005_TRN_01.mp3`: “Saya merasa perut mulas melilit ...”
- `train/SYN_0013_TRN_01.mp3`: “Saya merasa dada terasa sesak dan berat ...”

Karena audio identik, kedua target berbeda ini tidak dapat sama-sama menjadi transkripsi verbatim yang benar. **Keduanya dikarantina**, bukan memilih satu secara arbitrer.

Ada juga satu kelompok PCM identik lintas train/dev: `dev/SYN_0092_TRN_01.mp3` dan `train/SYN_0104_TRN_01.mp3`. Tidak ditemukan duplikat PCM train/test pada arsip ini. Exact PCM tidak mendeteksi seluruh rekaman ulang, re-encoding, overlap parsial, atau near-duplicate.

### 3. Seluruh FLEURS mempunyai durasi 2× berbeda

Semua **1.019 file FLEURS** berdurasi aktual **tepat 0,5 × durasi manifest**. Contoh `dev/fleurs_test_0002.wav`: aktual 9,48 detik, manifest 18,96 detik. Ini dapat berasal dari metadata yang salah atau tahap pemrosesan audio; audit belum menentukan penyebabnya. **Jangan mengubah sample rate atau meregangkan audio secara membabi buta.** Bandingkan dengan sumber dan dengarkan sebelum memperbaiki.

Total 1.089 file memenuhi karantina durasi `abs(actual-declared) > max(0,5 detik, 10% declared)`: 1.019 FLEURS dan 70 lainnya. Seluruh FLEURS saat ini terkarantina. ID `fleurs_spk_spk_train_1287` dan pola serupa berkorespondensi dengan ID klip; tidak cukup sebagai bukti identitas speaker. Jika data ini diperbaiki, v3 membatasinya ke train sampai metadata speaker sebenarnya tersedia.

### 4. Split lama lebih baik daripada acak klip, tetapi belum cukup

Tidak ditemukan overlap pada **group_id bawaan** antara train/dev/test. Ini temuan positif; tidak tepat mengatakan seluruh split lama bocor.

Namun:
- duplikat PCM train/dev di atas lolos dari group_id;
- **803 klip dev** memiliki referensi panjang (>=8 kata) identik dengan suatu referensi train; 17 klip test cocok dengan dev; tidak ada kecocokan panjang train/test dengan normalisasi yang dipakai;
- dev mengandung **374 audio sintetis**, sedangkan test tidak memiliki sintetis;
- speaker yang muncul pada video berbeda belum dapat diidentifikasi dari metadata video saja;
- Medisco memiliki banyak skrip yang sama dibaca speaker berbeda; pemisahan speaker saja tidak mengukur generalisasi ke kalimat baru.

**Perbaikan:** gabungkan komponen sumber/video/speaker/patient/session yang diketahui serta PCM identik; synthetic hanya train; hapus salinan PCM dan overlap teks panjang antarpartisi dengan prioritas test > dev > train. Split terkunci dengan fingerprint dan tidak dipilih berdasarkan skor model. Filtering ini mengubah populasi benchmark dan mengurangi data berulang; coverage serta jumlah yang dikeluarkan dilaporkan.

### 5. Sinyal clipping

246 klip memiliki >0,1% sampel beramplitudo absolut >=0,999. Ini **indikator**, terutama pada MP3 dapat dipengaruhi decoding/overshoot, bukan bukti pasti rekaman tidak dapat dipakai. Flag untuk didengar ulang; notebook tidak menghapusnya otomatis.

## Hasil kurasi v3 pada arsip ini

**5,763 klip unik dikarantina**. Alasan saling tumpang tindih sehingga tidak boleh dijumlahkan langsung. Tersisa 26,841 sebelum deduplikasi. Pemeriksaan tokenizer Whisper pada revisi `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` menguji seluruh label yang lolos: maksimum **112 token** setelah BOS dihapus; **0 label >448 token**.

Deduplikasi dan penghilangan overlap teks selanjutnya mengeluarkan **4,776 klip**. Rencana split setelah itu:

| Split | Klip | Jam | Grup independen yang diketahui |
|---|---:|---:|---:|
| train | 17,494 | 59.21 | 330 |
| dev | 2,477 | 9.05 | 43 |
| test | 2,094 | 6.97 | 34 |

Angka ini merupakan simulasi CPU atas inventory dan PCM yang sudah diaudit; notebook merekonstruksi serta memverifikasinya dari arsip. “Grup independen yang diketahui” tidak membuktikan seluruh speaker independen. Rincian sumber per split ada di JSON. Test tetap dataset internal, bukan cohort klinis eksternal.

## Apa yang diperbaiki dari notebook terlampir

| Masalah/risiko sebelumnya | Perubahan v3 |
|---|---|
| Download tetap ke rilis GitHub lama | Download arsip Drive yang diminta; SHA-256 tetap |
| Tidak memakai schema `raw_text`/`clean_text` arsip baru | Parser sesuai manifest dan kebijakan label eksplisit |
| Label dan waveform bisa dipotong diam-diam | Validasi durasi/token; tidak truncate target |
| Warning attention mask terlihat pada output training lama | Mask input dikirim saat training dan inference |
| RNG augmentasi bergantung PID | Seed worker dari Torch; seed dan konfigurasi dicatat |
| Speed perturbation via interpolasi mengubah pitch | Hapus default tersebut; mild noise/gain tanpa clipping tambahan |
| T4 otomatis hanya 1 epoch | Epoch maksimum sama; microbatch berbeda dengan batch efektif 32 |
| LoRA semua attention+MLP r32 cukup mahal | Attention r16 sebagai baseline efisien; wajib ablation dev untuk klaim lebih baik |
| Resume hanya mencari checkpoint terakhir | Run terikat data, split, kode, hyperparameter, paket dan revisi; cek file checkpoint |
| Kode training/benchmark berbeda versi | Modul tertanam yang sama persis di kedua notebook |
| CI sebagian berbasis klip yang saling berkorelasi | WER dan delta WER bootstrap pada kelompok sumber terhubung |
| Throughput batch dibaca sebagai kecepatan produk | Throughput dipisahkan dari p50/p95 latency batch=1; cakupan timing dijelaskan |
| Cache berpotensi tidak cocok dengan bobot/data baru | Fingerprint bobot, data, kode, decoding, Torch dan GPU |
| Target G7 lama relatif terhadap caption | WER referensi dokter dilaporkan terpisah; ambang klinis absolut perlu ditetapkan |
| SOAP dari LLM dapat dipakai seolah acuan klinis | Tidak menjadikan output LLM sebagai gold; fokus transkripsi RME |
| Klaim kelayakan dapat melampaui bukti | `NOT_READY` bila cohort klinis, error kritis, silence/noise, long-form atau SLA belum tervalidasi |

## Hasil lama versus hasil baru

Output tersimpan notebook training lama melaporkan A100 40GB, sekitar 7.944 detik training, best dev WER **19,26%**; pemeriksaan dev berikutnya base guarded **26,53%** versus v2 **19,02%**, atau beam fallback **25,57%** versus **18,51%**. Ini **hasil yang tercatat dalam notebook lama**, tidak direproduksi pada audit ini, dan bukan benchmark terhadap gold klinis independen. Jangan membandingkannya langsung dengan v3 setelah perubahan data, normalisasi dan split.

**Belum tersedia:** WER/CER model v3, perbaikan terhadap v2, runtime GPU v3, VRAM nyata, akurasi nama obat/dosis klinis, performa streaming/ambient/diarization, dan SLA produk. Tidak ada training GPU yang dijalankan dalam pekerjaan ini.

## Urutan menuju pilot yang masuk akal

1. Dengarkan dan adjudikasi `quarantine_review.csv`; selesaikan konflik audio-label dan penyebab durasi FLEURS. Simpan label verbatim, asal koreksi dan reviewer; jangan memakai LLM untuk mengarang bagian yang tidak terdengar.
2. Jalankan training v3 dan bandingkan resep (mis. r16 vs r32, LR, beam) **hanya pada dev**. Kunci konfigurasi sebelum test. Split test yang telah dipakai untuk tuning tidak lagi final holdout.
3. Bangun cohort dikte RS/klinik terpisah dengan speaker dan lokasi baru. Data pendidikan/video membantu adaptasi istilah, tetapi tidak menggantikan lingkungan produk nyata. Test juga perlu kelompok kontrol dengan ucapan biasa, noise, silence, koreksi diri dan audio panjang.
4. Buat referensi melalui dua reviewer; adjudikasi obat, dosis, angka, satuan, alergi, negasi, subjek dan waktu kejadian. Ukur error kritis dengan denominator dan CI, bukan hanya WER. Daftar istilah mengukur precision dan recall leksikal, bukan makna klinis.
5. Ukur p95 latency batch=1, throughput, cold-start, concurrency, durasi panjang dan kualitas perangkat target. Timing Colab tidak mencakup antrean/network/editor RME.
6. Tetapkan acceptance criteria bersama penanggung jawab klinis dan lakukan pilot dengan dokter meninjau serta mengesahkan draft. Hak komersial tiap sumber dan persetujuan data klinis harus dapat ditelusuri; belum diverifikasi oleh audit teknis ini.

## Validasi dan batas

Audit membaca seluruh audio, manifest dan tokenizer; tidak melakukan audit mendengar manual, transkripsi oleh dokter atau validasi klinis. Tes CPU mencakup pemisahan sumber, konflik audio, angka/desimal, negasi, coverage prediksi, bootstrap, ekspor aman, dan parser label sumber. Notebook disediakan tanpa output lama dan diuji struktur/sintaksnya. Kompatibilitas runtime GPU penuh masih perlu dibuktikan di Colab; jangan menyebut paket ini model enterprise siap jual.
