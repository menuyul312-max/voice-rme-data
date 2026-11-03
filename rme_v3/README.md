# RME ASR v3 — panduan penggunaan

Paket ini memperbaiki training dan benchmark **transkripsi/dikte medis Bahasa Indonesia**. Hasil ASR adalah draft yang diperiksa dokter. Tidak ada integrasi SIMRS, pembuat diagnosis, atau koreksi dosis otomatis.

## Mulai
1. Baca `LAPORAN_AUDIT.md`. Nama arsip “gold” belum berarti label sudah diverifikasi klinisi.
2. Upload `train_whisper_turbo_rme_v3.ipynb` ke Colab. Pilih GPU A100 (lebih cepat), L4, atau T4; gunakan runtime baru. Jalankan seluruh sel dan izinkan mount Google Drive.
3. Notebook mengunduh **arsip Drive yang diberikan**, memeriksa SHA-256, mengekstrak, mengaudit audio, mengarantina masalah, lalu melatih LoRA. Jangan memakai folder/checkpoint v2 untuk melanjutkan run ini.
4. Hasil berada di `MyDrive/rme_asr_v3/runs/<fingerprint>/`. Biarkan `split_plan.json` dan `latest_model.json` tetap bersama. Checkpoint tersimpan di Drive. Pemutusan sesi bisa kehilangan pekerjaan sejak checkpoint terakhir; resume mensyaratkan checkpoint lengkap.
5. Buka `benchmark_rme_v3_colab_t4.ipynb` di runtime T4 baru; Run all. Base dan model baru diuji dengan audio, decoding, dan normalisasi identik.
6. Baca `benchmark_report.json`; isi lembar referensi buta melalui dua reviewer dan adjudikasi. Unggah sebagai `blind_reference_review_filled.csv` di folder benchmark, lalu jalankan ulang scoring. Cache inference akan dipakai bila fingerprint cocok.

Kedua notebook **mandiri**: modul dan tes sudah tertanam. File `.py` terpisah tersedia untuk review/version control; tidak perlu di-upload ke Colab. Notebook tidak memiliki hasil training/inference lama yang dapat disalahartikan sebagai hasil baru.

## Konfigurasi awal
- Whisper `openai/whisper-large-v3-turbo`; commit base dikunci pada revisi yang dipakai audit tokenizer.
- LoRA r=16, alpha=32 pada q/k/v/out attention; LR 5e-5, maksimal 3 epoch, early stopping berdasarkan WER dev. Ini resep awal untuk diuji, bukan jaminan lebih baik dari r=32 v2.
- Batch efektif 32 untuk semua GPU; microbatch T4=2, L4=4, A100 40GB=8. AMP, SDPA, dan gradient checkpointing non-reentrant.
- Label/audio tidak dipotong diam-diam; max label 448 token setelah BOS dihapus; waveform 0,2–30 detik.
- Mild noise dan gain hanya pada train. Augmentasi speed via interpolasi lama dihapus karena mengubah pitch dan bukan time-stretch yang terkontrol.
- Evaluasi checkpoint maksimal 600 klip dev dengan sampling per grup. Test tidak dipakai memilih checkpoint. Epoch/batch/eval identik secara prinsip lintas perangkat; waktu aktual perlu diukur.
- Gunakan `WORK` baru untuk dataset/protokol baru. Setelah OOM, ubah batch dan accum dengan hasil kali 32; perubahan konfigurasi membentuk fingerprint run baru. Jangan mengaku hasil run baru identik dengan resume run lama.

## Berkas hasil audit
- `audit_summary.json`: angka hasil audit arsip penuh.
- `quarantine_review.csv`: kasus yang perlu dicocokkan kembali dengan audio, bukan daftar kesalahan klinis yang sudah dipastikan.
- `duplicate_audio_groups.csv`: kelompok PCM identik, termasuk konflik referensi.
- `LAPORAN_AUDIT.md`: interpretasi, bukti, perbaikan, dan batas evaluasi.

## Validasi yang diperlukan sebelum pilot enterprise
Bangun test klinis eksternal dengan dokter, pasien, sesi, ruangan, mikrofon dan lokasi yang tidak terlihat saat training. Pisahkan dikte dokter dari percakapan ambient; model ini belum memberi jaminan diarization atau atribusi dokter/pasien. Siapkan referensi verbatim yang didengar ulang oleh dua reviewer, terutama obat, dosis, angka, satuan, alergi, negasi dan subjek/temporality.

Tetapkan ambang absolut bersama pemilik klinis sebelum menguji test: WER/CER, tingkat error kritis beserta CI dan denominator, p95 latency batch=1 pada mesin target, hasil silence/noise dan rekaman panjang, serta hasil tiap aksen/perangkat/kondisi. Perbaikan relatif terhadap base tidak cukup. Laporkan juga coverage dan semua audio yang ditolak. Jangan hanya melaporkan subset yang mudah.

Opsional `clinical_terms.json` di WORK: list istilah klinis yang disepakati sebelumnya untuk precision/recall leksikal. Opsional `challenge_manifest.json` plus `challenge_audio/` mengaktifkan uji silence/noise dan rekaman >30 detik. Lihat format pada notebook. Challenge long-form memakai overlap-chunking dan masih harus divalidasi; tidak mengklaim streaming atau latency produksi.

Status notebook tetap `NOT_READY` sampai bukti eksternal dan review klinis tersedia. Itu penilaian kelengkapan bukti, bukan prediksi bahwa model pasti gagal. Hak penggunaan komersial setiap sumber dan persetujuan rekaman klinis perlu terdokumentasi sebelum produk dijual; notebook tidak dapat membuktikannya dari nama dataset.

## Batas pengujian paket ini
Logika CPU, struktur notebook, sintaks sel dan audit audio diuji di lingkungan kerja. Training GPU, kompatibilitas semua kombinasi runtime Colab, kualitas model baru dan SLA produksi **belum dijalankan/terukur**. Jika Colab meminta restart setelah instalasi, restart lalu Run all. Torch/CUDA bawaan tidak diganti untuk menghindari mismatch driver.
