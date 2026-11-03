# Jalankan di Colab

1. [Buka training](https://colab.research.google.com/github/menuyul312-max/voice-rme-data/blob/main/rme_v3/train_whisper_turbo_rme_v3.ipynb)
2. Pilih Runtime → Change runtime type → GPU (A100 disarankan; L4/T4 didukung profil awal).
3. Klik Runtime → Run all dan izinkan Google Drive. Audit, karantina, split, training, dan ekspor berjalan berurutan. Jika diminta restart setelah pemasangan dependensi, restart lalu Run all lagi.
4. Setelah training selesai, [buka benchmark T4](https://colab.research.google.com/github/menuyul312-max/voice-rme-data/blob/main/rme_v3/benchmark_rme_v3_colab_t4.ipynb) dan Run all dengan akun Drive yang sama.

[Unduh paket lengkap ZIP](https://github.com/menuyul312-max/voice-rme-data/raw/refs/heads/main/rme_v3/rme_v3_notebooks_dan_audit.zip). Daftar karantina dan inventory audio CSV berada di dalam ZIP.

Dataset otomatis diunduh dari tautan Drive asli dan diperiksa SHA-256-nya. Hasil tersimpan di MyDrive/rme_asr_v3. Notebook mandiri; tidak perlu mengunggah modul Python terpisah.

15 tes CPU telah lolos. Training/inference GPU belum dijalankan oleh penyusun; kesiapan runtime penuh dan akurasi model baru harus dibuktikan melalui run Colab. Selesai training tidak otomatis berarti layak enterprise. Baca README.md dan LAPORAN_AUDIT.md.
