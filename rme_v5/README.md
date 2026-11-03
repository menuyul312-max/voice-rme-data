# RME ASR v5 — petunjuk operasional

Buka **[training v5 di Colab](https://colab.research.google.com/github/menuyul312-max/voice-rme-data/blob/main/rme_v5/train_whisper_turbo_rme_v5.ipynb)**, pilih runtime GPU baru (A100 disarankan), Run all dan izinkan Drive. Tidak perlu mengunggah ZIP/modul. Bila dependensi meminta restart, restart lalu Run all lagi.

V5 adalah paket riset dan eksperimen lengkap, bukan bobot baru dengan akurasi terjamin. Ia memakai arsip Drive asli yang SHA256-nya diaudit, semua audit/karantina v3, screening VAD, review labels, teacher audit, dua profil LoRA, ekspor bobot dan pemeriksaan kandidat. Dua profil ditraining terpisah, masing-masing maksimum3 epoch dengan early stopping; waktu/biaya lebih besar dari satu run v4. Guru large-v3 memerlukan download bobot tambahan dan GPU RAM untuk inference, dimuat sebelum training dan dibersihkan sesudahnya.

## Urutan yang dianjurkan

1. Opsional: jalankan [refinement-only](https://colab.research.google.com/github/menuyul312-max/voice-rme-data/blob/main/rme_v5/refine_rme_data_v5.ipynb) untuk menghasilkan daftar prioritas review sebelum membayar training penuh. Teacher default mengaudit256 klip train; tidak mengklaim seluruh dataset sudah dimurnikan secara manual.
2. Dengarkan prioritas review, isi `train_review_template.csv` melalui dua reviewer, simpan sebagai `MyDrive/rme_asr_v3/v5/train_reviews.csv`. Kolom referensi jangan diisi dari keluaran teacher. Langkah ini opsional untuk eksperimen awal; tanpa review, label sumber tetap silver dan hasil tidak dinyatakan gold klinis.
3. Jalankan training v5. Folder `MyDrive/rme_asr_v3` tetap dipakai agar rencana split v3/v4 terkunci. Run/checkpoint v5 berada di `v5/runs/<fingerprint>/`; optimizer v4 tidak dipakai.
4. Dua profil default: `decoder_qv`, `top8_qv`. Profil `all_attention` dapat dimasukkan di `PROFILE_NAMES` sebagai ablation tambahan dengan biaya lebih besar. Untuk runtime terbatas gunakan satu profil; itu tidak mengukur perbandingan antarprofil.
5. Model dipilih pada selection-dev lalu diperiksa pada grup confirmation-dev yang terpisah. Tersimpan `latest_v5_candidate.json` dan, hanya jika lolos, `latest_v5_accepted.json`. Status FAIL berarti pertahankan v4 sebagai kandidat sebelumnya, bukan memaksa menambah epoch. Kandidat gagal tetap dapat dibenchmark untuk diagnosis.
6. Jalankan [benchmark v5](https://colab.research.google.com/github/menuyul312-max/voice-rme-data/blob/main/rme_v5/benchmark_rme_v5_colab.ipynb) dengan akun Drive yang sama. Base, v4 bila tersedia, dan v5 memakai aturan decoding sama. Kolom model dinamai `v5` secara benar.

## Konfigurasi dan biaya

- Base default Turbo, commit dikunci dari eksperimen awal. `BASE_MODEL_ID` dapat diubah ke `openai/whisper-large-v3` untuk eksperimen kualitas penuh; commit baru dikunci di file revision. Full large-v3 perlu microbatch lebih kecil dan waktu inference/training lebih besar. Belum ada hasil yang membuktikan ia lebih baik pada data ini.
- Semua bobot pretrained frozen; adapter LoRA yang dilatih. Persen trainable dicatat aktual per profil, bukan diasumsikan dari jumlah layer.
- `TEACHER_AUDIT_N=256`; set0 bila hanya ingin eksperimen training tanpa teacher review. Ini tidak mengubah label secara otomatis.
- `MAX_RECORDING_CLIPS=300`, synthetic<=5%, optional verified nonspeech sekitar<=2%. Semua keputusan merupakan konfigurasi proyek yang perlu ablation.
- Data/config/code/package/review fingerprints membentuk run baru. Run all melanjutkan checkpoint lengkap yang sesuai, kehilangan progres sejak save terakhir tetap mungkin terjadi. Mengubah daftar review atau profil dapat membentuk run baru.
- Notebook menyimpan checkpoint langsung ke Drive untuk ketahanan sesi; I/O Drive bisa menjadi bottleneck. Untuk pelatihan besar gunakan penyimpanan lokal cepat + sinkronisasi atomik pada job terkelola, bukan menganggap Colab layanan enterprise.

## Hasil utama

Di `v5/`: `refinement_plan.json`, `training_screen.json`, `near_duplicate_candidates.json`, `teacher_review_priority.csv`, `candidate_comparison.json`, `selected_before_confirmation.json`, `latest_v5_candidate.json`.

Di setiap `v5/runs/<id>/`: `run_spec.json`, `parameter_report.json`, `baseline_dev.json`, `training_result.json`, `adapter/`, `merged/`, `candidate.json`.

Di `v5/benchmarks/<id>/`: `benchmark_report.json`, cache prediksi, lembar review buta dan klinis; challenge metrics bila cohort tambahan disediakan. Latency batch1 sekarang mencakup VAD audit. GPU/Torch/CUDA disimpan.

## Batas hasil

Test lama telah diperiksa dalam iterasi pengembangan; ia berfungsi sebagai regresi internal. Confirmation-dev dipisahkan untuk v5, tetapi berasal dari korpus yang telah terlibat eksperimen lama. Tidak diklaim fresh external holdout.

V5 tidak otomatis memperbaiki isi audio/label yang belum didengar klinisi, mengunduh big data baru, memberi diagnosis, membangun aplikasi SIMRS, mengimplementasikan diarization atau memenuhi sertifikasi medis. Kajian dan arsitektur produk ada di `DEEP_RESEARCH_V5.md`; skema data tambahan/review di `DATA_CONTRACT.md`.

Tes logika dan actual tiny Whisper/PEFT CPU disediakan. Training Turbo penuh, inference GPU, kualitas klinis dan SLA produk v5 belum dijalankan oleh penyusun. Model masih berstatus `NOT_READY` untuk klaim enterprise sampai validasi eksternal klinis dan operasional tersedia.
