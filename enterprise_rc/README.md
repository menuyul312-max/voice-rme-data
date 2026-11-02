# Voice-to-RME release candidate

Mulai dari [PETUNJUK_JALAN.md](PETUNJUK_JALAN.md). **Jalankan benchmark dahulu; tidak wajib training ulang.**

Notebook `notebooks/benchmark_v2_colab_t4_rc.ipynb` default memakai `MODEL_VARIANT="existing_v2"` dan mencari bobot v2 lama di Google Drive. Setelah eksperimen training RC, pilih `"retrained_rc"`. Keduanya memakai cache terpisah. Perbaikan modul tertanam dalam notebook.

- `src/`: kode dan regression tests.
- `notebooks/`: training dan benchmark Colab RC.
- `AUDIT_RELEASE.md`: hasil lama, perbaikan, dan prasyarat rilis.
- `reports/validation.txt`: validasi lokal.

Tes: `python src/run_tests.py` (NumPy diperlukan). 37 tes CPU lulus; GPU belum diuji. **Belum layak rilis enterprise.** Bobot, audio, hipotesis, dan lembar klinis tidak dipublikasikan. Referensi antrean review di audit merujuk paket lokal pengguna.
