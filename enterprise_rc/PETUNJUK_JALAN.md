# Jalankan benchmark dahulu, training ulang opsional

Perbaikan kode sudah tertanam di sel `%%writefile` dalam kedua notebook RC. Tidak perlu menyalin modul satu per satu saat memakai notebook.

## Jalur pertama: benchmark dengan model lama

1. Buka `notebooks/benchmark_v2_colab_t4_rc.ipynb` di Colab T4.
2. Di langkah 3, biarkan `MODEL_VARIANT = "existing_v2"` (default).
3. Drive harus memuat model lama `whisper_large_v3_turbo_rme_v2_merged` atau arsip `.tar.gz`-nya, serta `rme_v2_split.json` dari training asli. Mount akun Drive yang menyimpan hasil training Anda.
4. Run all. Notebook menulis modul perbaikan, menjalankan tes CPU, lalu mengulang inferensi/evaluasi. Hasil masuk `bench_v2_rc_existing_v2_out`, terpisah dari benchmark lama.
5. LLM/SOAP dijalankan dengan filter baru. Vital yang ditolak tetap dicatat sebagai kegagalan model mentah dalam G6. Lengkapi verifikasi dokter untuk G7.

Jalur ini menguji perubahan kode dengan bobot tetap. Tidak menjanjikan WER/recall ASR meningkat: decoding dan bobot ASR pada dasarnya masih sama. CI berpasangan bisa berubah karena bootstrap per-video. Jangan menilai keberhasilan hanya dari jumlah vital yang lolos filter.

## Jalur kedua: setelah memutuskan eksperimen training baru

1. Jalankan `notebooks/train_whisper_turbo_rme_v2_rc.ipynb` di A100/L4/T4.
2. Training RC memakai nama checkpoint dan model terpisah: `v2_rc_checkpoints_*` dan `whisper_large_v3_turbo_rme_v2_rc_merged`.
3. Setelah selesai, buka benchmark RC dan ubah `MODEL_VARIANT = "retrained_rc"` di langkah 3.
4. Hasil masuk `bench_v2_rc_retrained_rc_out`. Jangan mencampur hasil kedua varian.

Training RC memperbaiki RNG dan validasi input. LR, LoRA, dan jumlah epoch belum dioptimasi ulang; training ulang recipe yang hampir sama belum tentu menyelesaikan G2/G3/G4. Kurasi label dan eksperimen berbasis dev diperlukan; test tetap dibekukan.

## Lokasi perbaikan utama

| File | Perubahan |
|---|---|
| `src/clinical_safety.py` | Bukti kontekstual vital dan validasi bentuk payload |
| `src/live_pipeline_t4.py` | Grounding, regimen tanpa default, preview FHIR, blokir fallback klinis |
| `src/local_med_llm.py` | Default vital kosong; hapus inferensi vital dari angka tidak terkait |
| `src/benchmark_real.py` | Evaluasi vital dan bootstrap berpasangan per-video |
| `src/release_policy.py` | Keputusan gerbang rilis yang gagal tertutup |
| `src/rme_asr_v2.py` | RNG worker dan larangan pemotongan label diam-diam |
| `notebooks/*_rc.ipynb` | Modul sama tertanam, G6/G7 diperbaiki, varian model/cache eksplisit |

37 tes CPU lulus pada patch sebelumnya; eksekusi GPU belum dilakukan. Baca `AUDIT_RELEASE.md` untuk hasil lama dan batas kelayakan enterprise. Ini bukan klaim siap production.
