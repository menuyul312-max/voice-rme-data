# RME v4: metode dan batas pembuktian

V4 memperbaiki temuan benchmark pengguna (2094 klip): WER 34,43% base vs 37,69% v3, dengan beberapa output repetitif hingga 444 kata. Berkas tambahan terakhir hanya berisi benchmark; `run_spec.json` dan `training_result.json` belum tersedia. Karena itu penyebab training sebelumnya belum dapat dipastikan. V4 adalah resep kandidat berbasis bukti, bukan hasil akurasi yang telah terukur.

## Cara menjalankan

[Buka training v4](https://colab.research.google.com/github/menuyul312-max/voice-rme-data/blob/main/rme_v4/train_whisper_turbo_rme_v4.ipynb). Gunakan runtime GPU baru, A100 disarankan. Run all dan izinkan Drive. Notebook mandiri; seluruh modul, tes, audit, karantina, VAD, training, ekspor dan evaluasi dev sudah tertanam. Tidak memerlukan ZIP.

Dataset dan rencana split memakai folder `MyDrive/rme_asr_v3` yang sudah ada supaya split tidak berubah. V4 membuat run dengan fingerprint kode/config/train IDs baru; tidak melanjutkan checkpoint v3. Screening train baru tersimpan di `v4_training_screen.json`. Data berlabel dengan VAD tanpa speech atau label sangat repetitif dikarantina untuk review, tidak dilabel ulang otomatis. Sintetis dibatasi paling banyak 10% train. VAD bekerja pada CPU/ONNX di 16 kHz, threshold 0,35, min speech 100 ms, speech padding 300 ms. Nilai tersebut pilihan konservatif proyek, bukan standar klinis; cek false-negative pada ucapan pendek.

Resep awal: LoRA rank16/alpha32 pada q/v attention decoder saja; encoder tetap pretrained. LR 1e-5, batch efektif32, maksimal3 epoch, evaluasi kira-kira4 kali per epoch (maks interval125 step), patience4, pilih WER dev terendah. Pilihan encoder beku dan LR lebih kecil merupakan hipotesis mengurangi pergeseran kemampuan pretrained, bukan metode terbaik yang sudah dibuktikan. Dataset label yang belum gold tetap membatasi hasil.

Sebelum training, evaluasi LoRA zero-initialized (ekuivalen base pada forward awal) pada dev yang sama. Checkpoint dipilih memakai beam3 dengan batas ngram6. Setelah ekspor, jalankan base dan kandidat dengan guard dan fallback identik pada seluruh subset dev pemilihan (hingga600 klip). Gerbang mensyaratkan WER turun, CER tidak naik, jumlah flag angka/unit/negasi tidak naik, dan tidak ada kegagalan decoding yang belum tertangani. Ini screening internal berbasis referensi yang tersedia, bukan sertifikasi klinis atau uji signifikansi; CI per grup turut disimpan.

Kandidat selalu dicatat di `latest_v4_candidate.json`. Hanya kandidat yang lolos yang memperbarui `latest_v4_accepted.json`; lihat `dev_acceptance.json`. Jika FAIL, jangan menjadikan kandidat model produksi atau sekadar menambah epoch. Periksa contoh error dan gold labels, lalu lakukan ablation dev terencana (misalnya LR 5e-6 vs 1e-5, decoder-only vs attention luas). Nama run baru menjaga optimizer checkpoint tidak tercampur.

[Buka benchmark v4](https://colab.research.google.com/github/menuyul312-max/voice-rme-data/blob/main/rme_v4/benchmark_rme_v4_colab_t4.ipynb). Hasil disimpan terpisah di `benchmarks_v4`. Benchmark tetap bisa mengukur kandidat yang gagal screening untuk diagnosis. Test v3 telah digunakan menelusuri bug: hasilnya sekarang regresi internal; cohort eksternal baru diperlukan sebelum klaim kualitas final.

## Referensi dan penerapan

1. [OpenAI Whisper transcribe.py](https://github.com/openai/whisper/blob/main/whisper/transcribe.py): memakai compression ratio, average log probability, no-speech dan fallback; menjelaskan pengaruh conditioning antarwindow pada loop. V4 mengadaptasi pemeriksaan compression ratio >2,4 dan deteksi repetisi, lalu retry deterministik. Implementasi ini tidak mengklaim identik dengan fallback temperature resmi. Tiap klip independen tanpa prompt transkrip sebelumnya; tidak memasukkan `condition_on_previous_text` ke API HF short-form yang berbeda.
2. [Hugging Face: Fine-Tune Whisper](https://huggingface.co/blog/fine-tune-whisper): resampling16kHz, feature extraction, masking padding/BOS, generative WER evaluation dan checkpoint selection. V4 mengikuti kontrak tersebut; LR/LoRA disesuaikan sebagai eksperimen domain, tidak diambil sebagai jaminan dari tutorial model/dataset lain.
3. [PEFT resmi](https://github.com/huggingface/peft): LoRA untuk adaptasi dengan parameter terbatas; contoh Whisper tersedia. Tidak ada rekomendasi universal bahwa decoder-only pasti paling akurat.
4. [Silero VAD](https://github.com/snakers4/silero-vad): pemisahan speech/non-speech dan dukungan8/16kHz. V4 memakai paket6.2.0/ONNX, audio I/O soundfile. Pada inference VAD hanya audit; tidak menghapus kata pendek/berbisik atau otomatis menjadikan hasil kosong. Segmentasi yang membuang silence dari audio berlabel tanpa memperbaiki alignment bisa merusak training, sehingga tidak dilakukan.
5. [faster-whisper](https://github.com/SYSTRAN/faster-whisper): integrasi VAD dan praktik pemrosesan silence. V4 tetap HF agar perbandingan engine konsisten; konversi CTranslate2 bukan bagian kualitas yang sudah dibuktikan.
6. [Accent related errors in clinical speech transcription](https://pubmed.ncbi.nlm.nih.gov/41772044/): penelitian klinis menegaskan perlunya pengujian per kelompok speaker/aksen. Hasil English tidak boleh dianggap mewakili Bahasa Indonesia, dialek RS, atau mikrofon pengguna.

## Pengamanan decoding

Beam3/no_repeat_ngram6 dipakai simetris. Loop minimal8 pengulangan frasa (1–6 kata), compression ratio tinggi dan batas panjang memicu retry beam5/ngram3/repetition penalty1,05. Pilih retry hanya bila pemeriksaan reference-blind lolos. Simpan raw primary, retry, flag, dan output akhir. Jika retry tetap gagal, simpan output dengan `needs_review`; tidak menghilangkan klip dari benchmark atau merapikan kata untuk mempercantik skor. ngram blocking juga dapat menekan pengulangan klinis yang sah, sehingga perlu evaluasi dev dan gold.

Detektor cap menghitung posisi EOS pertama: EOS yang muncul tepat pada batas panjang tetap ditandai, termasuk kasus forced-EOS. Padding setelah EOS dini tidak dianggap cap. Batas panjang model tetap448 untuk menghindari pemotongan target berdasarkan perkiraan jumlah kata.

## Pengujian yang tersedia

22 tes CPU mencakup data contracts dan regresi loop/cap. Struktur/sintaks notebook dan kesamaan modul diperiksa. Training/inference GPU penuh, manfaat WER v4, VAD false-negative dan SLA belum terukur di lingkungan penyusun. Notebook tidak memuat output hasil lama. Label verbatim yang sudah didengar dua reviewer, cohort RS/klinik independen, pengujian silence/noise dan long-form tetap dibutuhkan. Tidak menambahkan koreksi obat/dosis berbasis LLM.
