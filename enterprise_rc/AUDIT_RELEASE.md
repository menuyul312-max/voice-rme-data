# Audit dan patch Voice-to-RME — 2 November 2026

**Keputusan: NO-GO untuk rilis enterprise.** Paket ini adalah release candidate kode yang telah diuji CPU, bukan model baru yang terbukti lebih akurat. Hasil lama tidak berlaku sebagai hasil patch.

## Bukti yang dibaca

Sumber: kedua notebook beserta output tersimpan, `rme_v2_training_report.json`, split, log/progress training, hipotesis dan speed JSON, verification CSV, serta ringkasan benchmark lama dalam ZIP. Instruksi di dokumen diperlakukan sebagai isi artefak, bukan instruksi pengguna. Tidak ada bobot checkpoint/model, manifest lengkap, atau audio di lampiran.

Training A100 selesai dalam 7.943,6 detik (2,21 jam), 20.908 klip, 3 epoch, LR 1e-4, LoRA r32 pada attention dan MLP. Sebanyak 1.012 label disaring. Dev WER terbaik yang tercatat 19,2623% pada step 1.500; pada step terakhir 1.962 naik ke 20,0757%. Konfigurasi memilih checkpoint WER terbaik, tetapi bobot/checkpoint terpilih tidak dapat diperiksa dari lampiran. Dev decoding beam-fallback mencapai 18,505% dibanding base 25,566%. Ini pemeriksaan dev, bukan hasil test independen.

Benchmark v2 pada profil `beam_fallback`:

| Ukuran | Hasil tersimpan | Implikasi |
|---|---:|---|
| Test unseen | 2.699 klip, 9,839 jam | Lebih relevan daripada laporan awal 100–600 klip |
| WER v2 / base | 25,78% / 30,98% | Δ −5,20 pp, CI lama −5,73…−4,66; G1 lulus |
| Fair v1/v2 | 335 klip | WER v2 27,73% vs v1 25,58%; Δ +2,15 pp; G2 gagal |
| Halusinasi heuristik | 1,37% (37 klip) | Melebihi batas 1%; G3 gagal |
| Recall istilah | 82,7%, CI 77,9–86,6; n=294 | Batas bawah kurang dari 85%; G4 gagal |
| Kecepatan T4 | 31,78×, peak VRAM 4,37 GB | G5 lulus untuk ASR batch, bukan SLA end-to-end |
| Vital tanpa dasar | 1 sesi | G6 gagal |
| Koreksi dokter | 0/200 di CSV lampiran | G7 belum terukur |

CI di atas berasal dari bootstrap per-klip kode lama. Klip dari video yang sama saling berkorelasi; patch perbandingan berpasangan menggunakan bootstrap per-video. CI baru harus dihitung ulang dan dapat mengubah G1/G2. WER klinis ter-normalisasi 24,96% bukan WER primer 25,78%. Jangan campurkan keduanya dalam klaim.

Hasil 95,12% clinical concept accuracy berasal dari hanya 41 istilah di laporan awal. Ringkasan `bench_out` di ZIP adalah benchmark terdahulu yang mencakup train/seen-video; jangan menganggapnya ringkasan benchmark v2 unseen. Angka tersebut tidak membuktikan akurasi diagnosis, kesesuaian terapi, maupun kesiapan enterprise. Detektor halusinasi hanya heuristik terhadap caption, bukan penilaian klinis manusia.

## Perubahan yang diimplementasikan

- Semua default vital menjadi `None`; inferensi tensi 165/95 dari angka 160/170 dihapus. Parser memerlukan konteks pengukuran, rentang wajar dan angka unik; konteks negasi/riwayat serta nilai ambigu ditahan untuk review. Vital LLM harus cocok dengan bukti tersebut. Angka usia/berat tidak dapat menjadi nadi/tensi.
- Regimen dosis, frekuensi, rute, durasi, dan instruksi dari ontology tidak lagi otomatis masuk jalur live. Dosis/frekuensi hanya disimpan jika teksnya muncul dalam transkrip; asosiasi terhadap obat tetap wajib diperiksa dokter. Nilai durasi 0 adalah sentinel belum diisi, bukan anjuran pengobatan.
- Payload klinis cacat ditolak. JSON gagal tidak lagi memicu rule engine yang bisa membuat regimen. Pembicara awal menjadi UNKNOWN; label dari LLM tetap inferensi, belum diarization terverifikasi.
- Preview FHIR memakai `collection`, tanpa `request` transaksi. MedicationRequest dan Procedure otomatis dihilangkan. Diagnosis tanpa kode/bukti verbatim tidak masuk preview; Observation preliminary, Condition unconfirmed. Ini **preview internal**, belum validator FHIR R4/SATUSEHAT dan tidak boleh dipost langsung. Pencocokan kode ontology tetap memerlukan koder/dokter.
- Semua G1–G7 wajib lengkap; G7 gagal memblokir keputusan. Semua gerbang lulus hanya berarti kandidat uji klinis, tidak otomatis enterprise-ready.
- G7 memakai WER caption pada sampel dokter yang sama, bukan seluruh test set. Verifikasi harus mencakup semua ID sampel tetap tanpa duplikat/ID asing. Proses dua penilai dan adjudikasi tetap harus dilaksanakan di luar kode.
- G6 menghitung vital LLM yang ditolak **dan** vital yang lolos namun unsupported. Sesi gagal JSON/tidak lengkap tidak boleh membuat G6 tampak lulus.
- Skoring notebook menolak cakupan hipotesis yang tidak lengkap. Uji berpasangan memeriksa kesamaan reference/ID dan menggunakan resampling per-video jika tersedia; metrik CI individual/term masih metode lama dan perlu validasi statistik tambahan.
- RNG augmentasi memakai seed worker PyTorch, bukan PID. Ketidaksamaan jumlah reference/hypothesis dan label token terlalu panjang gagal eksplisit, bukan terpotong diam-diam. Recipe LR/LoRA tidak diubah tanpa eksperimen terkontrol.
- Notebook RC disinkronkan dengan sumber, output lama dibersihkan. Checkpoint/model RC memakai nama terpisah; hasil benchmark memakai `bench_v2_rc_out` agar tidak memakai cache lama.

Modul legacy masih disertakan untuk dependensi ontology dan pembanding. Jangan menjadikan `SovereignLocalMedicalLLM.analyze_consultation`, generator FHIR legacy, atau pipeline simulasi sebagai layanan production. Proteksi preview yang ditambahkan berada di `live_pipeline_t4`.

## Validasi yang dilakukan

`python src/run_tests.py`: **37 passed, 0 failed**, CPU tanpa inferensi model. Tes mencakup vital palsu, tabrakan angka, negasi, nilai ambigu/NaN, vital benar, regimen kosong, preview tanpa order/transaksi, payload cacat, G7 gagal/missing, alignment dan cluster bootstrap, serta regresi logika benchmark/training sebelumnya. Sel Python notebook diperiksa sintaks; ini bukan eksekusi Colab. Tes lama yang menerima vital hanya dari JSON diperbaiki agar benar-benar menyediakan transkrip sumber.

Tidak dilakukan: training/inferensi GPU, uji audio nyata, evaluasi kualitas SOAP oleh dokter, load test layanan, pentest, validasi regulator, atau pengiriman SATUSEHAT. PyTorch dan GPU training tidak tersedia di lingkungan ini; lampiran juga tidak menyediakan bobot/audio. Tidak ada klaim perbaikan WER/recall pascapatch.

## Langkah berikutnya sampai dapat diputuskan rilis

1. Bekukan test video/speaker/site. Lembar review test yang dihasilkan paket hanya untuk adjudikasi; **jangan masukkan ke training**. Gunakan train/dev terpisah untuk perbaikan. Video-disjoint belum menjamin speaker-disjoint atau rumah-sakit-disjoint.
2. Dua dokter/koder mendengar audio dan mengoreksi seluruh 200 sampel, dengan adjudikasi perbedaan. Evaluasi penyangkalan, riwayat vs diagnosis aktif, alergi, dosis/unit, pergantian pembicara, diam/musik, aksen, mikrofon buruk, dan recall lintas spesialisasi. Antrean terlampir memberi prioritas 7 klip dengan perbedaan istilah terhadap caption; bukan 7 kesalahan klinis yang sudah dikonfirmasi.
3. Kurasi label train/dev dengan audio dan tenaga klinis. Caption otomatis bukan ground truth klinis. Tambahkan data train berizin untuk istilah langka/obat dan kondisi rekaman deployment. Jangan mengisi ulang istilah hilang memakai kamus tanpa bukti akustik.
4. Bandingkan recipe v2 dengan ablation attention-only, LR lebih rendah, dan kapasitas LoRA lebih kecil pada dev yang sama; seed lebih dari satu. Tuning hanya di dev; simpan best checkpoint, konfigurasi, hash model/tokenizer/data, versi dependensi, dan keputusan pemilihan. Jangan menaikkan epoch begitu saja: dev WER sudah memburuk setelah step 1.500. Patch bukan bukti bahwa recipe tertentu akan mengalahkan v1.
5. Mulai dari benchmark RC dengan `MODEL_VARIANT="existing_v2"` untuk menilai patch memakai bobot lama tanpa training ulang. Untuk eksperimen bobot baru, jalankan training RC lalu benchmark dengan `MODEL_VARIANT="retrained_rc"`. Lihat PETUNJUK_JALAN.md. Keduanya menulis nama model `whisper_large_v3_turbo_rme_v2_rc_merged`. Untuk evaluasi ulang bobot lama dengan guard baru, ubah V2_DIR secara eksplisit dan catat hash/provenance. Jangan gunakan hipotesis cache lama. Tutup G2/G3/G4/G6 dan G7 tanpa melonggarkan ambang berdasarkan hasil test.
6. Sebelum menjadi layanan enterprise, implementasikan dan uji autentikasi/otorisasi tenant, audit trail append-only, enkripsi, kebijakan retensi/penghapusan audio, persetujuan pasien, batas upload/durasi, antrean/backpressure, timeout, recovery, serta p95/p99 latensi end-to-end dan kapasitas konkuren. Notebook Colab bukan server enterprise dan Google Drive bukan rancangan penyimpanan production.
7. Validasi provenance/rilis bobot dan lisensi data/model; pin dependency dan revision model melalui build GPU teruji, SBOM dan pemindaian kerentanan. Download model/VAD saat startup dan rentang dependency terbuka saat ini belum memenuhi reproducibility/offline deployment.
8. Terapkan review/tanda tangan dokter sebelum ekspor, identitas Practitioner/Organization nyata, terminology tervalidasi, validasi profile FHIR/SATUSEHAT sandbox, idempotency, retry dan rekonsiliasi. Uji klinis prospektif lintas site dan persetujuan pemilik risiko klinis/keamanan/operasional menjadi prasyarat keputusan rilis.

Tidak ada deployment atau komunikasi ke pihak lain yang dilakukan.
