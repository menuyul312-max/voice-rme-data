# Kontrak data RME v5

Semua file opsional berada di `MyDrive/rme_asr_v3/v5/`. Notebook dapat berjalan dengan arsip awal tanpa file tambahan; hasilnya tetap eksperimen silver. Data klinis dan audio disimpan di Drive pengguna, tidak diunggah ke GitHub oleh notebook.

## 1. Koreksi referensi yang sudah ada

CSV `train_reviews.csv` memiliki kolom:

```csv
id,audio_sha256,reviewer_1,reviewer_2,adjudicated,verified_verbatim
train/contoh.wav,HASH_AUDIO_ASLI,dokter_pseudonim_1,dokter_pseudonim_2,true,"tidak ada alergi obat"
```

`id` dan hash harus sama dengan `split_plan.json`. Dua reviewer harus berbeda. `adjudicated` harus `true`. Nilai transkrip merupakan ucapan verbatim yang didengar, termasuk koreksi diri; bukan kesimpulan klinis. Kosong tidak diizinkan di jalur ini karena audio nonspeech perlu verifikasi berbeda.

Koreksi hanya berlaku pada train; ID test/dev ditolak. CSV dengan ID sama dua kali, hash berubah atau reviewer belum lengkap membuat notebook berhenti. Untuk dev selection/confirmation, file `selection_dev_reviews.csv`/`confirmation_dev_reviews.csv` memakai schema sama dan hanya ID pada subset masing-masing. Ambil ID dari refinement plan. Jangan mencampur ID seluruh dev ke salah satu subset. Untuk benchmark, `test_gold_reviews.csv` berisi seluruh ID test agar skor full-gold dapat dihitung; parsial tetap dicatat tetapi tidak diberi label evaluasi gold seluruh test.

Karantina arsip lama di luar locked split tidak otomatis dipulihkan melalui review ini. Memulihkan data yang semula tidak masuk plan membutuhkan rencana dataset baru serta audit grup/test agar tidak mengubah evaluasi diam-diam. v5 dapat menerima data baru melalui jalur berikut dengan pemeriksaan heldout.

## 2. Data training klinis tambahan

`additional_train_audited.json` adalah list record yang sudah diberi ID dan grup. Contoh struktur (semua nilai contoh; hash harus dihitung dari file nyata):

```json
[{
  "id":"extra/siteA_session007_clip001",
  "audio_file":"extra_training/siteA_session007_clip001.wav",
  "ref":"pasien tidak memiliki riwayat alergi obat",
  "split":"train",
  "source":"clinical_dictation",
  "source_detail":"authorized_siteA_dictation",
  "group_id":"session007",
  "cluster_id":"speaker03_session007",
  "speaker_id":"speaker03",
  "session_id":"session007",
  "patient_id":"patient_pseudonym_optional",
  "video_id":"",
  "synthetic":false,
  "reference_verified":true,
  "reviewer_1":"reviewerA",
  "reviewer_2":"reviewerB",
  "adjudicated":true,
  "commercial_use_reviewed":true,
  "audio_sha256":"SHA256_FILE_BYTES",
  "pcm_sha256":"SHA256_DECODED_FLOAT32_ALWAYS_2D_BYTES",
  "sr":16000,
  "channels":1,
  "seconds":8.5,
  "declared_seconds":8.5
}]
```

Letakkan audio relatif terhadap `ROOT=/content/gold_rme_v3`, misalnya `ROOT/extra_training/...`; file di Drive dapat disalin ke sana sebelum sel refinement. Path absolut dan `..` tidak diterima. Contoh path/id harus diganti, tidak ada file sampel audio dalam paket.

Setiap record harus train-only, reference_verified, dua reviewer dan adjudikasi. `commercial_use_reviewed` berarti pengguna telah meninjau hak pemakaian; bukan kesimpulan otomatis notebook. Record ditolak jika ID, speaker, session, patient, group, PCM atau teks panjang identik bertabrakan dengan heldout. SHA256 file dan PCM diverifikasi lagi melalui decoding. Gunakan kode `inspect_audio` di notebook untuk menghasilkan statistik/hash; jangan menebak atau memakai contoh literal.

Selain disjoint terhadap heldout, duplikat PCM di refined train ditolak. Source group IDs harus bermakna dan stabil; ID baru per-klip tidak membuktikan speaker independen. Tambahan korpus yang sama dokter/pasiennya tetapi pseudonim berubah tidak dapat dideteksi notebook; pemilik dataset wajib menjaga pemetaan identitas dan provenance. MinHash hanya kandidat duplikasi teks; near-duplicate audio, speaker matching atau overlap parsial masih memerlukan audit tambahan.

## 3. Real nonspeech training opsional

`verified_nonspeech_train.json` memakai struktur record tambahan, dengan:

- `ref` string kosong;
- `human_verified_nonspeech: true`;
- dua reviewer berbeda yang benar-benar memastikan tidak ada ucapan;
- hak pemakaian ditinjau, speaker/session/grup independen dari heldout;
- audio lingkungan nyata (room tone, kipas, alat, noise) dengan durasi0,2–30 detik;
- statistik audio/hash dihitung oleh notebook; `pcm_sha256` input contoh tidak dipercaya.

Notebook memeriksa audio dan batas jumlah sekitar2% data positif. Ini eksperimen negative examples, bukan reproduksi Calm-Whisper head-selection. Jangan mengubah audio speech yang gagal VAD menjadi label kosong. Sertakan hard positives (ucapan pelan, kata singkat, repetisi sah) dalam cohort uji untuk memeriksa apakah kemampuan mengucapkan kata sah ikut ditekan.

## 4. Cohort challenge independen

`challenge_manifest.json`, audio di `v5/challenge_audio/`:

```json
[
 {"id":"room001","audio_file":"room001.wav","kind":"noise","ref":""},
 {"id":"dict001","audio_file":"dict001.wav","kind":"clinical","ref":"tidak ada riwayat alergi obat"},
 {"id":"long001","audio_file":"long001.wav","kind":"long_form","ref":"TRANSKRIP_VERBATIM_SESI_PANJANG"}
]
```

`kind`: `silence`, `noise`, `clinical`, atau `long_form`. Noise/silence tidak boleh mengandung ucapan dan referensinya kosong. Long-form harus >30 detik. Input ini dievaluasi offline dengan overlap-chunking30s/5s; belum klaim streaming atau diarization. Pisahkan dokter/session/site/pasien dari seluruh training dan dev. Simpan metadata cohort/protokol di luar contoh minimal ini; notebook tidak dapat membuktikan independensi hanya dari nama file.

## 5. Review klinis benchmark

Notebook mengekspor `clinical_error_review.csv`. Sesudah mendengar audio dan adjudikasi, simpan versi terisi sebagai `clinical_error_review_filled.csv` di folder benchmark yang sama. Semua planned IDs harus ada tepat sekali. Isi reviewer berbeda, `adjudicated=true`, `critical_errors` dan `critical_opportunities` integer dengan 0<=errors<=opportunities.

Definisikan opportunity dan severity sebelum melihat hasil: misalnya per fakta obat/dosis, alergi, negasi, angka vital, subject/temporality. Jangan mengganti denominator antar model untuk mempercantik hasil. Kolom `error_types` mendokumentasikan kategori; ahli klinis menentukan signifikansi. Statistik yang dihasilkan adalah rate descriptif, bukan bukti risiko nol atau sertifikasi.

`blind_reference_review.csv` tidak memperlihatkan hipotesis kepada penyusun gold. Lembar klinis memperlihatkan hipotesis sesudah referensi dikunci. Ini mengurangi anchoring review, tetapi kualitas proses tetap perlu diaudit.
