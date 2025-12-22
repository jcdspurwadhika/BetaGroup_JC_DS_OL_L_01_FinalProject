**Final Project Telco Customer Churn**

Dalam github link ini terdapat 3 file yang perlu di download agar model dapat berjalan dengan baik.

Final Project Telco Customer Churn.ipynb (Jupyter Notebook File)
churn_model.pkl (Pickle File)
streamlit_app (Streamlit File)

**Notebook Contents**
1. Business Process Understanding
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Machine Learning (Methodology: Modeling/Analysis)
6. Conclusion and Recommendation

**Context**
Sebuah perusahaan telekomunikasi (telco) yang berlokasi di Amerika Serikat ingin melakukan analisa terhadap pelanggan (customer subscriber) perusahaannya dan ingin menganalisa ciri khas customer yang berhenti berlangganan (churn). Perusahaan telco tersebut menawarkan berbagai jasa antara lain: PhoneService (saluran telpon rumah) and InternetServices (layanan internet) yang akan di jelaskan lebih lanjut di bawah ini. 

Kami sebagai Data Consultant telah dikontrak oleh perusahaan telco tersebut untuk menganalisa data-data yang terkumpul sehabis internal audit mereka. Internal audit data tersebut didapat dari kuartal 3 (Q3) tahun 2019 dan seluruh datanya adalah berdasarkan data internal perusahaan telco ini saja, bukan gabungan dari perusahaan-perusahaan lain.
(ref: https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

Sangat diharapkan data-data yang terkumpul sehabis internal audit, dan digunakannya machine learning (ML) diharapkan dapat membuahkan insights kepada demographic/ciri khas seperti apa customer baru yang akan di target pada marketing campaign selanjutnya. Dengan adanya insights target demographic/ciri khas diharapkan biaya marketing di masa mendatang dapat lebih tepat sasaran, lebih terfokus dan lebih ditekan.

Target
- 0: Not Churn (Continue Service)
- 1: Churn (Cancel Service)

**Problem Statement**
Biaya untuk menarik pelanggan baru adalah suatu resources yang terbatas, tentu perusahaan telco ini ingin mengurangi pelanggan yang putus berlangganan (churn). Berdasarkan data internal audit churn rate pada kuartal 3 tahun 2019 adalah 26%. Mengingat biaya agency atau freelancer mulai dari $50-150 per jam hingga biaya campaign bulanan $300-$2000 semua berdampak pada berapa banyak biaya yang dikeluarkan hanya untuk mendapatkan satu pelanggan baru. Maka churn rate sangat diharapkan dapat turun ke bawah 10%. Dampak turunannya ini akan sangat meningkatkan efisiensi biaya agency atau freelancer yang bisa mencapai puluhan ribu hingga ratusan ribu dollar. 

Diharapkan dengan studi kasus dengan machine learning, dapat ditemukannya insights pada alasan-alasan pelanggan yang putus berlangganan. 

Pertanyaan-pertanyaan dibawah ini akan dicoba untuk menjawab studi kasus dengan Machine Learning:
- Faktor-faktor demographic eg. Gender (Jenis Kelamin), SeniorCitizen (Lansia), Partner (Pasangan), Dependents (Tanggungan) yang mempunyai dampak tinggi terhadapat churn rate. 
- Faktor-faktor lain seperti tenure (lamanya berlangganan), Internet/Phone Service eg. PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges juga akan dianalisa mana yang mempunyai dampak tinggi terhadap churn rate.

Maka pada akhirnya dalam marketing campaign untuk menarik pelanggan baru di masa mendatang, dapat ditargetkan ke demographic calon-calon customer berdasarkan insights tersebut. 

**Goals**
- Mengurangi tingkat churn rate dibawah 10% per kwartal 3 bulan dengan menggunakan model machine learning (ML) yang telah di buat. Dengan tingkat churn rate dibawah 10% akan sangat menghemat biaya agency atau freelancer untuk mendapatkan pelanggan baru dan akhirnya dapat membantu benchmark profit perusahaan. Data 10% tersebut didapat dari penelitian artikel https://djournals.com/klik/article/download/829/596/3804 yang mengindikasikan churn rate per bulan di industri telekomunikasi di Amerika adalah 2-3% (Atau sekitar 10% per kwartal 3 bulan)
- Mampu memprediksi variabel-variable atau sasaran demographic potensi customer yang akan ditargetkan. Ini juga akan meningkatkan efisiensi biaya marketing campaign dan mengurangi tingkat kebocoran.
- Mendukung strategi bisnis dengan memberikan daftar pelanggan berisiko tinggi yang siap ditindaklanjuti untuk retensi.

**Metric Evaluation**
Dalam penentuan metric yang digunakan untuk studi kasus ini, akan dilihat dari komponen Confusion Matrix (TP, FP, TN dan FN). Terakit dengan kasus tersebut maka FN (False Negatives) - pelanggan churn tapi tidak terdeteksi - kerugian besar dan permanen. Bila dibanding dengan FP (False Positives) - pelanggan loyal tapi tetap diberikan diskon atau promo - biaya relatif kecil dan terkontrol. Bila dilihat dari sisi bisnis, FN jauh lebih mahal dari FP. 

Menurut artikel dalam https://tridenstechnology.com/telecom-churn/ data APRU (average revenue per customer - pendapatan per pelanggan) adalah $50 per perlanggan per bulan. Dan biaya retensi untuk satu pelanggan adalah rata-rata $50 (misalkan dengan pemberian gratis langganan 1 bulan atau voucher diskon). Untuk medapatkan pelanggan baru cost nya 6-7x daripada mempertahankan pelanggan loyal - yang artinya memakan biaya $300-350 untuk mendapatkan pelanggan baru.

Singkatnya komponen - komponen dalam confusion_matrix antara lain: 
- TP (True Positives) - pelanggan prediksi churn dan benar churn - retensi tepat sasaran
- FP (False Positives) - prediksi churn tapi loyal - dampak biaya promo
- FN (False Negatives) - tidak terdeteksi tapi churn - kehilangan revenue - Yang difokuskan pada studi kasus ini
- TN (True Negatives) - loyal dan diprediksi loyal - tidak ada kerugian biaya

Metrik yang berhubungan erat dengan komponen confusion matrix FN adalah:
- F2-Score - Recall diberikan bobot 4x dibanding Precision - F2 memberi bobot lebih pada penurunan FN.


Sangat diharapkan model yang telah dibuat dapat meminimalisasi False Negatives (Menekan biaya marketing yang terbuang sia-sia) dan False Positives. Metric yang akan digunakan adalah F2 Score.

**Batasan Masalah (Project Limitation)**
- Target Data Imbalance (Data yang dipelajari jauh lebih sedikit). 
- Jumlah pelanggan yang tersedia hanya sekitar 7000 pelanggan. Jumlah ini diasumsikan sebagai sample dari keseluruhan data pelanggan yang ada. Dengan kata lain data tersebut kurang menjadi representasi menyeluruh dengan perkiraan jumlah pelanggan perusahaan telco yang secara real.

**Analytic Approach**

Project ini akan terbagi dalam tahap-tahap berikut:
- Load & Explore Data
- Data Understanding
- Data Cleaning (EDA)
- Split Train/Test
- Preprocessing
- Modelling
- Evaluation
- Hyperparameter Tuning
- Confusion Matrix
- Feature Importance
- Model Deployment
