# CardioGraph-RL: EKG Sinyalinden Açıklanabilir Aritmi Tespiti için Graf Dikkat Ağı ve Nöro-Sembolik Füzyon

**Proje Raporu**

---

## Özet

Bu çalışmada, 12-derivasyonlu EKG sinyallerinden kalp ritim bozukluklarını tespit eden ve kararlarını klinik olarak açıklayabilen bir derin öğrenme sistemi geliştirilmiştir. Sistem; sinyal önişleme, zaman serisi görünürlük grafı oluşturma, Graf Dikkat Ağı (GAT) sınıflandırması ve Prolog tabanlı sembolik kural füzyonu aşamalarından oluşan uçtan uca bir pipeline üzerine inşa edilmiştir. PTB-XL veri setinde yapılan deneylerde AUC-ROC = 0.8702, F1 makro = 0.5974 değerlerine ulaşılmış; post-hoc Temperature Scaling ile GAT kalibrasyon hatası ECE = 0.017 düzeyine indirilmiştir. Modelin açıklanabilirliği için özgün Faithfulness@K metriği tanımlanmış ve attention ağırlıklarının klinik bulgularla örtüşme oranı %38 olarak ölçülmüştür. Çalışmanın temel katkısı, nöral ve sembolik bileşenlerin füzyonunun HYP ve CD sınıflarında F1 değerini iyileştirdiğinin deneysel olarak gösterilmesidir.

---

## 1. Giriş

Elektrokardiyografi (EKG), kalp ritim bozukluklarının tanısında altın standart olmayı sürdürmektedir. Bununla birlikte, 12-derivasyonlu kayıtların manuel okunması uzman hekimler için zaman alıcı ve yorucu bir süreçtir. Derin öğrenme tabanlı otomatik EKG yorumlama sistemleri bu açığı kapatma potansiyeli taşısa da klinik ortamlarda benimsenmelerinin önünde iki temel engel bulunmaktadır: (1) modellerin kararlarını klinisyene açıklayamaması ve (2) tahmin güvenilirliklerinin gerçekçi şekilde kalibre edilmemiş olması.

Bu proje, söz konusu iki sorunu aynı anda ele almak amacıyla tasarlanmıştır. EKG sinyalini bir zaman serisi görünürlük grafına dönüştürerek her kalp atımını düğüm olarak temsil eden sistem, Graf Dikkat Ağı (GAT) kullanarak hangi atımın tanıya daha fazla katkı sağladığını attention mekanizması aracılığıyla ortaya koymaktadır. Bunun yanı sıra, öğrenilen olasılık dağılımı ile klinisyen bilgisini kodlayan Prolog kuralları nöro-sembolik bir füzyon katmanında birleştirilerek son karar üretilmektedir.

---

## 2. İlgili Çalışmalar

EKG sınıflandırmasında konvolüsyonel ağlar (CNN) ve uzun-kısa süreli bellek (LSTM) ağları yaygın biçimde kullanılmaktadır. Hannun ve ark. (2019) derin sinir ağının kardiyologlarla karşılaştırılabilir performans sergilediğini göstermiştir. Ribeiro ve ark. (2020) PTB-XL üzerinde residual ağ ile yüksek AUC değerleri elde etmiştir. Ancak bu çalışmaların büyük çoğunluğu açıklanabilirlik boyutunu göz ardı etmektedir.

Graf sinir ağlarının EKG'ye uygulanması görece yeni bir alandır. Chen ve ark. (2021) elektrot konumlarından oluşturulan statik graf yapılarını kullanmış; bu yaklaşım derivasyonlar arası korelasyonu yakalamak açısından avantajlı olsa da atım düzeyinde dinamik örüntüleri temsil edememektedir. Bu çalışmada önerilen dinamik görünürlük grafı yaklaşımı, her kaydın kendine özgü topolojisini atım dizisinden otomatik olarak çıkarması bakımından söz konusu sınırlamayı aşmaktadır.

Açıklanabilir yapay zeka (XAI) bağlamında SHAP ve LIME yöntemleri post-hoc açıklama için kullanılmaktadır; ancak bu yöntemler modelin iç yapısından bağımsız olduğundan klinik anlamlılıkları tartışmalıdır. Bu projedeki attention tabanlı açıklanabilirlik, modelin karar mekanizmasıyla yapısal olarak bütünleşik olması nedeniyle bu boşluğu doldurmaktadır.

---

## 3. Veri Seti

Çalışmada PTB-XL veri seti kullanılmıştır. PTB-XL, 71.135 kayıttan oluşan halka açık en büyük klinik 12-derivasyonlu EKG veri setlerinden biridir. Çalışmada 21.837 hastanın kayıtları kullanılmış; kayıtlar 500 Hz örnekleme hızında ve yaklaşık 10 saniye uzunluğunda 12 derivasyondan oluşmaktadır.

Veri seti, PTB-XL'nin önerdiği stratified 10-fold bölme şeması kullanılarak ayrılmıştır:
- **Eğitim:** fold 1–8 (~17.469 kayıt)
- **Doğrulama:** fold 9 (~2.183 kayıt)
- **Test:** fold 10 (~2.185 kayıt)

Sınıflandırma hedefi beş üst-tanısal sınıftır (superdiagnostic):

| Sınıf | Açıklama | Kayıt Sayısı |
|---|---|---|
| NORM | Normal ritim | 8.994 |
| CD | İletim bozukluğu | 4.108 |
| MI | Miyokard enfarktüsü | 3.797 |
| STTC | ST-T değişikliği | 3.234 |
| HYP | Hipertrofi | 1.230 |

Veri setinde belirgin bir sınıf dengesizliği mevcuttur: NORM sınıfı HYP sınıfının yaklaşık 7.3 katıdır. Bu dengesizlik, model geliştirme sürecinde ciddi bir zorluk kaynağı olmuş ve deney sürecinde birden fazla farklı stratejiyi denemeyi gerektirmiştir.

---

## 4. Sistem Mimarisi

Önerilen sistem altı işlevsel bileşenden oluşmaktadır:

```
Ham EKG (500 Hz, 12 lead)
       ↓
[1] Sinyal Önişleme
       ↓
[2] Görünürlük Grafı İnşası
       ↓
[3] Graf Dikkat Ağı (CardioGAT)
       ↓
[4] Nöro-Sembolik Füzyon
       ↓
[5] Temperature Scaling (Kalibrasyon)
       ↓
Tanı + Güven Skoru + Açıklama
```

### 4.1 Sinyal Önişleme

Ham EKG sinyaline önce bant geçiren filtre uygulanmış, ardından Pan-Tompkins algoritması ile R-tepeleri tespit edilmiş ve her atım segmentlere ayrılmıştır.

**Bant Geçiren Filtre:** 0.5–40 Hz Butterworth filtresi, 4. derece Sobol (SOS) yapısında gerçekleştirilmiştir. İlk uygulamada scipy'nin varsayılan padlen değeri (24 örnek) kullanılmış; ancak 0.5 Hz kesim frekansı için gerekli yerleşme süresi yaklaşık 320 örnektir. Bu durum düşük frekanslı bileşenler içeren kayıtlarda ciddi kenar artefaktlarına yol açmıştır. Sorun, sinyalin her iki ucuna "wrap" modunda ~2 saniyelik dolgu eklenerek ve filtreleme sonrası orijinal bölgeye kırpılarak çözülmüştür.

**R-Tepe Tespiti:** neurokit2 kütüphanesinin Pan-Tompkins 1985 implementasyonu kullanılmıştır. İlk denemelerde R-tepesi tespitinin ham sinyal üzerinde çalıştırıldığı görülmüş; bu durumda bazı kayıtlarda 29 hatalı tepe tespit edilmiştir. Filtreleme adımının Pan-Tompkins'ten önce uygulanmasıyla bu sorun giderilmiştir.

**Beat Segmentasyonu:** Her R-tepesi etrafında ±200 örnek (±0.4 saniye) pencere alınmıştır. Kayıt sınırlarına taşan atımlar sessizce atlanmıştır.

**Özellik Çıkarımı:** Her atımın her derivasyonu için 8 istatistiksel özellik hesaplanmıştır: ortalama, standart sapma, çarpıklık, tepe-tepe genliği, basıklık, RMS, sıfır-geçiş oranı ve spektral ağırlık merkezi. 12 derivasyon × 8 özellik = **96 boyutlu özellik vektörü** elde edilmiştir.

Özellik sayısının artırılması (12 özellikten 96 özelliğe) model başarımında en belirleyici iyileştirmelerden birine yol açmıştır: doğrulama F1 değeri 0.44'ten 0.58'e yükselmiştir.

### 4.2 Zaman Serisi Görünürlük Grafı

Beat dizisinin bir grafik yapıya dönüştürülmesinde Natural Visibility Graph (NaturalVG) algoritması kullanılmıştır. NaturalVG'de iki atım arasında kenar oluşabilmesi için aralarındaki tüm atımların genliklerinin geometrik "görünürlük" koşulunu sağlaması gerekmektedir.

Klinik anlamlılığı artırmak amacıyla 12 derivasyon üç grupta ele alınmıştır:
- **İnferior:** II, III, aVF
- **Lateral:** I, aVL, V5, V6
- **Anterior:** V1, V2, V3, V4

Her grup için ayrı NaturalVG kurulmuş, kenar kümeleri birleşim (union) operasyonuyla birleştirilmiştir. Bu klinik gruplama, yalnızca tek bir derivasyondan elde edilen VG'ye kıyasla daha zengin ve klinik açıdan anlamlı bir topoloji sağlamaktadır.

**Kenar Ağırlıkları:** Her kenarın ağırlığı 1/(1+DTW) formülüyle hesaplanmıştır; burada DTW, kaynak ve hedef atımlar arasındaki Dynamic Time Warping uzaklığıdır. Bu formül ağırlığı (0, 1] aralığına normalize etmekte, benzer atımlar arasında daha güçlü bağlantı kurmaktadır. Hesaplama maliyetini düşürmek için DTW sonuçları kenar başına önbelleğe alınmıştır.

Toplam **21.363 graf** başarıyla oluşturulmuş ve PyTorch Geometric Data formatında kaydedilmiştir.

### 4.3 Graf Dikkat Ağı (CardioGAT)

Modelin omurgası üç katmanlı, sekiz başlıklı bir Graf Dikkat Ağıdır. Mimari ayrıntılar şu şekildedir:

| Parametre | Değer |
|---|---|
| Giriş boyutu | 96 |
| Gizli boyut | 128 |
| Dikkat başı | 8 |
| Dropout | 0.3 |
| Çıkış sınıfı | 5 |

Her katmana residual bağlantı eklenmiştir. Giriş boyutundan (96) gizli boyuta (128) geçişteki boyut uyumsuzluğu, 96→128 projeksiyon katmanıyla çözülmüştür.

Üçüncü GAT katmanının `return_attention_weights=True` parametresiyle çağrılması, her kenar için dikkat ağırlığı vektörü döndürmektedir. Bu ağırlıklar, hedef düğüm bazında toplanarak beat düzeyinde "önem skoru" olarak yorumlanmaktadır.

### 4.4 Nöro-Sembolik Füzyon

GAT'ın olasılıksal çıktısı ile klinisyen bilgisini kodlayan sembolik kuralların ağırlıklı ortalaması alınmaktadır:

```
p_final = (1 - α) · p_GAT + α · p_sembolik
```

Sembolik bileşen, 96 boyutlu özellik vektöründen türetilen klinik proxy değerlerini değerlendiren Prolog kurallarından oluşmaktadır:

| Kural | Eşik | Hedef Sınıf |
|---|---|---|
| ST elevasyonu > 0.10 mV | V2, V3, V4 ortalaması | MI |
| Patolojik Q < −0.10 mV | II, III, aVF minimumu | MI |
| T inversiyonu skoru < −0.50 | V3-V5 skewness | MI |
| Sokolow-Lyon > 3.50 mV | p2p_V1 + p2p_V5 | HYP |
| aVL voltajı > 1.10 mV | RMS aVL | HYP |
| ST çökmesi < −0.05 mV | II, V4-V6 minimumu | STTC |
| T değişikliği > 0.50 | Prekordiyal skewness std | STTC |
| QRS süresi > 120 ms | ZCR tabanlı proxy | CD |

Alpha parametresi doğrulama setinde grid search ile optimize edilmiş ve α = 0.35 değeri belirlenmiştir.

---

## 5. Geliştirme Süreci ve Karşılaşılan Zorluklar

### 5.1 Model Geliştirme: 11 Deneme

Model geliştirme süreci, birçok başarısızlık ve geri adımı kapsayan iteratif bir süreç olmuştur. Aşağıdaki tablo tüm denemeleri özetlemektedir:

| Deney | Val F1 | Konfigürasyon | Sonuç |
|---|---|---|---|
| D1 | 0.4282 | Baseline: 12-özellik, CrossEntropy | Başlangıç noktası |
| D2 | 0.2431 | FocalLoss + WeightedRandomSampler | Felaket — NORM çöktü |
| D3 | 0.4436 | Residual bağlantı + lr=1e-3 | Küçük iyileşme |
| D4 | 0.4444 | 5 özellik değişimi | Önemsiz fark |
| D5 | 0.5832 | 48-özellik (12×4 stat) | Büyük sıçrama |
| D6 | ~0.6017 | Klinik graf + cosine annealing | Anlamlı ilerleme |
| D7 | — | batch=256 | Gerileme, iptal edildi |
| D8 | 0.5857 | SAGPool pooling | D6'dan kötü, iptal |
| D9 | 0.6111 | 96-özellik (12×8 stat) | Yeni zirve |
| D10 | 0.5434 | FocalLoss + Sampler (tekrar) | Büyük gerileme |
| **D11-A** | **0.6130** | CE + sınıf ağırlıkları, shuffle | Final model |

#### Kritik Hata: D2 ve D10 — FocalLoss ile WeightedRandomSampler Etkileşimi

Deney sürecinin en öğretici kısmı D2 ve D10 denemeleridir. İki denemede de FocalLoss ve WeightedRandomSampler aynı anda kullanılmıştır. Her iki durumda da doğrulama F1 değeri dramatik biçimde düşmüştür:

- D2'de NORM F1 değeri 0.81'den neredeyse sıfıra yaklaşmıştır.
- D10'da NORM 0.81→0.64'e, HYP ise 0.43→0.31'e gerilemiştir.

Bu davranışın nedeni şudur: WeightedRandomSampler azınlık sınıflarını 75 kata varan oranlarda örneklerken, FocalLoss da aynı zor/azınlık örneklerine çift ceza uygulamaktadır. Bu çifte baskı, modeli yalnızca azınlık sınıflarına odaklanmaya ve çoğunluk sınıfı olan NORM'u görmezden gelmeye zorlamaktadır. Çözüm basit ama kritiktir: iki teknikten yalnızca biri kullanılmalıdır. D11-A'da sınıf ağırlıklı CrossEntropyLoss tercih edilmiş ve bu sorun giderilmiştir.

#### Özellik Zenginleştirmenin Önemi: D1→D5

D1'de her derivasyon için yalnızca 12 basit özellik (P amplitüdü, QRS amplitüdü, RR aralığı vb.) kullanılmıştır. D5'te bu yaklaşım tamamen değiştirilmiş; her derivasyon için 4 istatistiksel özellik (ortalama, standart sapma, tepe-tepe, RMS) hesaplanmıştır. Bu değişiklik F1'i 0.44'ten 0.58'e, yani yaklaşık %32 iyileştirmiştir.

Bu sonuç, EKG sinyalinin ham morfolojik özelliklerinin görece basit istatistiksel temsiliyle bile önemli ayrımsal bilgi taşıdığını göstermektedir. D9'da özellik sayısı 48'den 96'ya çıkarıldığında (4 → 8 istatistik) ek bir F1 iyileşmesi elde edilmiştir.

#### SAGPool Denemesi: D8

Graf havuzlama için Self-Attention Graph Pooling (SAGPool) denenmiştir (D8). Beklentinin aksine SAGPool, global ortalama havuzlamadan daha düşük performans göstermiştir (0.5857 vs 0.6017). Bu sonuç, havuzlama sonrası bilgi kaybının, SAGPool'un sağladığı seçici düğüm vurgusundan daha ağır bastığına işaret etmektedir. Kısa kayıtlarda (8–15 atım) global ortalama havuzlama bilgiyi daha iyi koruduğu için tercih edilmiştir.

### 5.2 Sembolik Katman: MI Proxy Sorunu

Nöro-sembolik füzyonun performansa katkısı sınıf bazında incelendiğinde ilginç bir örüntü ortaya çıkmaktadır:

| Sınıf | GAT F1 (α=0) | Füzyon F1 (α=0.35) | Değişim |
|---|---|---|---|
| NORM | — | — | Değişmedi |
| MI | 0.5095 | 0.4992 | ↓ −0.0103 |
| STTC | — | — | Nötr |
| CD | 0.64 | 0.69 | ↑ +0.05 |
| HYP | 0.45 | 0.50 | ↑ +0.05 |

Sembolik katman HYP ve CD için net bir iyileşme sağlarken MI için hafif bir gerilemeye yol açmıştır. Bunun temel nedeni, MI tespiti için kullanılan proxy özelliklerin (96 boyutlu özellik vektöründen türetilen ST elevasyonu ve Q dalgası proxy'leri) gerçek ST segmenti analizinden önemli ölçüde farklı olmasıdır.

96 boyutlu özellik vektörü beat başına istatistiksel özetler içermektedir; oysaki MI için kritik olan ST elevasyonu, milisaniye düzeyindeki segment analizi gerektirmektedir. Beat ortalama voltajından hesaplanan bir proxy, QRS kompleksi ile ST segmentini birbirine karıştırmaktadır. Bu sınırlama gelecek çalışmalar için önemli bir iyileştirme noktası olarak tespit edilmiştir.

### 5.3 Kalibrasyon: Temperature Scaling

Model kalibrasyonunu ölçmek için Expected Calibration Error (ECE) metriği kullanılmıştır. Temperature Scaling öncesinde GAT tek başına ECE = 0.0205 değeriyle zaten iyi kalibre olduğu görülmüştür. LBFGS optimizasyonuyla bulunan optimal sıcaklık parametresi T = 1.0256 ile GAT ECE = 0.0171'e iyileştirilmiştir.

İlginç bir bulguy olarak, Temperature Scaling'in füzyon çıktısı üzerinde beklenen iyileştirmeyi sağlamadığı gözlemlenmiştir (Füzyon ECE: 0.1533 → 0.1573). Bu durum şöyle açıklanmaktadır: sembolik bileşen deterministik Prolog kurallarından oluşmakta ve yalnızca 0 veya sabit değerler üretmektedir. Bu durum füzyon olasılık dağılımının kalibrasyonunu bozarken GAT'ın kendi çıktısı iyi kalibre kalmaktadır. Bu bulgu, demo arayüzünde güven skoru için füzyon yerine GAT çıktısını kullanma kararını doğrulamaktadır.

### 5.4 Görünürlük Grafı: ts2vg Kütüphane Uyumsuzluğu

Graf oluşturma aşamasında ts2vg kütüphanesinin belgelerinde yer alan sınıf adı `NaturalVisibilityGraph` iken kurulu sürümde (v1.2.4) sınıfın `NaturalVG` olarak yeniden adlandırıldığı görülmüştür. Bu uyumsuzluk başlangıçta `ImportError` ile karşılaşılmasına neden olmuş ve kütüphane kaynak kodunun incelenmesiyle çözülmüştür.

### 5.5 Veri Bölme ve Sızıntı Önleme

PTB-XL'nin 10-fold stratified bölme şeması kullanılmıştır. Bu şemada fold 9 doğrulama, fold 10 test seti olarak ayrılmıştır. Hiperparametre optimizasyonu (alpha grid search, Temperature Scaling) yalnızca doğrulama seti üzerinde gerçekleştirilmiş; test seti yalnızca final değerlendirmede bir kez kullanılmıştır. Bu yaklaşım veri sızıntısını önlemektedir.

---

## 6. Deneysel Sonuçlar

### 6.1 Model Performansı

Final model (D11-A, α=0.35) PTB-XL test seti üzerinde aşağıdaki sonuçları elde etmiştir:

| Metrik | Değer |
|---|---|
| AUC-ROC (macro OvR) | 0.8702 |
| F1 makro | 0.5974 |
| F1 ağırlıklı | 0.6635 |
| GAT ECE (T=1.026) | 0.0171 |
| Füzyon ECE | 0.1573 |

**Sınıf Bazlı F1 (test seti):**

| NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|
| 0.81 | 0.49 | 0.58 | 0.69 | 0.50 |

NORM ve CD sınıflarında güçlü performans elde edilirken MI ve HYP, sınıf dengesizliği ve temsil güçlükleri nedeniyle daha düşük F1 değerleri sergilemiştir.

### 6.2 Faithfulness@K Analizi

Önerilen Faithfulness@K metriği, attention ağırlıklarının en yüksek K atımının klinik olarak anlamlı bulgularla (ST elevasyonu, Q dalgası, T inversiyonu) ne ölçüde örtüştüğünü ölçmektedir.

| K | Faithfulness@K |
|---|---|
| 1 | 0.3876 |
| 3 | 0.3829 |
| 5 | 0.3802 |

Rastgele seçim beklentisi 1/5 = 0.20 olduğundan, elde edilen değerler (%38) modelin dikkatinin klinik açıdan anlamlı bölgelere yöneldiğini göstermektedir; ancak bu örtüşme tam değildir. Bu sonuç, attention mekanizmasının açıklanabilirlik aracı olarak kullanımında dikkatli olmak gerektiğine işaret etmektedir.

### 6.3 Cross-Dataset Değerlendirmesi (MIT-BIH)

Modelin genellenebilirliğini test etmek amacıyla MIT-BIH Arrhythmia Database üzerinde değerlendirme yapılmıştır. MIT-BIH yalnızca 2 derivasyon içerdiğinden 12 derivasyona sıfır-dolgu uygulanmıştır. Elde edilen F1 = 0.0385 değeri oldukça düşüktür; bu beklenen bir sonuçtur çünkü:

1. 2 → 12 derivasyon sıfır-dolgusu ciddi veri bozulmasına yol açmaktadır.
2. PTB-XL ile MIT-BIH etiket sistemleri arasındaki kaba eşleme (örn. V/E → MI) anlam kayıplarına neden olmaktadır.
3. MIT-BIH kayıtları 360 Hz örnekleme hızındadır; 500 Hz'e çevrimde bilgi kaybı oluşmaktadır.

Bu sonuç, çapraz veri seti genellemesinin hâlâ açık bir araştırma problemi olduğunu doğrulamaktadır.

---

## 7. Tartışma

### 7.1 Graf Temsili ile CNN Karşılaştırması

Bu çalışmada konvolüsyonel ağlarla doğrudan karşılaştırma yapılmamıştır; ancak elde edilen AUC = 0.87 değeri literatürdeki CNN tabanlı yaklaşımlarla (Ribeiro ve ark., 2020: AUC ~0.90) karşılaştırılabilir düzeydedir. Graf temsilinin asıl avantajı performanstan ziyade yorumlanabilirliktir: attention ağırlıkları hangi atımın kritik olduğunu doğrudan gösterirken CNN'lerdeki bu tür bir yorumlama dolaylı yollarla (Grad-CAM gibi) elde edilmektedir.

### 7.2 Nöro-Sembolik Füzyonun Sınırları

Füzyon yaklaşımı HYP ve CD için net kazanımlar sağlamış, ancak MI için beklenmedik bir gerilemeye neden olmuştur. Bu sonuç, istatistiksel özellik vektörü tabanlı klinik proxy'lerin MI morfolojisini yeterince temsil edemediğini göstermektedir. Gelecek çalışmalarda neurokit2'nin sinyal düzeyinde P/QRS/T segmentasyon çıktılarının doğrudan kullanılması bu sorunu çözebilir.

Füzyon kalibrasyon sorunu (ECE = 0.157) da önemli bir sınırlılıktır. Deterministik Prolog kuralları yumuşak olasılıklar üretmediğinden, füzyon çıktısının kalibrasyonu bozulmaktadır. Prolog kurallarını belirsizlik modellemeyle destekleyen fuzzy logic veya Bayesian kural motoru, bu soruna çözüm getirebilir.

### 7.3 Açıklanabilirlik ve Klinik Kullanım

Faithfulness@K = 0.38 değeri, modelin dikkatinin klinik açıdan anlamlı bulgularla yaklaşık %38 oranında örtüştüğünü göstermektedir. Bu değer, rastgele seçimden (%20) anlamlı biçimde yüksek olmakla birlikte, klinik karar desteği için yeterli bir güven düzeyini temsil etmemektedir. Attention tabanlı açıklanabilirliğin klinik benimsenmesi için Faithfulness@K değerinin en az 0.70 seviyesine çıkarılması gerekmektedir.

---

## 8. Sonuç

Bu çalışmada, 12-derivasyonlu EKG sinyallerinden klinik açıklanabilirlik sağlayan bir aritmi tespit sistemi tasarlanmış ve hayata geçirilmiştir. PTB-XL veri setinde AUC-ROC = 0.87, F1 makro = 0.60 değerleri elde edilmiş; GAT kalibrasyon hatası ECE = 0.017 düzeyine indirilmiştir.

Geliştirme süreci boyunca en kritik bulgular şunlardır:

1. **FocalLoss ile WeightedRandomSampler birlikte kullanılmamalıdır.** İkisi birden uygulandığında model azınlık sınıflarına çifte baskı uyguladığından ciddi performans kayıpları ortaya çıkmaktadır.

2. **Özellik zenginleştirme, mimari karmaşıklığından daha etkilidir.** 12 özellikten 96 özelliğe geçiş, model mimarisindeki değişikliklerden daha büyük F1 iyileşmesi sağlamıştır.

3. **Sembolik katman sınıf bazında seçici katkı sunar.** HYP ve CD için faydalı, MI için zararlı olan bu asimetri, klinik proxy özelliklerin yeterliliğine doğrudan bağlıdır.

4. **Güven kalibrasyonu için füzyon ve GAT çıktıları ayrı kullanılmalıdır.** GAT iyi kalibre iken sembolik füzyon kalibrasyonu bozmaktadır.

Gelecek çalışmalar için üç temel yön önerilmektedir: (i) MI tespiti için neurokit2 sinyal düzeyinde segmentasyon entegrasyonu, (ii) Prolog kurallarının fuzzy mantıkla yumuşatılarak füzyon kalibrasyonunun iyileştirilmesi, (iii) Faithfulness@K hedefini ≥0.70'e taşıyacak attention düzenleyicilerin araştırılması.

---

## Kaynaklar

1. Hannun, A. Y., et al. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*, 25(1), 65–69.

2. Ribeiro, A. H., et al. (2020). Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications*, 11(1), 1760.

3. Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7(1), 154.

4. Veličković, P., et al. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.

5. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230–236.

6. Guo, C., et al. (2017). On calibration of modern neural networks. *International Conference on Machine Learning (ICML)*.

7. Lin, T. Y., et al. (2017). Focal loss for dense object detection. *IEEE International Conference on Computer Vision (ICCV)*.

8. Luque-Laguna, D., et al. (2021). Visibility graphs for time series analysis. *arXiv preprint*.

9. Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43–49.

10. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH arrhythmia database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.

---

*Geliştirme ortamı: Python 3.10, PyTorch 2.2, PyTorch Geometric 2.5, CUDA 12.x*
*Veri seti: PTB-XL (physionet.org/content/ptb-xl), MIT-BIH (physionet.org/content/mitdb)*
