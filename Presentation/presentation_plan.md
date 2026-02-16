# PyTorch ve Computer Vision Eğitim Süreci - Sunum Planı

> **Sunum Prensibi:** Much images / Less text / No code / No math

---

## 📊 Genel Özet

| Bölüm | Slayt Sayısı | Süre (dk) |
|-------|--------------|-----------|
| Giriş | 2 | 2 |
| Bölüm 1: PyTorch Getting Started | 3 | 4 |
| Bölüm 2: PyTorch Blitz | 1 | 2 |
| Bölüm 3: MMDetection | 16 | 12-15 |
| Bölüm 4: MMSegmentation | 13 | 10-12 |
| Bölüm 5: MediaPipe | 8 | 7-9 |
| Sonuç | 1 | 2 |
| **TOPLAM** | **44** | **37-46 dk** |

---

## 📋 Slayt Detayları

| Slayt No | Bölüm | Başlık | İçerik (1 Cümle Özet) | Görsel Tasviri |
|---|---|---|---|---|
| 1 | Giriş | PyTorch ve Computer Vision Eğitim Yolculuğum | Sunum başlığı ve genel tanıtım | PyTorch, OpenMMLab ve MediaPipe logolarının estetik, modern bir kolajı. |
| 2 | Giriş | İçindekiler | Sunum akışını gösteren genel bakış slaytı | PyTorch, MMDetection, MMSegmentation ve MediaPipe başlıklarının ikonlarla listelendiği şık bir menü tasarımı. |
| 3 | Bölüm 1 | PyTorch Getting Started - Genel Bakış | FashionMNIST ile tensors, datasets, model building, autograd, optimization konularının özeti | FashionMNIST veri setinden örnek görüntüler (ayakkabı, çanta) ve basit bir sinir ağı şeması. |
| 4 | Bölüm 1 | Custom Dataset Nasıl Oluşturulur? | Dataset sınıfından miras alıp __init__, __len__ ve __getitem__ metodlarını implement ederek özel veri seti oluşturma | Klasör yapısı -> __getitem__ -> Tensor dönüşümünü gösteren adım adım akış şeması. |
| 5 | Bölüm 1 | Lambda Fonksiyonu ve One-Hot Encoding | Loss function ile target formatının uyumlu olması gerekir; MSELoss için lambda ile one-hot encoding yapılır | Tamsayı etiket (5) ile One-Hot vektör ([0,0,0,0,0,1...]) dönüşümünü gösteren ok diyagramı. |
| 6 | Bölüm 2 | PyTorch Evrimi - 8 Adımda İlerleme | NumPy'dan başlayıp dynamic graphs'a kadar PyTorch ile model geliştirme evriminin gösterimi | NumPy'dan Dynamic Graph'a giden 8 adımı gösteren merdiven veya zaman çizelgesi görseli. |
| 7 | Bölüm 3 | Object Detection ile Nesneleri Bul ve Tanımla | MMDetection bölümü başlangıç slaytı | Karmaşık bir sokak görüntüsü üzerinde tespitilmiş çok sayıda nesne (yaya, araç) ve renkli kutular. |
| 8 | Bölüm 3 | Object Detection Temelleri | Classification vs detection, bbox formatları, anchor-based vs anchor-free, one-stage vs two-stage kavramları | Bir yanda "Classification" (tek etiket), diğer yanda "Detection" (bbox + etiket) görsel karşılaştırması. |
| 9 | Bölüm 3 | MMDetection Framework | Modüler tasarım, 300+ model, config sistemi ve pretrained model desteği | MMDetection logosu ve merkezde modüler yapıyı simgeleyen (yapboz parçaları gibi) şema. |
| 10 | Bölüm 3 | Model Mimarisi: Backbone → Neck → Head | Input'tan output'a kadar tüm pipeline ve her bileşenin rolü | Backbone, Neck ve Head bloklarını birbirine bağlayan boru hattı (pipeline) diyagramı. |
| 11 | Bölüm 3 | Faster R-CNN: Two-stage Detector | RPN ile bölge önerir, sonra sınıflandırır - tam ekran mimari diyagramı ve detection örnekleri | Faster R-CNN mimari şeması (RPN + ROI Pooling) ve yanında insan tespiti örneği. |
| 12 | Bölüm 3 | YOLO: One-stage Detector | Tek seferde tüm resmi analiz eden real-time model - grid sistemi ve hız karşılaştırması | Görüntüyü SxS gridlere bölen ızgara sistemi ve hız (FPS) karşılaştırma grafiği. |
| 13 | Bölüm 3 | RTMDet: Modern Anchor-free Detector | Anchor-free yaklaşım ile hız-hassasiyet dengesi - performans grafikleri ve sonuçlar | RTMDet'in Hız (Latency) vs Doğruluk (AP) saçılım grafiği (scatter plot). |
| 14 | Bölüm 3 | Loss Fonksiyonları | Classification loss (Cross Entropy, Focal) ve localization loss (Smooth L1, IoU, GIoU) | Loss değerinin düşüş grafiği ve IoU kesişim alanlarını gösteren şema. |
| 15 | Bölüm 3 | Evaluation Metrics | IoU, NMS, mAP, AP50, AP75 ve COCO metriği | İki kutunun (Ground Truth vs Prediction) kesişimini ve birleşimini (IoU) gösteren renkli diyagram. |
| 16 | Bölüm 3 | Dataset ve Annotation | COCO formatı, JSON yapısı ve custom dataset entegrasyonu | Bir JSON dosyasının ağaç yapısı ve yanında buna karşılık gelen etiketli bir görüntü görseli. |
| 17 | Bölüm 3 | Training ve Optimization | Data loading, forward/backward pass, pretrained models ve transfer learning | Veri akışını (Data -> Model -> Loss -> Optimizer) gösteren döngüsel şema. |
| 18 | Bölüm 3 | Overfitting ve Regularization | Training-validation loss ayrışması, data augmentation ve early stopping | Training ve Validation loss eğrilerinin ayrıştığı grafik ve data augmentation örnekleri. |
| 19 | Bölüm 3 | Zorluk: Küçük Objeler | Downsampling ile bilgi kaybı problemi ve FPN ile çözümü - before/after görselleri | Before: Tespit edilememiş küçük kuşlar. After: FPN ile tespit edilmiş halleri. |
| 20 | Bölüm 3 | Zorluk: Class Imbalance | Arka plan-nesne dengesizliği problemi ve Focal Loss ile çözümü - dağılım grafikleri | Dengesiz sınıf dağılımı grafiği ve Focal Loss ile dengelenmiş terazi görseli. |
| 21 | Bölüm 3 | Pratik Uygulama - Balloon Dataset | RTMDet-tiny ile balon tespiti projesinin training ve inference sonuçları | Renkli balonlar üzerinde RTMDet-tiny modelinin bounding box çıktıları. |
| 22 | Bölüm 3 | MMDetection - Öğrendiklerim | Object detection, model mimarileri, custom dataset ve evaluation konularında kazanımlar | Anahtar kavramların (Backbone, IoU, Config) ikonlarla gösterildiği bir zihin haritası (mind map). |
| 23 | Bölüm 4 | Semantic Segmentation: Piksel Seviyesinde Anlama | MMSegmentation bölümü başlangıç slaytı | Bir şehir görüntüsünün piksel piksel renklendirilmiş semantic segmentation çıktısı. |
| 24 | Bölüm 4 | Segmentation Temelleri | Semantic vs instance segmentation, pixel-level prediction ve output formatı | Aynı görüntü üzerinde Semantic (tüm arabalar aynı renk) vs Instance (her araba farklı renk) karşılatırması. |
| 25 | Bölüm 4 | Encoder: Özellik Çıkarma | Downsampling ile spatial boyut azaltma ve soyut özellik çıkarma - feature map görselleştirmeleri | Görüntünün küçülerek (downsampling) feature map'e dönüşmesini gösteren huni şeklinde diyagram. |
| 26 | Bölüm 4 | Decoder: Piksel Tahmini | Upsampling ile orijinal çözünürlüğe dönme ve piksel-level sınıflandırma - reconstruction görselleri | Feature map'in büyüyerek (upsampling) orijinal boyuta ve segmentation mask'e dönüşümü. |
| 27 | Bölüm 4 | Skip Connections | Encoder'daki detay bilgisini decoder'a taşıma - U-Net diyagramı ve etki karşılaştırması | U-Net mimarisindeki detay taşıyan gri okları (skip connections) gösteren şema. |
| 28 | Bölüm 4 | MMSegmentation Framework | PyTorch tabanlı framework, backbone + decode head mimarisi ve config sistemi | MMSegmentation modüllerini (Datasets, Stylized Models, Backbones) gösteren blok diyagram. |
| 29 | Bölüm 4 | Popüler Segmentation Modelleri | FCN, U-Net, PSPNet, DeepLab ve farklı decode head'lerin karşılaştırması | FCN, PSPNet ve DeepLab mimarilerinin basitleştirilmiş yan yana çizimleri. |
| 30 | Bölüm 4 | Loss ve Evaluation | Cross Entropy, Dice Loss, Focal Loss ve mIoU metriği | Prediction mask ile Ground Truth mask'in çakışmasını (mIoU) gösteren görsel. |
| 31 | Bölüm 4 | Dataset ve Annotation | RGB image + pixel mask formatı, index-based mask ve train/val split | Orijinal fotoğraf ve yanında renk kodlu segmentation maskesi (PNG). |
| 32 | Bölüm 4 | Training Process | Pretrained backbones, data augmentation ve epoch bazlı validation | Pretrained bir backbone ağırlıklarının (ImageNet) segmentation modeline aktarılmasını simgeleyen görsel. |
| 33 | Bölüm 4 | Zorluklar ve Çözümler | Class imbalance (Dice Loss) ve boundary accuracy (skip connections) | Net olmayan sınıf sınırları (bulanık) vs Skip connection ile netleşmiş sınırlar. |
| 34 | Bölüm 4 | Pratik Uygulama - Stanford Background | 8 sınıflı dataset ile FCN+ResNet50 segmentation sonuçları | Stanford Background veri setinden örnek bir çıktı (gökyüzü, ağaç, yol ayrımı). |
| 35 | Bölüm 4 | MMSegmentation - Öğrendiklerim | Semantic segmentation, encoder-decoder, pixel-level prediction ve custom dataset konularında kazanımlar | Segmentation pipeline'ını özetleyen ikonik bir akış şeması. |
| 36 | Bölüm 5 | MediaPipe: Real-time Computer Vision | MediaPipe bölümü başlangıç slaytı | MediaPipe logosu ve el/yüz/vücut takibi yapan bir insan silüeti. |
| 37 | Bölüm 5 | MediaPipe Framework | Google'ın real-time, cross-platform, TFLite optimizasyonlu computer vision çözümleri | Android, iOS, Web ve Python logoları ile cross-platform vurgusu. |
| 38 | Bölüm 5 | Face Detection | BlazeFace modeli ile ultra-fast yüz tespiti ve 6 keypoint çıkarımı | Yüz etrafında bounding box ve 6 temel nokta (göz, burun, kulak, ağız). |
| 39 | Bölüm 5 | Face Landmark Mesh: 468 Nokta | 468 3D landmark ile detaylı yüz haritası - tam ekran mesh görselleştirmesi ve bölge detayları | Yüz üzerinde 468 noktanın oluşturduğu detaylı, örümcek ağına benzeyen 3D mesh yapısı. |
| 40 | Bölüm 5 | Pose Landmark Detection | BlazePose ile 33 3D landmark kullanarak üst ve alt vücut takibi | İnsan vücudu üzerinde 33 temel eklem noktasını gösteren iskelet çizimi. |
| 41 | Bölüm 5 | Real-time Performance ve Uygulamalar | TFLite, GPU acceleration ve AR filtreler, fitness tracking gibi kullanım alanları | Cep telefonunda çalışan bir AR filtre uygulaması ekran görüntüsü. |
| 42 | Bölüm 5 | Pratik Uygulama - Face Landmark Camera | Webcam'den real-time face landmark detection ve FPS gösterimi | Webcam görüntüsü üzerinde gerçek zamanlı yüz mesh'i ve köşede FPS sayacı. |
| 43 | Bölüm 5 | MediaPipe - Öğrendiklerim | Real-time vision, face/pose detection, TFLite deployment ve webcam uygulaması konularında kazanımlar | Real-time processing ve TFLite kavramlarını simgeleyen hız/işlemci ikonları. |
| 44 | Sonuç | Genel Özet ve Çıkarımlar | PyTorch, MMDetection, MMSegmentation ve MediaPipe ile öğrenme yolculuğunun özeti | Tüm yolculuğu (PyTorch -> MMDet -> MMSeg -> MediaPipe) birleştiren bir yol haritası görseli. |

---

## 🎨 Tasarım Prensipleri

- **Görsel ağırlık:** Her slaytın en az %60'ı görsel
- **Metin:** Maksimum 3-4 bullet point, kısa cümleler
- **Renk paleti:** PyTorch turuncu, OpenMMLab mavi, MediaPipe yeşil
- **Font:** Sans-serif, büyük boyutlar (başlık 44pt, metin 28-32pt)
- **Animasyon:** Minimal, dikkat dağıtmayan

---

## ✅ Hazırlık Checklist

- [ ] Notebook'lardan en iyi görsel örnekleri seç
- [ ] Model mimarisi diyagramlarını hazırla (Faster R-CNN, YOLO, RTMDet)
- [ ] Before/after karşılaştırma görselleri oluştur
- [ ] Feature map ve heatmap görselleştirmeleri export et
- [ ] Face landmark mesh görsellerini hazırla
- [ ] Grafikleri ve sonuçları export et
- [ ] İkonları topla (Font Awesome, Material Icons)
- [ ] Template seç ve renk paletini uygula
- [ ] Her slaytı plan doğrultusunda hazırla
- [ ] Akışı kontrol et ve son düzeltmeleri yap
