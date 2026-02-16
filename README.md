# EE563 - Artificial Intelligence in Practice

Bu repo, **EE563 - Artificial Intelligence in Practice** dersi kapsamında işlenen konuların kodlarını ve uygulamalarını içermektedir.

## 📁 Klasör Yapısı

```text
.
├── 01_getting_started/              # PyTorch'a Giriş - Temel Kavramlar
├── 02_pytorch_blitz/                # PyTorch Blitz ve Örneklerle PyTorch
│   ├── 01_PyTorch_60_Minutes_Blitz/ # 60 Dakikada PyTorch
│   └── 02_learning_pytorch_with_examples/ # Örneklerle PyTorch Öğrenimi
├── 04_openmmlab/                    # OpenMMLab Kütüphaneleri
│   ├── mmdetection/                 # Nesne Tespiti ve Instance Segmentation
│   └── mmsegmentation/              # Semantic Segmentation
├── 05_mediapipe/                    # MediaPipe Uygulamaları
├── Presentation/                    # Ders Sunumu ve Notlar
└── README.md                        # Proje Açıklaması
```

## 📚 Ders İçeriği

### 1️⃣ PyTorch'a Giriş (Getting Started with PyTorch)

**Klasör:** `01_getting_started/`

PyTorch'un temel bileşenlerini ve çalışma mantığını anlamak için hazırlanan başlangıç eğitimleri. FashionMNIST veri seti kullanılarak sıfırdan model eğitimi gerçekleştirilir.

| Notebook | Konu | Açıklama |
|----------|------|----------|
| [0_quickstart.ipynb](./01_getting_started/0_quickstart.ipynb) | Hızlı Başlangıç | PyTorch'un genel akışına hızlı giriş |
| [1_tensors.ipynb](./01_getting_started/1_tensors.ipynb) | Tensorlar | PyTorch'un temel veri yapısı |
| [2_datasets_and_dataloaders.ipynb](./01_getting_started/2_datasets_and_dataloaders.ipynb) | Veri Yükleme | Dataset ve DataLoader kullanımı |
| [3_transforms.ipynb](./01_getting_started/3_transforms.ipynb) | Transformlar | Veri ön işleme ve augmentation |
| [4_build_model.ipynb](./01_getting_started/4_build_model.ipynb) | Model Oluşturma | nn.Module ile sinir ağı tasarımı |
| [5_autograds.ipynb](./01_getting_started/5_autograds.ipynb) | Autograd | Otomatik türev hesaplama |
| [6_optimization.ipynb](./01_getting_started/6_optimization.ipynb) | Optimizasyon | Model parametrelerini optimize etme |
| [7_saveloadrun.ipynb](./01_getting_started/7_saveloadrun.ipynb) | Kaydet/Yükle | Model kaydetme ve yükleme |

**Kaynaklar:** [PyTorch - Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)

---

### 2️⃣ PyTorch Blitz ve Örneklerle PyTorch

**Klasör:** `02_pytorch_blitz/`

#### 📖 Deep Learning with PyTorch: A 60 Minute Blitz

PyTorch'u hızlı ve etkili bir şekilde öğrenmek için hazırlanan seri. CIFAR10 veri seti ile görüntü sınıflandırma uygulaması.

| Notebook | Konu | Açıklama |
|----------|------|----------|
| [03_neural_networks.ipynb](./02_pytorch_blitz/01_PyTorch_60_Minutes_Blitz/03_neural_networks.ipynb) | Sinir Ağları | nn modülü ile ağ oluşturma, forward/backward |
| [04_cifar10_tutorial.ipynb](./02_pytorch_blitz/01_PyTorch_60_Minutes_Blitz/04_cifar10_tutorial.ipynb) | Sınıflandırıcı Eğitimi | CIFAR10 ile CNN eğitimi ve test |

**Not:** Tensors ve Autograd konuları için bölüm 1'deki ilgili notebooklar kullanılmıştır.

**Kaynaklar:** [Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

#### 📖 Learning PyTorch with Examples

Numpy'dan başlayarak PyTorch'un modern yeteneklerine kademeli geçişi örneklerle gösterir.

| Notebook | Konu | Açıklama |
|----------|------|----------|
| [01_learning_pytorch_wiith_examples.ipynb](./02_pytorch_blitz/02_learning_pytorch_with_examples/01_learning_pytorch_wiith_examples.ipynb) | Örneklerle PyTorch | Numpy → Tensor → Autograd → nn → optim |

**Kaynaklar:** [Learning PyTorch with Examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)

---

### 3️⃣ OpenMMLab - Bilgisayarlı Görü Uygulamaları

**Klasör:** `04_openmmlab/`

#### 🎯 MMDetection - Nesne Tespiti ve Instance Segmentation

**Alt Klasör:** `mmdetection/`

MMDetection kütüphanesi ile nesne tespiti, bbox tahmini ve instance segmentation uygulamaları.

| Notebook | Konu | Açıklama |
|----------|------|----------|
| [0_overview_and_getstarted.ipynb](./04_openmmlab/mmdetection/0_overview_and_getstarted.ipynb) | Kurulum ve İnference | DetInferencer kullanımı, batch/URL inference |
| [1_config.ipynb](./04_openmmlab/mmdetection/1_config.ipynb) | Config Sistemi | MMDetection config yapısı ve özelleştirme |
| [2_finetuning_on_custom_dataset.ipynb](./04_openmmlab/mmdetection/2_finetuning_on_custom_dataset.ipynb) | Fine-tuning | Balloon dataset ile özel model eğitimi |
| [MMDet_InstanceSeg_Tutorial.ipynb](./04_openmmlab/mmdetection/MMDet_InstanceSeg_Tutorial.ipynb) | Instance Segmentation | Mask R-CNN ile instance segmentation |

**Kullanılan Modeller:**
- RTMDet (Real-time Detection)
- Mask R-CNN (Instance Segmentation)

**Kaynaklar:** [MMDetection Documentation](https://mmdetection.readthedocs.io/)

#### 🖼️ MMSegmentation - Semantic Segmentation

**Alt Klasör:** `mmsegmentation/`

MMSegmentation kütüphanesi ile semantic segmentation uygulamaları. Stanford Background ve Cityscapes veri setleri kullanılır.

| Notebook | Konu | Açıklama |
|----------|------|----------|
| [0_installing_and_getstart.ipynb](./04_openmmlab/mmsegmentation/0_installing_and_getstart.ipynb) | Kurulum ve Başlangıç | MMSegmentation kurulumu ve temel inference |
| [1_config.ipynb](./04_openmmlab/mmsegmentation/1_config.ipynb) | Config Yapısı | Config dosyaları ve model yapılandırması |
| [2_mmseg_tutorial.ipynb](./04_openmmlab/mmsegmentation/2_mmseg_tutorial.ipynb) | Segmentation Tutorial | Stanford Background ile model eğitimi |

**Kullanılan Modeller:**
- PSPNet (Pyramid Scene Parsing Network)
- FCN (Fully Convolutional Networks)

**Kaynaklar:** [MMSegmentation Documentation](https://mmsegmentation.readthedocs.io/)

---

### 4️⃣ MediaPipe - Gerçek Zamanlı Bilgisayarlı Görü

**Klasör:** `05_mediapipe/`

Google MediaPipe kütüphanesi ile gerçek zamanlı yüz ve poz analizi uygulamaları.

| Notebook | Konu | Açıklama |
|----------|------|----------|
| [1_face_detection.ipynb](./05_mediapipe/1_face_detection.ipynb) | Yüz Tespiti | Resimlerde yüz tespiti ve bbox çizimi |
| [2_face_landmark.ipynb](./05_mediapipe/2_face_landmark.ipynb) | Yüz Landmark | 468 noktalı yüz mesh tespiti |
| [3_pose_landmark.ipynb](./05_mediapipe/3_pose_landmark.ipynb) | Poz Kestirimi | 33 noktalı vücut poz analizi |

**Ek Dosyalar:**
- `face_landmark_camera.py` - Kamera ile gerçek zamanlı yüz landmark tespiti
- Pre-trained model dosyaları (.tflite, .task)

**Kaynaklar:** 
- [MediaPipe Face Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)

---

### 5️⃣ Sunum ve Dokümantasyon

**Klasör:** `Presentation/`

Ders çalışmaları kapsamında hazırlanan sunum materyalleri ve notlar.

- `presentation.html` - Final sunumu
- `plan.md` - Sunum planı ve yapısı
- `notes.md` - Ders notları
- `images/` - Sunum görselleri

---

## 🛠️ Gereksinimler

### Temel Kütüphaneler
```bash
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
pillow
```

### OpenMMLab Kütüphaneleri
```bash
# MMDetection
pip install openmim
mim install mmengine
mim install mmcv
mim install mmdet

# MMSegmentation
mim install mmsegmentation
```

### MediaPipe
```bash
pip install mediapipe
opencv-python
```

**Not:** Detaylı kurulum talimatları ilgili klasörlerdeki README.md dosyalarında bulunabilir.

---

## 📖 Kaynaklar

### Resmi Dökümantasyon
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
- [MMDetection Documentation](https://mmdetection.readthedocs.io/)
- [MMSegmentation Documentation](https://mmsegmentation.readthedocs.io/)
- [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe/solutions/guide)

### Veri Setleri
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [COCO Dataset](https://cocodataset.org/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html)

---

## 👨‍🏫 Ders Bilgileri

- **Ders Kodu:** EE 563
- **Ders Adı:** Artificial Intelligence (AI) in Practice
- **Eğitmen:** Cihan Göksu, PhD.
- **Dönem:** 2025-2026 Güz

---

## 📝 Lisans

Bu repo eğitim amaçlı hazırlanmıştır. Kullanılan tüm kütüphaneler ve veri setleri kendi lisanslarına tabidir.
