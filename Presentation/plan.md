## 📋 Slayt Detayları

| Slayt No | Bölüm | Başlık | İçerik (1 Cümle Özet) | Görsel Tasviri |
|---|---|---|---|---|
| 1 | Giriş | PyTorch ve Computer Vision Eğitim Yolculuğum | Sunum başlığı ve genel tanıtım | PyTorch, OpenMMLab ve MediaPipe logolarının estetik, modern bir kolajı. |
| 2 | Giriş | İçindekiler | Sunum akışını gösteren genel bakış slaytı | PyTorch, MMDetection, MMSegmentation ve MediaPipe başlıklarının ikonlarla listelendiği şık bir menü tasarımı. |
| 3 | Bölüm 1 | PyTorch Getting Started - Genel Bakış | FashionMNIST ile tensors, datasets, model building, autograd, optimization konularının özeti | FashionMNIST veri setinden örnek görüntüler (ayakkabı, çanta) ve basit bir sinir ağı şeması. |
| 4 | Bölüm 1 | Custom Dataset Nasıl Oluşturulur? | Dataset sınıfından miras alıp __init__, __len__ ve __getitem__ metodlarını implement ederek özel veri seti oluşturma | Klasör yapısı -> __getitem__ -> Tensor dönüşümünü gösteren adım adım akış şeması. |
| 5 | Bölüm 1 | Lambda Fonksiyonu ve One-Hot Encoding | Loss function ile target formatının uyumlu olması gerekir; MSELoss için lambda ile one-hot encoding yapılır | Tamsayı etiket (5) ile One-Hot vektör ([0,0,0,0,0,1...]) dönüşümünü gösteren ok diyagramı. |
| 6 | Bölüm 2 | PyTorch Evrimi - 8 Adımda İlerleme | NumPy'dan başlayıp dynamic graphs'a kadar PyTorch ile model geliştirme evriminin gösterimi | NumPy'dan Dynamic Graph'a giden 8 adımı gösteren merdiven veya zaman çizelgesi görseli. |

* 7 mmdetection giriş

* 8 mmdetection 7 main parts

MMDetection consists of 7 main parts, apis, structures, datasets, models, engine, evaluation and visualization.

* **apis** provides high-level APIs for model inference.
* **structures** provides data structures like ``bbox``, ``mask``, and ``DetDataSample``.
* **datasets** supports various dataset for ``object detection``, ``instance segmentation``, and ``panoptic segmentation``.
    * ***transforms** contains a lot of useful data augmentation transforms.
    * **samplers** defines different data loader sampling strategy.
* **models** is the most vital part for detectors and contains different components of a detector.
    * **detectors** defines all of the detection model classes.
    * **data_preprocessors** is for preprocessing the input data of the model.
    * **backbones** contains various ``backbone`` networks.
    * **necks** contains various ``neck`` components.
    * **dense_heads** contains various detection ``heads that perform dense predictions``.
    * **roi_heads** contains various detection ``heads that predict from RoIs``.
    * **seg_heads** contains various ``segmentation heads``.
    * **losses** contains various loss functions.
    * **task_modules** provides modules for detection tasks. E.g. ``assigners``, ``samplers``, ``box coders``, and ``prior generators``.
    * **layers** provides some basic neural network layers.
* **engine** is a part for runtime components.
    * **runner** provides extensions for MMEngine’s runner.
    * **schedulers** provides schedulers for adjusting optimization hyperparameters.
    * **optimizers** provides optimizers and optimizer wrappers.
    * **hooks** provides various ``hooks`` of the runner.
* **evaluation** provides different metrics for evaluating model performance.
    * **metrics** contains different evaluation metrics.
    * **evaluators** provides evaluators for dataset evaluation.
* **visualization** is for visualizing detection results.

* 9 installation süreci

**1. Anaconda Prompt içinde yeni bir ortam oluşturun (Python 3.8 önerilir):**
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**2. PyTorch'u yükleyin (Versiyon 2.1.2 - Kritik Adım!):**
>En yeni sürümü yüklemeyin, uyumluluk için bu sürüm şart.

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

**3. OpenMIM ve MMEngine araçlarını kurun:**

```bash
pip install -U openmim
mim install mmengine
```

**4. MMCV'yi kurun (Versiyon 2.1.0):**
>PyTorch 2.1.2 ile tam uyumlu olan ve Windows'ta derleme hatası vermeyen sürüm budur.

```bash
mim install "mmcv==2.1.0"
```

**5. MMDetection'ı kurun:**

```bash
mim install mmdet
```

* 10 mmdetecetion quickstart - 1: detinferencer - DetInferencer'ı başlatmak için sadece **model adı** yeterli. Weights otomatik indirilecek.

* 11 2: görsel indirme işlemi - C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmdetection\demo\test_images\demo.jpg

* 12 3: inferencer ile inference perform edebiliyoruz. C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmdetection\demo\outputs\vis\demo.jpg

* 13 6 görsel ve 6 model testi. C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\6 images.png models = {
    'RTMDet-tiny': 'rtmdet_tiny_8xb32-300e_coco',       # En hızlı
    'RTMDet-small': 'rtmdet_s_8xb32-300e_coco',         # Dengeli
    'RTMDet-large': 'rtmdet_l_8xb32-300e_coco',         # En doğru RTMDet
    'Faster-RCNN': 'faster-rcnn_r50_fpn_1x_coco',       # Klasik two-stage
    'Mask-RCNN': 'mask-rcnn_r50_fpn_1x_coco',           # Instance segmentation
    'RetinaNet': 'retinanet_r50_fpn_1x_coco',           # Single-stage with focal loss
}

* 14 C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\1.png

* 15 C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\2.png

* 16 C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\3.png

* 17 C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\4.png

* 18 C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\5.png

* 19 C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\6.png

* 20 inferencer a direkt url linki veya batch (içinde birden fazla görsel olan liste) de verebiliyoruz

* 21 çıktıdan şu bilgilere erişebiliyoruz: Nesne 1: Label ID=13, Score=0.8762, BBox=[217.5468292236328, 172.820068359375, 457.94659423828125, 385.8176574707031]

* 22 mmcv.VideoReader ile modellere videolar da iletebiliyoruz input video: C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmdetection\demo\demo.mp4

* 23 video output C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmdetection\demo\output.mp4

* 24 Neden config var? (1 slide)

MMDetection’da “kod yazmaktan çok deneyi tarif ediyorsun”.
Model, dataset, eğitim döngüsü, optimizer, scheduler, hook’lar… hepsi tek bir yerde “deney tarifi” olarak duruyor.
Aynı altyapıyla farklı modelleri/datasetleri kolayca değiştirmenin ana yolu config.

* 25 Config’in ana blokları (1 slide)

model: backbone/neck/head + train_cfg/test_cfg (algoritmanın kendisi)
data_*: train_dataloader, val_dataloader, test_dataloader + pipeline(Load/Resize/Flip/Pack…)
train_cfg, val_cfg, test_cfg: runner/loop tipi ve epoch/iter ayarları
optim_wrapper: optimizer + gradient clip + (istersen AMP)
param_scheduler: LR warmup + MultiStep/Cosine vb.
default_hooks / custom_hooks: log, checkpoint, eval, visualization, seed…
env_cfg ve runtime: distributed backend, log level, resume/load_from vs.

* 26 En kritik fikir: Inheritance (base config) (1 slide)

Çoğu config “tam dosya” değil; bir veya birkaç base config’ten miras alıp sadece farkı yazar.
Tipik pattern:
_base_ = [model_base, dataset_base, schedule_base, runtime_base]
Böylece:
Tek satırla backbone veya dataset değiştirip deney üretirsin
Kopyala-yapıştır config şişmesini engellersin

* 27 Küçük değişiklik nasıl yapılır? (demo/örnek slide)

İki yol:
Config dosyasında alanı override etmek
Komut satırında --cfg-options ile “in-place” değiştirmek
Örnek anlatım:
“LR’ı değiştir, batch size’ı değiştir, pipeline’a bir augment ekle… hepsi config ile”

* 28 mask r cnn ile instance segmentation
- https://user-images.githubusercontent.com/40661020/143967081-c2552bed-9af2-46c4-ae44-5b3b74e5679f.png
- Mask R-CNN, “two-stage (iki aşamalı) detector” ailesinden bir instance segmentation modelidir: hem nesnenin kutusunu (bbox) hem de piksel seviyesinde maskesini üretir.

- Two-stage detector ne demek?
    - Stage 1 (Aday üretme / Proposal): Model önce görüntü üzerinde “nesne olabilir” dediği bölgeler üretir. Mask R-CNN’de bunu genelde RPN (Region Proposal Network) yapar. Çıkış: çok sayıda proposal (aday kutu).
    - Stage 2 (Sınıflandırma + ince ayar): Üretilen proposal’lar üzerinden daha detaylı işlem yapılır:
        - proposal’lar feature map’ten RoIAlign ile kırpılıp sabit boyuta getirilir,
        - bbox head ile sınıf tahmini + kutu koordinatlarını refine eder,
        - Mask R-CNN’e özel olarak ayrıca mask head her RoI için piksel seviyesinde maske üretir.

- Mask R-CNN’in “Mask” kısmı

Faster R-CNN’e ek olarak ikinci aşamada bir de mask branch vardır.
Bu sayede aynı RoI’den hem bbox hem mask çıkışı alınır.

- Artı: Genelde tek-aşamalılara göre daha yüksek doğruluk (özellikle zor sahnelerde).
- Eksi: Proposal + RoI işlemleri yüzünden daha yavaş ve daha ağırdır.

* 29 buradaki “model yapısı” yorumu, Mask R-CNN’in parçalarının ne işe yaradığını okuyucuya bağlamak için yazılmış:

Backbone (ResNet)
Girdi görüntüsünü alıp çok katmanlı “feature map”’lere çeviren temel CNN omurgasıdır. ResNet burada “özellik çıkarıcı” gibi çalışır: kenar/köşe gibi basitten başlayıp daha soyut nesne özelliklerine kadar temsil üretir.

Neck (FPN – Feature Pyramid Network)
ResNet farklı çözünürlüklerde özellik haritaları üretir (erken katmanlar daha yüksek çözünürlük, geç katmanlar daha düşük çözünürlük ama daha semantik). FPN bu katmanları birleştirip çok ölçekli (multi-scale) bir piramit oluşturur.
Neden önemli? Küçük nesne-büyük nesne gibi farklı boyutlardaki objeleri daha iyi yakalamak için.

RPN Head (Region Proposal Network)
FPN’den gelen feature map’ler üzerinde “burada nesne olabilir” dediği bölgeler için aday kutular (proposals) üretir.
İki temel çıktı verir:

Objectness skoru (bu kutu nesne mi arka plan mı?)
BBox regression (kutuyu daha iyi oturtmak için düzeltme)
RoI Head (Stage-2 başlık)
RPN’in ürettiği proposal’lar ikinci aşamaya gelir. Burada önce RoIAlign ile her proposal bölgesinin feature’ı sabit boyuta kırpılıp çıkarılır (RoIPooling’e göre daha hassas hizalama).
Sonra iki ayrı “dal” çalışır:

Box head: proposal’ı sınıflandırır (hangi sınıf?) ve bbox’u daha da refine eder.
Mask head: aynı proposal için piksel seviyesinde maske tahmini üretir (instance segmentation kısmı).

Özetle cümle şunu demek istiyor: “Modelin çıktıları (bbox + mask) tesadüfen değil; ResNet→FPN ile güçlü özellik çıkarıp, RPN ile aday bölgeler bulup, RoI Head içinde kutu + maske dallarıyla sonucu üretiyor.”

* 30 init_detector ve inference_detector methodları ile inference yapıyoruz
- C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmdetection\demo\demo.jpg
- Düşük/orta seviye, “tek iş” API: Verilen model + image ile forward + postprocess yapıp sonucu döndürür.
Modeli sen kurarsın: genelde önce init_detector(config, checkpoint, device) çağırıp model alırsın, sonra inference_detector(model, img) dersin.

* 31 mask r cnn output
C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\mask_output.png

* 32 custom bir dataset ile yeni bir detector train etmek
- 1 dataset i uyumlu hale getir
- 2 config dosyalarını revize et
- train et

There are three ways to support a new dataset in MMDetection:
  1. Reorganize the dataset into a COCO format
  2. Reorganize the dataset into a middle format
  3. Implement a new dataset

ilk 2si öneriliyor
mmdetection coco formatı öneriyor, implement etmesi daha kolay

* 33 balloon dataseti kullanacağız
- C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\balloon.png
- convert to VIA (VGG Image Annotator) format to coco format
- C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\coco.png

* 34 ve
* 35 config ayarları

```python
# Modify dataset classes and color
cfg.metainfo = {
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}

# Modify dataset type and path
cfg.data_root = './data/balloon'

cfg.train_dataloader.dataset.ann_file = 'train/annotation_coco.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'train/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'val/annotation_coco.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'val/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root+'/'+'val/annotation_coco.json'
cfg.test_evaluator = cfg.val_evaluator

# Modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# We can still the pre-trained Mask RCNN model to obtain a higher performance
cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'


# We can set the evaluation interval to reduce the evaluation times
cfg.train_cfg.val_interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 3

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optim_wrapper.optimizer.lr = 0.02 / 8
cfg.default_hooks.logger.interval = 10


# Set seed thus the results are more reproducible
# cfg.seed = 0
set_random_seed(0, deterministic=False)

# We can also use tensorboard to log the training process
cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})
```

* 36 runner ile training

mmengine.runner.Runner ile training — artıları

Tek çatı altında eğitim orkestrasyonu: train/val/test loop’larını, epoch/iter kontrolünü ve val_interval gibi akışı config’ten yönetir; kod tarafında “training script” yazma ihtiyacı azalır.
Config-first yaklaşım: Optimizer (optim_wrapper), scheduler (param_scheduler), dataloader, hook’lar vb. her şey config üzerinden yönetildiği için deneyi tekrarlamak/versiyonlamak kolaylaşır.
Hook sistemiyle genişletilebilirlik: Logging, checkpoint, evaluation, visualization gibi işleri “hook” olarak standart bir şekilde tak-çıkar yaparsın; eğitim kodun sade kalır.
Checkpoint & resume kolaylığı: default_hooks.checkpoint + resume/load_from ile kesintiden devam, en iyi modeli saklama, periyodik kayıt gibi pratikler hazır gelir.
Dağıtık eğitim uyumu: Aynı Runner tasarımı single GPU/CPU’dan distributed ortamlara daha doğal taşınır (altyapı MMEngine tarafında).
Standart log/metric akışı: LogProcessor/LoggerHook ile metriklerin düzenli toplanması ve raporlanması daha tutarlı olur.

* 37 mmdetection sonu, neler öğrendik

* 38 mmsegmentation giriş ve installation
```bash
pip install "mmsegmentation>=1.0.0"
```

* 39 init_model inference_model kullanarak demo
- pspnet_r50 modeli kullandık
- input: C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmsegmentation\demo\demo.png
- output: C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmsegmentation\outputs\result.jpg

* 40 video işleme
- input: C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmsegmentation\demo\demo.mp4
- output: C:\Users\eraye\Desktop\Eray\ee563\04_openmmlab\mmsegmentation\outputs\video_result.mp4

* 41 config

1) Config = “deneyin tarifi”
MMSeg’de model, dataset, eğitim schedule’ı ve runtime ayarlarının tamamı bir config dosyasında toplanır.
Aynı kodu değiştirmeden farklı deneyleri sadece config değiştirerek çalıştırırsın.
2) Modüler yapı + kalıtım (inheritance)
Config’ler genelde parça parça gelir ve bir ana config bunları “miras alır”:

models/ → mimari (backbone, decode_head, loss, num_classes…)
datasets/ → veri yolu, pipeline, augmentation
schedules/ → optimizer + LR scheduler + max_iters
default_runtime/ → log, checkpoint, seed, env
Örnek mantık:

“PSPNet + Cityscapes + 40k schedule + default runtime” = tek config.

* 42 CONFIG
* 43 CONFIG

* 44 segmentation çeşitleri

Semantic segmentation: Her piksele sınıf etiketi (aynı sınıftaki tüm nesneler birleşik).
Instance segmentation: Her nesneyi ayrı maske olarak ayırır (aynı sınıfta birden çok obje ayrı).
Panoptic segmentation: Semantic + instance birlikte (things = instance, stuff = semantic).

* 45 finetune a semantic segmentation model on a new dataset
- 1 dataset indir ve uygun hale getir
- 2 config ayarları
- 3 training

* 46 stanford background datasetini kullandık
- C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\stanford.png
- C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\stanford2.png
- class ve palette belirledik
```python
classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
```
* 47 yeni bir class tanımlamak?
- bu tutorial kendi datasetini nasıl implement edersin diye yola çıktığı için böyle bir örnek vermiş
- Kendi dataset sınıfın, ama işi kolay olsun diye MMSeg’in hazır temel sınıfından türetiyorsun. 
- Config’ten çağrılabilir bir dataset sınıfı tanımlıyorsun; sınıf isimleri ve renk paletini de meta bilgi olarak ekliyorsun.

* 48 ve
* 49 CONFIG ÖRNEĞİ

```python
# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.crop_size = (256, 256)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 8
cfg.model.auxiliary_head.num_classes = 8

# Modify dataset type and path
cfg.dataset_type = 'StanfordBackgroundDataset'
cfg.data_root = data_root

cfg.train_dataloader.batch_size = 8

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 240), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader

# Çoklu işlem (multiprocessing) hatasını önlemek için worker sayısını 0 yapıyoruz
cfg.train_dataloader.num_workers = 0
cfg.val_dataloader.num_workers = 0
cfg.test_dataloader.num_workers = 0

# num_workers=0 olduğunda persistent_workers=False olmak zorundadır
cfg.train_dataloader.persistent_workers = False
cfg.val_dataloader.persistent_workers = False
cfg.test_dataloader.persistent_workers = False


# Load the pretrained weights
cfg.load_from = 'models/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/tutorial'

cfg.train_cfg.max_iters = 200
cfg.train_cfg.val_interval = 200
cfg.default_hooks.logger.interval = 10
cfg.default_hooks.checkpoint.interval = 200

# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
```

* 50 runner ile train. ve inference örneği
- input: C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\stanford3.png
- output: C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\stanford4.png

* 51 mmsegmentation bitti, neler öğrendik

* 52 mediapipe
- mediapipe nedir?
- 3 örnek: face-detection, face-landmark, pose-landmark

* 53 face-detection 1
Slayt 1 — MediaPipe Face Detection: Kurulum + Pipeline Mantığı
Amaç: Görüntü/video frame’lerinde yüz(ler)i tespit etmek
Çıktı: bbox + 6 landmark (gözler, burun ucu, ağız, tragion noktaları) + confidence
Model: BlazeFace (detector.tflite)
Notebook’ta linkten indiriliyor
Akış (yüksek seviye):
Model dosyasını indir
FaceDetector’ı modelle başlat
Görüntüyü MediaPipe formatında yükle
detect() ile inference
Sonucu bbox + keypoint olarak çiz
- input: C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\face_detection1.png

* 54 face-detection 2
Slayt 2 — Kod Adımları: Detect + Görselleştirme
1) Detector oluşturma
BaseOptions(model_asset_path='detector.tflite')
FaceDetectorOptions(...)
FaceDetector.create_from_options(options)
2) Görüntüyü yükleme ve inference
image = mp.Image.create_from_file(IMAGE_FILE)
detection_result = detector.detect(image)
3) Sonucu çizdirme (visualize fonksiyonu)
detection_result.detections üzerinde döngü
Her detection için:
bounding_box → cv2.rectangle(...)
keypoints (normalize [0,1]) → piksele çevir → cv2.circle(...)
score/label → cv2.putText(...)
Not: image.numpy_view() zaten RGB → matplotlib ile direkt gösteriliyor
4) Elde edilen çıktı
Konsolda print(detection_result) ile tüm bbox/landmark/score bilgileri görülebilir
output: C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\face_detection2.png

* 55 face_landmark 1
Slayt 1 — MediaPipe Face Landmarker: Ne yapıyor, ne üretir?
Amaç: Yüzü bulup face mesh (478 landmark) çıkarmak (2D/3D landmark koordinatları).
Ek çıktılar (opsiyonel):
Blendshapes (52 skor): mimik/ifade katsayıları (örn. smile, eyeBlink vb.)
Facial transformation matrices: efekti/3D yüz modelini doğru hizalamak için dönüşüm matrisleri
Model paketi (face_landmarker.task): içerde birden fazla model var
önce face detection, sonra landmark/mesh, sonra blendshape aşaması.
- input: C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\face_landmark1.png

* 56 face_landmark 2
1) Landmarker’ı başlatma
BaseOptions(model_asset_path='face_landmarker.task')
FaceLandmarkerOptions(..., num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
FaceLandmarker.create_from_options(options)
2) Görüntü yükle + inference
image = mp.Image.create_from_file("image.png")
detection_result = detector.detect(image)
3) Görselleştirme (mesh çizimi)
draw_landmarks_on_image(...) içinde drawing_utils.draw_landmarks ile:
Tesselation (mesh üçgen ağı)
Contours (yüz hatları)
Left/Right iris bağlantıları çizilir
4) Sonuçları okumak
detection_result.face_landmarks → landmark listeleri
detection_result.face_blendshapes[0] → bar plot ile ifade skorları
detection_result.facial_transformation_matrixes → 3D hizalama için matris çıktısı
- output: C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\face_landmark2.png

* 57 pose landmark 1
Slayt 1 — Pose Landmarker: Ne yapar, ne üretir?
Amaç: Görüntü/video’da insan pozunu tespit edip 33 vücut landmark’ı çıkarmak.
Çıktılar:
pose_landmarks: görüntüye göre normalize (0–1) koordinatlar
(opsiyonel) pose_world_landmarks: 3D world koordinatları
(opsiyonel) segmentation_masks: kişi silueti için pose mask
Model paketi: pose_landmarker_heavy.task (detector + landmarker pipeline’ı tek dosyada)

- input: 
C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\pose_landmark1.png


* 58 pose landmark 2
Slayt 2 — Kod akışı: Kurulum → Detect → Çiz → Maskeyi görselleştir
1) Nesneyi oluşturma
BaseOptions(model_asset_path='pose_landmarker_heavy.task')
PoseLandmarkerOptions(output_segmentation_masks=True)
PoseLandmarker.create_from_options(options)
2) Inference
image = mp.Image.create_from_file("image1.jpg")
detection_result = detector.detect(image)
3) Landmark çizimi
draw_landmarks_on_image(image.numpy_view(), detection_result)
drawing_utils.draw_landmarks(..., connections=POSE_LANDMARKS) ile skeleton bağlantıları
4) Segmentation mask hazırlama
segmentation_mask = detection_result.segmentation_masks[0].numpy_view() (float 0–1)
np.squeeze ile (H,W) yap
*255 + astype(np.uint8) ile görüntülenebilir maske
np.stack([mask]*3, axis=-1) ile 3 kanal (matplotlib/cv2 için)

- output: C:\Users\eraye\Desktop\Eray\ee563\Presentation\images\pose_landmark2.png

* 59 mediapipe sonu, ne öğrendik

* 60 kapanış