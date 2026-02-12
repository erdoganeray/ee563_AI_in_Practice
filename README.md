# EE563 - Artificial Intelligence in Practice

Bu repo, **EE563 - Artificial Intelligence in Practice** dersi kapsamında işlenen konuların kodlarını ve uygulamalarını içermektedir.

## Klasör Yapısı

```text
.
├── 01_getting_started/     # PyTorch'a Giriş
├── 02_pytorch_blitz/       # PyTorch Blitz ve Örneklerle PyTorch
├── 04_openmmlab/           # OpenMMLab (MMDetection & MMSegmentation)
├── 05_mediapipe/           # MediaPipe Uygulamaları
└── README.md               # Proje açıklaması
```

## Ders İçeriği

Repo, ders programına uygun olarak aşağıdaki haftaları kapsamaktadır:

*   **01_getting_started**: PyTorch'a Giriş (Getting Started with PyTorch)
    *   [0. Quickstart](./01_getting_started/0_quickstart.ipynb)
    *   [1. Tensors](./01_getting_started/1_tensors.ipynb)
    *   [2. Datasets & DataLoaders](./01_getting_started/2_datasets_and_dataloaders.ipynb)
    *   [3. Transforms](./01_getting_started/3_transforms.ipynb)
    *   [4. Build Model](./01_getting_started/4_build_model.ipynb)
    *   [5. Autograd](./01_getting_started/5_autograds.ipynb)
    *   [6. Optimization](./01_getting_started/6_optimization.ipynb)
    *   [7. Save & Load Model](./01_getting_started/7_saveloadrun.ipynb)

*   **02_pytorch_blitz**: PyTorch Blitz ve Örneklerle PyTorch
    *   **PyTorch: A 60 Minute Blitz** [Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
        *   [Autograd (Referans: 01)](./01_getting_started/5_autograds.ipynb)
        *   [Transforms (Referans: 01)](./01_getting_started/3_transforms.ipynb)
        *   [Neural Networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
        *   [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    *   **Learning PyTorch with Examples**
        *   [PyTorch with Examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)

*   **04_openmmlab**: Bilgisayarlı Görü Uygulamaları (OpenMMLab)
    *   **MMDetection**
        *   [0. Overview & Inference](./04_openmmlab/mmdetection/0_overview_and_getstarted.ipynb): MMDetection kurulumu, `DetInferencer` kullanımı, Batch/URL inference ve Video testi.
    *   MMSegmentation ile Segmentasyon [MMSegmentation GitHub](https://github.com/open-mmlab/mmsegmentation)

*   **05_mediapipe**: Bilgisayarlı Görü için Bir Başka Platform: MediaPipe
    *   Yüz tespiti (Face detector) [Face Detector Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python)
    *   Poz kestirimi (Pose estimation) [Pose Landmarker Docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)

## Kaynaklar/Syllabus

*   **Ders Kodu:** EE 563
*   **Ders Adı:** Artificial Intelligence (AI) in Practice
*   **Eğitmen:** Cihan Göksu, PhD.
