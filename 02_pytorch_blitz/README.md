# 02 - PyTorch Blitz ve Örnekler

Bu klasör, PyTorch'un resmi dökümantasyonunda yer alan iki temel eğitimi kapsamaktadır: "A 60 Minute Blitz" ve "Learning PyTorch with Examples".

## 1. Deep Learning with PyTorch: A 60 Minute Blitz
PyTorch'u hızlı ve etkili bir şekilde öğrenmek için hazırlanan bu seri, 4 ana bölümden oluşmaktadır.
**Ana Eğitim Linki:** [Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

| Bölüm | Konu | Durum / Link |
| :--- | :--- | :--- |
| **1. Tensors** | Tensorların yapısı ve kullanımı | [01 - Tensors (Daha önce işlendi)](../01_getting_started/1_tensors.ipynb) <br> [Resmi Link](https://docs.pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) |
| **2. Autograd** | Otomatik türev alma (Automatic Differentiation) | [01 - Autograd (Daha önce işlendi)](../01_getting_started/5_autograds.ipynb) <br> [Resmi Link](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) |
| **3. Neural Networks** | `nn` modülü ile sinir ağı oluşturma | [Neural Networks Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) |
| **4. Training a Classifier** | CIFAR10 veri seti ile sınıflandırıcı eğitimi | [Training a Classifier Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) |

## 2. Learning PyTorch with Examples
Bu eğitim serisi, tek bir akış üzerinden PyTorch'un temel yapıtaşlarını (Tensors, Autograd, nn, optim) örneklerle anlatır. Numpy ile başlayıp PyTorch'un modern yeteneklerine evrilen bir yapı sunar.
**Ana Eğitim Linki:** [Learning PyTorch with Examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)

Bu bölüm alt başlıklar halinde ayrılmamıştır, tek bir sayfa üzerinde aşağıdaki kavramları sırasıyla işler:
*   Mevcut bir problemi Numpy ile çözme
*   Aynı problemi PyTorch Tensorları ile çözme
*   Autograd kullanarak türev hesabı
*   `nn` modülü ile model tanımlama
*   `optim` modülü ile optimizasyon
*   Özel `nn` modülleri tanımlama
