# WekaImageProject: Görüntü İşleme ve Özellik Çıkarımı

Bu proje, **Brain Tumor** veri seti üzerinde görüntü işleme teknikleri kullanarak özellik çıkarımı yapar ve elde edilen verileri Weka (Machine Learning) yazılımında kullanılmak üzere `.arff` formatına dönüştürür.

Proje, Python kullanılarak geliştirilmiş olup aşağıdaki özellik çıkarım yöntemlerini içermektedir:
* **GLCM** (Gray Level Co-occurrence Matrix)
* **LBP** (Local Binary Patterns)
* **HOG** (Histogram of Oriented Gradients)
* **Wavelet Transform**

`ArrfProgram.py` dosyası çalıştırıldığında, görüntüler işlenir ve sınıflandırma algoritmalarında kullanılmak üzere yapılandırılmış veri setleri oluşturulur.

---
Veri Seti olarak Kaggle sitesinden Mark Otto ve Andrew Fong'un MIT lisansladığı Beyin Tümörleri kullandım.

