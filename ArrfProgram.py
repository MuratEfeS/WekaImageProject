import os
import cv2
import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# --- AYARLAR ---
DATASET_PATH = r"C:\Users\murat\Desktop\MuratEfeSahin230404056Proje\BrainTumor" 

def extract_glcm_features(image):
    # GLCM özelliklerini 0, 45, 90 ve 135 derece yönleri için hesaplar 
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    distances = [1]
    glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)
    
    # PDF'de istenen 6 ana doku özelliği 
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    # Farklı yönlerden elde edilen özelliklerin ortalamasını alarak tek bir vektör oluşturur 
    features = [np.mean(graycoprops(glcm, p)) for p in props]
    return features

def extract_lbp_features(image):
    # Local Binary Pattern (LBP) (Yerel doku deseni) özelliklerini çıkarır 
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist.tolist()

def extract_hog_features(image):
    # Histogram of Oriented Gradients (HOG) (kenar ve şekil) özelliklerini çıkarır 
    # Özellik sayısını optimize etmek için 32x32 boyutlandırma
    resized_img = cv2.resize(image, (32, 32)) 
    fd = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False)
    return fd.tolist()

def save_arff(filename, feature_names, data_list, classes):
    # Özellikleri Weka tarafından okunabilir .arff formatında kaydeder 
    with open(filename, 'w') as f:
        f.write("@RELATION image_processing_final\n\n")
        for name in feature_names:
            f.write(f"@ATTRIBUTE {name} NUMERIC\n")
        class_str = ",".join(classes)
        f.write(f"@ATTRIBUTE class {{{class_str}}}\n\n") 
        f.write("@DATA\n")
        for row in data_list:
            line = ",".join([str(v) for v in row])
            f.write(line + "\n")

# --- ANA DÖNGÜ ---
CLASSES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
glcm_data, glcm_lbp_data, glcm_lbp_hog_data, wavelet_data = [], [], [], []

print(f"Toplam {len(CLASSES)} sınıf için işlem başlatıldı...")

# Tek tek resimleri döngü ile işliyoruz
for class_name in CLASSES:
    class_dir = os.path.join(DATASET_PATH, class_name)
    image_names = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"> {class_name} sınıfı işleniyor ({len(image_names)} görüntü)...")
    
    for img_name in image_names:
        img_path = os.path.join(class_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Görüntüyü gri tonlamaya dönüştürme 
        if image is None: continue

        # 1. Standart Özellikler (GLCM, LBP, HOG) 
        f_glcm = extract_glcm_features(image)
        f_lbp = extract_lbp_features(image)
        f_hog = extract_hog_features(image)
        
        # 2. Bonus: Wavelet Transform (LL alt bandı üzerinden GLCM)
        coeffs2 = pywt.dwt2(image.astype(float), 'bior1.3')
        LL, _ = coeffs2
        LL_norm = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        f_wavelet_glcm = extract_glcm_features(LL_norm)

        # Veri Birleştirme (Feature Fusion) 
        glcm_data.append(f_glcm + [class_name])
        glcm_lbp_data.append(f_glcm + f_lbp + [class_name])
        glcm_lbp_hog_data.append(f_glcm + f_lbp + f_hog + [class_name])
        wavelet_data.append(f_wavelet_glcm + [class_name])

# --- KAYIT İŞLEMLERİ ---
#Değerlerin İsimlendirmelerini Yapıyoruz.
glcm_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'] 
lbp_names = [f'LBP_Bin_{i}' for i in range(10)] 
hog_names = [f'HOG_{i+1}' for i in range(len(glcm_lbp_hog_data[0]) - 17)] 
wavelet_names = [f'Wavelet_{n}' for n in glcm_names]

print("ARFF dosyaları oluşturuluyor...")
save_arff('glcm.arff', glcm_names, glcm_data, CLASSES) 
save_arff('glcm_lbp.arff', glcm_names + lbp_names, glcm_lbp_data, CLASSES) 
save_arff('glcm_lbp_hog.arff', glcm_names + lbp_names + hog_names, glcm_lbp_hog_data, CLASSES) 
save_arff('wavelet_glcm.arff', wavelet_names, wavelet_data, CLASSES) 

print("İşlem tamamlandı.")