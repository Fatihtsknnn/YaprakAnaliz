# -*- coding: utf-8 -*-
"""
Toplu yaprak analizi (ekransiz / Docker uyumlu).

Bir klasordeki tum yaprak goruntulerini U-Net segmentasyon ve pseudo-NIR
modellerinden gecirir, her goruntu icin RGB / segmentasyon / NIR panelini
PNG olarak kaydeder ve hastalik yuzdelerini ozet olarak yazdirir.

Kullanim:
    python analiz.py [--girdi KLASOR] [--cikti KLASOR]
"""
import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")  # Ekran olmadan (container icinde) cizim yapabilmek icin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 256
ESIK = 0.5
UZANTILAR = ('.jpg', '.jpeg', '.png')


def goruntu_oku(yol):
    with open(yol, "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)


def analiz_et(girdi, cikti):
    seg_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "yaprak_analiz_modeli.h5"), compile=False)
    nir_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "yaprak_nir_modeli.h5"), compile=False)
    print("Modeller basariyla yuklendi!")

    os.makedirs(cikti, exist_ok=True)
    dosyalar = sorted(f for f in os.listdir(girdi) if f.lower().endswith(UZANTILAR))
    if not dosyalar:
        print(f"Hata: {girdi} klasorunde goruntu bulunamadi.")
        return

    sonuclar = []
    for dosya in dosyalar:
        img = goruntu_oku(os.path.join(girdi, dosya))
        if img is None:
            print(f"  [atlandi] {dosya} okunamadi")
            continue

        img_res = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_input = np.expand_dims(img_res / 255.0, axis=0)

        mask = (seg_model.predict(img_input, verbose=0)[0] > ESIK).astype(np.uint8)
        pseudo_nir = nir_model.predict(img_input, verbose=0)[0]
        hastalik_yuzdesi = mask.mean() * 100

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.title("Orijinal RGB")
        plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Segmentasyon\nHastalik Siddeti: %{:.2f}".format(hastalik_yuzdesi))
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Tahmini Pseudo-NIR\n(Bitki Saglik Haritasi)")
        plt.imshow(pseudo_nir.squeeze(), cmap='inferno')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(cikti, os.path.splitext(dosya)[0] + "_analiz.png"))
        plt.close()

        sonuclar.append((dosya, hastalik_yuzdesi))
        print(f"  {dosya:<70} Hastalik Alani: %{hastalik_yuzdesi:6.2f}")

    with open(os.path.join(cikti, "sonuclar.csv"), "w", newline="", encoding="utf-8") as f:
        yazici = csv.writer(f)
        yazici.writerow(["dosya", "hastalik_yuzdesi"])
        yazici.writerows((d, f"{y:.2f}") for d, y in sonuclar)

    print("-" * 30)
    print(f"{len(sonuclar)} goruntu analiz edildi, sonuclar: {cikti}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yaprak hastalik analizi")
    parser.add_argument("--girdi", default=os.path.join(BASE_DIR, "test_verileri", "hastalikli_ornekler"),
                        help="Analiz edilecek goruntulerin klasoru")
    parser.add_argument("--cikti", default=os.path.join(BASE_DIR, "sonuclar"),
                        help="Sonuc PNG'lerinin ve CSV'nin yazilacagi klasor")
    args = parser.parse_args()
    analiz_et(args.girdi, args.cikti)
