# 🛡️ Conformal Risk Control (Uyumlu Risk Kontrolü): Theory, Benchmarks & Real-World Visual Guide

<p align="center">
  <b>An Academic & Visual Study Suite for the Foundational Paper:</b><br>
  <i>"Conformal Risk Control" by Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster (UC Berkeley, MIT, Stanford, Google Research)</i>
</p>

---

## 🌐 Language Navigation / Dil Seçimi
* 🇹🇷 [Türkçe Açıklamalı ve Örnekli Rehber](#-türkçe-kapsamlı-ve-örnekli-rehber)
* 🇬🇧 [English Detailed Guide & Examples](#-english-detailed-guide--examples)

---

<a name="-türkçe-kapsamlı-ve-örnekli-rehber"></a>
# 🇹🇷 Türkçe Kapsamlı ve Örnekli Rehber

Bu çalışma, standart yapay zeka modellerinin ürettiği tahminlerin güvenilirliğini ve hata payını matematiksel olarak garanti altına alan **Conformal Risk Control (CRC)** algoritmasını tüm yönleriyle açıklar ve görselleştirir.

---

### 1. Temel Mantık: Standart Yöntemden Risk Kontrolüne Geçiş

#### ❌ Geleneksel Yöntemin (Standart Conformal Prediction) Çıkmazı:
Geleneksel Conformal Prediction sadece **ikili (0 veya 1) hata** mantığıyla çalışır. Gerçek etiket tahmin kümesinde yoksa sistem "Tam Hata ($L=1$)" sayar.
* **💡 Gerçek Hayat Örneği:** Bir cerrah $100\text{ mm}^3$ boyutundaki bir tümörün $96\text{ mm}^3$'ünü temizleyip sadece $4\text{ mm}^3$'ünü kaçırsa bile, eski sistem bunu başarısız (1) kabul ederdi. Ancak gerçek hayatta bu %96'lık bir başarıdır!

####  Conformal Risk Control (CRC) Çözümü:
CRC, hatayı **sürekli ve monoton bir kayıp fonksiyonu ($L \in [0, 1]$)** olarak tanımlar ve modelin ortalama kaybını kullanıcı tarafından belirlenen tolerans seviyesinde ($\alpha$) garanti altına alır:

$$\mathbb{E}\left[\ell\left(\mathcal{C}_{\hat{\lambda}}(X_{n+1}), Y_{n+1}\right)\right] \le \alpha$$

#### 📐 Kalibrasyon Eşik Formülü (Equation 4):
$n$ adet kalibrasyon örneği üzerinden modelin aşırı güvenini kıran optimal eşik ($\hat{\lambda}$) şu şekilde hesaplanır:

$$\hat{\lambda} = \inf \left\{ \lambda : \frac{n}{n+1} \widehat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha \right\}$$

![Teorik Kalibrasyon](outputs/fig1_theory_calibration.png)

---

### 2. Makaledeki 4 Ana Uygulama Alanı ve Örnekleri

---

#### 🩺 Bölüm 3.1: Tümör ve Bağırsak Polipi Segmentasyonu (FNR Kontrolü)
* **Problem:** MR/Endoskopi görüntüsünde kaçırılan doku hayati risk taşır.
* **Kayıp Fonksiyonu:** Kaçırılan doku oranı (False Negative Rate):
  
  $$L_i^{\text{FNR}}(\lambda) = 1 - \frac{|Y_i \cap \mathcal{C}_\lambda(X_i)|}{|Y_i|}$$

* **Hedef:** Kaçırılan doku ortalaması en fazla $\%10$ olsun ($\alpha = 0.10$).
* **Elde Edilen Sonuç:** 1000 bağımsız testte ortalama kayıp **%9.87** olarak gerçekleşti.

![Tümör Segmentasyonu](outputs/fig2_tumor_segmentation.png)

---

#### 📷 Bölüm 3.2: Çoklu Etiket Nesne Tespiti (MS COCO)
* **Problem:** Bir fotoğrafta birden fazla nesne (insan, araba, sandalye, masa) varken modelin önemli nesneleri atlamaması gerekir.
* **Kayıp Fonksiyonu:** Kaçırılan etiketlerin gerçek etiket sayısına oranı.
* **Hedef:** Kaçırılan nesne oranı $\le \%10$ olsun ($\alpha = 0.10$).
* **Elde Edilen Sonuç:** 1000 denemede kaçırılan nesne oranı **%9.96**'da tutuldu.

![Nesne Tespiti](outputs/fig3_multilabel_detection.png)

---

#### 🌳 Bölüm 3.3: Hiyerarşik Görüntü Sınıflandırma (ImageNet WordNet)
* **Problem:** Model "Golden Retriever" cinsinden emin değilse rastgele tahmin yapmak yerine ağaç yapısında bir üst kavrama ("Köpek" veya "Memeli Hayvan") çıkarak güvenli tahmin üretmelidir.
* **Kayıp Fonksiyonu:** Taksonomi ağacı mesafesi ($d_H / D$).
* **Hedef:** Hata mesafesi $\le 0.05$ olsun ($\alpha = 0.05$).
* **Elde Edilen Sonuç:** Ortalama mesafe **0.0499** olarak kanıtlandı.

---

#### 💬 Bölüm 3.4: Açık Alan Soru-Cevap (Google Natural Questions - LLM)
* **Problem:** "Barack Obama nerede doğdu?" sorusuna model birden fazla olası cevap kümesi döndürür: `{"Honolulu", "Hawaii", "Kapi'olani Hastanesi"}`.
* **Kayıp Fonksiyonu:** $1 - \text{Token F1-Score}$.
* **Hedef:** Token uyumsuzluk riski $\le 0.30$ olsun ($\alpha = 0.30$).
* **Elde Edilen Sonuç:** 1000 denemede ortalama token kaybı **0.2996** seviyesinde kaldı.

![Hiyerarşik ve NLP](outputs/fig4_hierarchical_and_nlp.png)

---

### 3. Bölüm 4.1: Dağılım Kayması (Distribution Shift) Altında Ağırlıklı Kalibrasyon

* **Problem:** Yapay zeka modeli hastanedeki A cihazı ile kalibre edilip B cihazında (veya farklı hasta grubunda) test edildiğinde veri dağılımı kayar ($P_{\text{test}} \neq P_{\text{train}}$) ve standart modeller çöker.
* **Çözüm (Ağırlıklı CRC):** Olursallık oranı $w(x) = \frac{\text{d}P_{\text{test}}(x)}{\text{d}P_{\text{train}}(x)}$ ile örneklere ağırlık verilerek garanti yeni koşullarda da korunur:

$$\hat{\lambda}(x) = \inf \left\{ \lambda : \frac{\sum_{i=1}^n w(X_i)L_i(\lambda) + w(x)B}{\sum_{i=1}^n w(X_i) + w(x)} \le \alpha \right\}$$

![Dağılım Kayması](outputs/fig5_distribution_shift.png)

---
---

<a name="-english-detailed-guide--examples"></a>
# 🇬🇧 English Detailed Guide & Examples

This repository provides an end-to-end framework and visual benchmark reproducing the statistical risk guarantees introduced in **Conformal Risk Control (Angelopoulos et al., 2022)**.

---

### 1. Core Concept: Beyond Binary Miscoverage

#### ❌ The Limitation of Standard Conformal Prediction:
Standard Conformal Prediction only evaluates 0/1 miscoverage:
$$\mathbb{P}(Y_{n+1} \notin \mathcal{C}(X_{n+1})) \le \alpha$$
* **💡 Real-World Example:** In medical tumor resection, if a surgeon identifies $96\text{ mm}^3$ of a $100\text{ mm}^3$ lesion, missing $4\text{ mm}^3$, standard methods classify the whole prediction as a **complete failure ($L=1$)**. In clinical practice, this is a 96% success.

####  The Conformal Risk Control (CRC) Solution:
CRC extends conformal coverage to **arbitrary bounded monotone loss functions** $\ell(\mathcal{C}_\lambda(X), Y) \in [0, B]$:

$$\mathbb{E}\left[\ell\left(\mathcal{C}_{\hat{\lambda}}(X_{n+1}), Y_{n+1}\right)\right] \le \alpha$$

#### 📐 Finite-Sample Calibration Formula (Equation 4):
Given $n$ calibration instances, the optimal threshold $\hat{\lambda}$ is computed as:

$$\hat{\lambda} = \inf \left\{ \lambda : \frac{n}{n+1} \widehat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha \right\}$$

---

### 2. The 4 Benchmark Applications (Section 3)

| Domain | Dataset | Loss Metric ($\ell$) | Target ($\alpha$) | Empirical Result |
| :--- | :--- | :--- | :--- | :--- |
| **1. Gut Polyp Segmentation** | Kvasir / CVC | False Negative Rate ($\text{FNR}$) | $\alpha = 0.10$ | **%9.87** ($\le 10\%$) |
| **2. Multi-Label Object Detection** | MS COCO | Fraction of missed classes | $\alpha = 0.10$ | **%9.96** ($\le 10\%$) |
| **3. Hierarchical Classification** | ImageNet WordNet | Taxonomy tree distance ($d_H / D$) | $\alpha = 0.05$ | **0.0499** ($\le 0.05$) |
| **4. Open-domain QA (LLMs)** | Google Natural Questions | $1 - \max(\text{Token F1})$ | $\alpha = 0.30$ | **0.2996** ($\le 0.30$) |

---

### 3. Weighted Risk Control Under Distribution Shift (Section 4.1)

When the testing distribution shifts from the training distribution ($P_{\text{test}} \neq P_{\text{train}}$), standard algorithms violate error bounds. By incorporating importance weights $w(x) = \frac{\text{d}P_{\text{test}}(x)}{\text{d}P_{\text{train}}(x)}$, CRC guarantees exact finite-sample coverage in shifted test domains.

---

## 🚀 Quick Start / Hızlı Başlangıç

### 1. Install Dependencies / Bağımlılıkları Yükle
```bash
pip install -r requirements.txt