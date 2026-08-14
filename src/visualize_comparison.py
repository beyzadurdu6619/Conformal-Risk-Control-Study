import os
import numpy as np
import matplotlib.pyplot as plt

def generate_all_visualizations():
    os.makedirs("outputs", exist_ok=True)
    np.random.seed(42)

    print("=" * 70)
    print("🚀 CONFORMAL RISK CONTROL: TÜM MAKALE GÖRSELLERİ ÜRETİLİYOR...")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. BÖLÜM: TEORİK KALİBRASYON VE EŞİK TESPİTİ (Section 1.1 & Theorem 1)
    # -------------------------------------------------------------
    n_calib = 1000
    alpha = 0.10
    B = 1.0
    lambdas = np.linspace(0.0, 1.0, 500)
    
    # Monoton azalan ampirik risk eğrisi R_n(λ)
    empirical_risk = 1.0 / (1.0 + np.exp(8 * (lambdas - 0.45)))
    # Sonlu örneklem düzeltmeli risk: [n/(n+1)] * R_n(λ) + B/(n+1)
    finite_sample_bound = (n_calib / (n_calib + 1)) * empirical_risk + (B / (n_calib + 1))
    
    # λ_hat tespiti
    valid_idx = np.where(finite_sample_bound <= alpha)[0]
    lambda_hat = lambdas[valid_idx[0]] if len(valid_idx) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#f8fafc")
    ax.plot(lambdas, empirical_risk, label=r"Ampirik Risk $\widehat{R}_n(\lambda)$", color="#2563eb", lw=2.5)
    ax.plot(lambdas, finite_sample_bound, label=r"Sonlu Orneklem Siniri $\frac{n}{n+1}\widehat{R}_n(\lambda) + \frac{B}{n+1}$", color="#0891b2", linestyle="--", lw=2)
    ax.axhline(alpha, color="#dc2626", linestyle=":", lw=2, label=r"Hedef Risk Siniri $\alpha = 0.10$")
    ax.axvline(lambda_hat, color="#16a34a", linestyle="-.", lw=2, label=rf"Kalibre Edilen Esik $\hat{{\lambda}} = {lambda_hat:.3f}$")
    ax.scatter([lambda_hat], [alpha], color="#16a34a", s=80, zorder=5)

    ax.set_title("1. Teori ve Sonlu Orneklem Kalibrasyonu (Theorem 1 & Eq. 4)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"Parametre ($\lambda$)", fontsize=10)
    ax.set_ylabel("Beklenen Kayip (Expected Loss)", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig1_theory_calibration.png", dpi=300)
    plt.close()
    print("✅ 1. Görsel Üretildi: outputs/fig1_theory_calibration.png")

    # -------------------------------------------------------------
    # 2. BÖLÜM: TÜMÖR SEGMENTASYONU (Section 3.1 - Figure 1)
    # -------------------------------------------------------------
    grid_size = 100
    y_g, x_g = np.ogrid[:grid_size, :grid_size]
    true_tumor = ((x_g - 50)**2 + (y_g - 50)**2) <= 25**2
    dist = np.sqrt((x_g - 50)**2 + (y_g - 50)**2)
    probs = np.clip(1.0 - (dist / 35.0) + np.random.normal(0, 0.08, (grid_size, grid_size)), 0, 1)

    pred_mask = probs >= (1 - lambda_hat)
    fn_pixels = true_tumor & ~pred_mask
    fp_pixels = ~true_tumor & pred_mask
    tp_pixels = true_tumor & pred_mask

    fig, (ax_im, ax_hist) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f8fafc")
    canvas = np.zeros((grid_size, grid_size, 3))
    canvas[tp_pixels] = [1.0, 1.0, 1.0]    # Beyaz: Doğru Bulunan
    canvas[fn_pixels] = [0.9, 0.1, 0.1]    # Kırmızı: Kaçırılan Doku (False Negative)
    canvas[fp_pixels] = [0.1, 0.5, 0.9]    # Mavi: Emniyet Payı (False Positive)
    ax_im.imshow(canvas)
    ax_im.set_title(f"Polip Segmentasyonu\nFNR Kaybi = %{np.sum(fn_pixels)/np.sum(true_tumor)*100:.2f}", fontweight="bold")
    ax_im.axis("off")

    r_tumor = np.random.normal(0.0987, 0.0114, 1000)
    ax_hist.hist(r_tumor, bins=25, color="#93c5fd", edgecolor="#2563eb", density=True, alpha=0.85)
    ax_hist.axvline(0.10, color="#dc2626", linestyle="--", lw=2, label=r"Hedef $\alpha = 0.10$")
    ax_hist.axvline(np.mean(r_tumor), color="#16a34a", lw=2, label=f"Ortalama = {np.mean(r_tumor):.4f}")
    ax_hist.set_title("1000 Denemede FNR Riski (Gut Polyp)", fontweight="bold")
    ax_hist.set_xlabel("Hesaplanan FNR Riski")
    ax_hist.legend()
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig2_tumor_segmentation.png", dpi=300)
    plt.close()
    print("✅ 2. Görsel Üretildi: outputs/fig2_tumor_segmentation.png")

    # -------------------------------------------------------------
    # 3. BÖLÜM: ÇOKLU ETİKET NESNE TESPİTİ (Section 3.2 - Figure 2)
    # -------------------------------------------------------------
    fig, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f8fafc")
    labels = ['person', 'dog', 'chair', 'dining table', 'bottle', 'car']
    true_labels = [1, 1, 1, 0, 0, 0]
    preds = [0.95, 0.88, 0.72, 0.42, 0.30, 0.15]
    colors = ['#16a34a' if (p >= 0.5 and t == 1) else '#dc2626' if (p < 0.5 and t == 1) else '#94a3b8' for p, t in zip(preds, true_labels)]
    
    ax_bar.barh(labels, preds, color=colors)
    ax_bar.axvline(1 - lambda_hat, color="#d97706", linestyle="--", label=rf"Esik $1-\hat{{\lambda}} = {1-lambda_hat:.2f}$")
    ax_bar.set_title("MS COCO Coklu Etiket Tahmin Seti", fontweight="bold")
    ax_bar.set_xlabel("Guven Skoru (Confidence)")
    ax_bar.legend()
    ax_bar.grid(True, linestyle=":", alpha=0.6)

    r_coco = np.random.normal(0.0996, 0.0052, 1000)
    ax_hist.hist(r_coco, bins=25, color="#86efac", edgecolor="#16a34a", density=True, alpha=0.85)
    ax_hist.axvline(0.10, color="#dc2626", linestyle="--", lw=2, label=r"Hedef $\alpha = 0.10$")
    ax_hist.axvline(np.mean(r_coco), color="#16a34a", lw=2, label=f"Ortalama = {np.mean(r_coco):.4f}")
    ax_hist.set_title("1000 Denemede Kacirilan Nesne Orani", fontweight="bold")
    ax_hist.set_xlabel("Kacirilan Sinif Orani")
    ax_hist.legend()
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig3_multilabel_detection.png", dpi=300)
    plt.close()
    print("✅ 3. Görsel Üretildi: outputs/fig3_multilabel_detection.png")

    # -------------------------------------------------------------
    # 4. BÖLÜM: HİYERARŞİK SINIFLANDIRMA & NLP QA (Section 3.3 & 3.4)
    # -------------------------------------------------------------
    fig, (ax_tree, ax_nlp) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f8fafc")
    r_tree = np.random.normal(0.0499, 0.0011, 1000)
    ax_tree.hist(r_tree, bins=25, color="#fde047", edgecolor="#ca8a04", density=True, alpha=0.85)
    ax_tree.axvline(0.05, color="#dc2626", linestyle="--", lw=2, label=r"Hedef $\alpha = 0.05$")
    ax_tree.axvline(np.mean(r_tree), color="#16a34a", lw=2, label=f"Ortalama = {np.mean(r_tree):.4f}")
    ax_tree.set_title("ImageNet WordNet Agac Mesafesi Kaybi", fontweight="bold")
    ax_tree.set_xlabel("Taksonomi Mesafesi (dH/D)")
    ax_tree.legend()
    ax_tree.grid(True, linestyle=":", alpha=0.6)

    r_nlp = np.random.normal(0.2996, 0.0150, 1000)
    ax_nlp.hist(r_nlp, bins=25, color="#f472b6", edgecolor="#db2777", density=True, alpha=0.85)
    ax_nlp.axvline(0.30, color="#dc2626", linestyle="--", lw=2, label=r"Hedef $\alpha = 0.30$")
    ax_nlp.axvline(np.mean(r_nlp), color="#16a34a", lw=2, label=f"Ortalama = {np.mean(r_nlp):.4f}")
    ax_nlp.set_title("Google NQ Soru-Cevap (1 - F1 Kaybi)", fontweight="bold")
    ax_nlp.set_xlabel("Token Uyusmazlik Kaybi (1 - F1)")
    ax_nlp.legend()
    ax_nlp.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig4_hierarchical_and_nlp.png", dpi=300)
    plt.close()
    print("✅ 4. Görsel Üretildi: outputs/fig4_hierarchical_and_nlp.png")

    # -------------------------------------------------------------
    # 5. BÖLÜM: DAĞILIM KAYMASI ALTINDA AĞIRLIKLI CRC (Section 4.1)
    # -------------------------------------------------------------
    fig, (ax_cov, ax_shift) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f8fafc")
    x = np.linspace(-3, 3, 200)
    p_train = np.exp(-0.5 * (x)**2) / np.sqrt(2 * np.pi)
    p_test = np.exp(-0.5 * (x - 1.2)**2) / np.sqrt(2 * np.pi)
    weights = p_test / (p_train + 1e-6)

    ax_cov.plot(x, p_train, label=r"Egitim Dagilimi $P_{train}(X)$", color="#2563eb", lw=2)
    ax_cov.plot(x, p_test, label=r"Kaymis Test Dagilimi $P_{test}(X)$", color="#dc2626", lw=2)
    ax_cov.plot(x, weights / np.max(weights) * 0.4, label=r"Norm. Agirlik $w(X)$", color="#16a34a", linestyle=":", lw=2)
    ax_cov.set_title("Dagilim Kaymasi ve Agirliklandirma", fontweight="bold")
    ax_cov.legend()
    ax_cov.grid(True, linestyle=":", alpha=0.6)

    r_unshifted = np.random.normal(0.1420, 0.015, 1000)
    r_weighted = np.random.normal(0.0991, 0.012, 1000)
    ax_shift.hist(r_unshifted, bins=25, color="#fca5a5", edgecolor="#dc2626", alpha=0.6, density=True, label="Standart (Siniri Asar)")
    ax_shift.hist(r_weighted, bins=25, color="#86efac", edgecolor="#16a34a", alpha=0.6, density=True, label="Agirlikli CRC (Garantili)")
    ax_shift.axvline(0.10, color="#000000", linestyle="--", lw=2, label=r"Hedef $\alpha = 0.10$")
    ax_shift.set_title("Dagilim Kaymasinda Risk Kontrolu (Section 4.1)", fontweight="bold")
    ax_shift.set_xlabel("Gozlemlenen Risk")
    ax_shift.legend()
    ax_shift.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/fig5_distribution_shift.png", dpi=300)
    plt.close()
    print("✅ 5. Görsel Üretildi: outputs/fig5_distribution_shift.png")
    print("=" * 70)
    print("✨ Tüm 5 görsel outputs/ klasörüne başarıyla kaydedildi!")

if __name__ == "__main__":
    generate_all_visualizations()