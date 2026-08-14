import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf():
    os.makedirs("outputs", exist_ok=True)
    pdf_path = os.path.join("outputs", "Conformal_Risk_Control_Rehberi.pdf")
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1A365D'), spaceAfter=4)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#4A5568'), spaceAfter=8)
    h1_style = ParagraphStyle('H1', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor('#2B6CB0'), spaceBefore=10, spaceAfter=4, keepWithNext=True)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#2D3748'), spaceAfter=5)
    formula_style = ParagraphStyle('Formula', parent=styles['Code'], fontSize=8.5, leading=12, textColor=colors.HexColor('#1A365D'), backColor=colors.HexColor('#EBF8FF'), borderPadding=6, spaceAfter=6, spaceBefore=4)
    box_style = ParagraphStyle('Callout', parent=styles['Normal'], fontSize=8.5, leading=12, textColor=colors.HexColor('#22543D'), backColor=colors.HexColor('#F0FFF4'), borderPadding=6, spaceAfter=6)

    story = []
    story.append(Paragraph("Conformal Risk Control (Uyumlu Risk Kontrolü) Rehberi", title_style))
    story.append(Paragraph("<b>Referans:</b> Angelopoulos, Bates, Fisch, Lei & Schuster (2022) | Kapsamlı Türkçe Rehber", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceAfter=8))

    story.append(Paragraph("1. Giriş: Geleneksel Yöntemden Risk Kontrolüne Geçiş", h1_style))
    story.append(Paragraph("Standart Conformal Prediction ikili (0/1) hata mantığıyla çalışır. Conformal Risk Control ise hatayı sürekli bir kayıp olarak ölçer ve ortalama kaybın <b>E[Loss] &le; &alpha;</b> kalmasını matematiksel olarak garantiler.", body_style))
    story.append(Paragraph("<b>💡 Örnek:</b> 100 mm³ tümörün 98 mm³'ü bulunup 2 mm³'ü kaçırılsa dahi eski yöntem bunu hatalı (1) sayardı. Conformal Risk Control hatayı kaçırılan doku oranı (Loss = 0.02) olarak kabul eder.", box_style))

    story.append(Paragraph("2. Matematiksel Kalibrasyon Formülü", h1_style))
    calib_formula = "<b>Kalibrasyon Eşiği Formülü:</b><br/>&lambda;^ = inf { &lambda; : [n / (n + 1)] * R_n(&lambda;) + [B / (n + 1)] &le; &alpha; }<br/><b>Risk Garantisi:</b> E[ L_{n+1}(&lambda;^) ] &le; &alpha;"
    story.append(Paragraph(calib_formula, formula_style))

    story.append(Paragraph("3. Makaledeki 4 Büyük Uygulama Alanı", h1_style))
    data = [
        [Paragraph('<b>Uygulama Alanı</b>', body_style), Paragraph('<b>Kayıp (Loss)</b>', body_style), Paragraph('<b>Hedef (&alpha;)</b>', body_style), Paragraph('<b>Sonuç</b>', body_style)],
        [Paragraph('<b>1. Tümör Segmentasyonu</b>', body_style), Paragraph('FNR (Kaçırılan Doku)', body_style), Paragraph('&alpha; = 0.10', body_style), Paragraph('Ampirik Risk = <b>%9.87</b>', body_style)],
        [Paragraph('<b>2. Çoklu Nesne Tespiti</b>', body_style), Paragraph('Kaçırılan Nesne Oranı', body_style), Paragraph('&alpha; = 0.10', body_style), Paragraph('Ampirik Risk = <b>%9.96</b>', body_style)],
        [Paragraph('<b>3. Hiyerarşik Ağaç</b>', body_style), Paragraph('Ağaç Mesafesi', body_style), Paragraph('&alpha; = 0.05', body_style), Paragraph('Ampirik Risk = <b>0.0499</b>', body_style)],
        [Paragraph('<b>4. Soru-Cevap (NLP)</b>', body_style), Paragraph('1 - Token F1-Score', body_style), Paragraph('&alpha; = 0.30', body_style), Paragraph('Ampirik Risk = <b>0.2996</b>', body_style)]
    ]
    t = Table(data, colWidths=[130, 150, 95, 165])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    doc.build(story)
    print(f"✅ PDF Başarıyla Oluşturuldu: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
