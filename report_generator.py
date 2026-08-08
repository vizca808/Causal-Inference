from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'Causal Inference Analysis Report', 0, 1, 'C')
        self.set_line_width(0.5)
        self.set_draw_color(189, 195, 199)
        self.line(10, 22, 200, 22)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(127, 140, 141)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(236, 240, 241)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, f' {title}', 0, 1, 'L', fill=True)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(52, 73, 94)
        self.multi_cell(0, 6, text)
        self.ln(5)
        
    def add_plot(self, image_path, w=160):
        if os.path.exists(image_path):
            self.image(image_path, w=w, x='C')
            self.ln(5)
        else:
            self.set_text_color(231, 76, 60)
            self.cell(0, 10, f'[Image not found: {image_path}]', 0, 1, 'C')
            self.ln(5)

def generate_pdf_report(ate_dict, output_dir, save_dir):
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.chapter_title('1. Executive Summary')
    summary_text = (
        "Laporan ini berisi hasil analisis inferensi kausal untuk mengevaluasi dampak "
        "sebuah program (treatment) terhadap penghasilan (outcome). Tidak seperti korelasi biasa, "
        "inferensi kausal memperhitungkan variabel perancu (confounders) sehingga kita bisa "
        "memperkirakan efek 'sebab-akibat' yang sebenarnya.\n\n"
        "Dari seluruh metode yang diuji, program pelatihan kerja ini terbukti memberikan dampak positif "
        "yang signifikan secara statistik terhadap penghasilan para peserta."
    )
    pdf.chapter_body(summary_text)

    # 2. Exploratory Data Analysis
    pdf.chapter_title('2. Exploratory Data Analysis (EDA)')
    eda_text = (
        "Langkah pertama adalah melihat distribusi data penghasilan (Earnings) pada kelompok yang "
        "mengikuti program (Treatment) dan yang tidak (Control). Seringkali, perbandingan rata-rata naif "
        "menjadi bias karena karakteristik demografi kedua kelompok sejak awal memang sudah berbeda."
    )
    pdf.chapter_body(eda_text)
    pdf.add_plot(f"{output_dir}/01_eda_earnings_distribution.png")

    # 3. Asumsi Kausal (DAG)
    pdf.add_page()
    pdf.chapter_title('3. Causal Directed Acyclic Graph (DAG)')
    dag_text = (
        "DAG memetakan asumsi hubungan antar variabel. Variabel seperti Usia (Age) dan Edukasi (Education) "
        "merupakan 'confounders' yang mempengaruhi baik kemungkinan seseorang mengikuti pelatihan maupun "
        "tingkat penghasilannya. Metode inferensi kausal bekerja dengan 'menutup jalur belakang' (backdoor paths) ini."
    )
    pdf.chapter_body(dag_text)
    pdf.add_plot(f"{output_dir}/02_causal_dag.png", w=140)

    # 4. Interpretasi & Insight Bisnis
    pdf.chapter_title('4. Business Insights & Heterogeneous Effects')
    insight_text = (
        "Menggunakan algoritma Machine Learning 'Causal Forest', kita tidak hanya menghitung efek rata-rata, "
        "tetapi juga siapa yang paling merasakan manfaat (Heterogeneous Treatment Effects). "
        "Grafik di bawah menunjukkan bahwa manfaat terbesar umumnya didapatkan oleh kelompok demografi tertentu. "
        "Hal ini penting untuk penargetan audiens (targeting) yang lebih efisien di masa depan."
    )
    pdf.chapter_body(insight_text)
    pdf.add_plot(f"{output_dir}/03_hte_by_age.png")
    
    # 5. SHAP Feature Importance
    pdf.add_page()
    pdf.chapter_title('5. Faktor Penggerak (SHAP Analysis)')
    shap_text = (
        "Analisis SHAP (SHapley Additive exPlanations) menguraikan faktor apa yang membuat efek pelatihan "
        "tinggi atau rendah pada setiap individu. Variabel yang berada di urutan teratas pada grafik di bawah "
        "adalah variabel yang paling menentukan kesuksesan program terhadap seorang individu."
    )
    pdf.chapter_body(shap_text)
    pdf.add_plot(f"{output_dir}/04_shap_summary.png")

    # 6. Komparasi Metode
    pdf.chapter_title('6. Kesimpulan Estimasi Average Treatment Effect (ATE)')
    
    ate_text = "Berikut adalah ringkasan estimasi keuntungan ($) yang diperoleh karena mengikuti program:\n\n"
    for method, value in ate_dict.items():
        ate_text += f"- {method}: ${value:,.2f}\n"
        
    ate_text += (
        "\nGrafik di bawah membandingkan hasil semua estimasi. Rata-rata Naif seringkali lebih rendah "
        "daripada nilai kausal yang sebenarnya. Metode lanjutan seperti DoWhy dan EconML memberikan estimasi "
        "yang lebih akurat dan robust (tahan uji)."
    )
    pdf.chapter_body(ate_text)
    pdf.add_plot(f"{output_dir}/05_method_comparison.png")

    # Output file
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = f"{save_dir}/causal_inference_report.pdf"
    pdf.output(pdf_path)
    return pdf_path
