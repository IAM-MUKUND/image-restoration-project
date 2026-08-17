#!/usr/bin/env python3
"""Generate submission PDF presentation using ReportLab for SEMICON India Hackathon 2026."""

from pathlib import Path
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "submission"
OUTPUT_DIR.mkdir(exist_ok=True)
PDF_PATH = OUTPUT_DIR / "TeamName_KLA_PS01.pdf"


def build_pdf():
    # Landscape 11 x 8.5 inches
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(letter),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F2C59'),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        'SlideSubtitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10,
    )

    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#222222'),
    )

    bullet_style = ParagraphStyle(
        'SlideBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=4,
    )

    highlight_style = ParagraphStyle(
        'SlideHighlight',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#005B96'),
    )

    story = []

    def add_header(title, subtitle):
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(subtitle, subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#005B96'), spaceAfter=10))

    # ---------------------------------------------------------
    # SLIDE 1: Team Details
    # ---------------------------------------------------------
    add_header("Team Details", "SEMICON India Hackathon 2026 | Track 1 - KLA PS01")
    story.append(Paragraph("<b>Problem Statement:</b> AI-Based Restoration of Degraded Images for Semiconductor Inspection_KLA", body_style))
    story.append(Spacer(1, 10))

    team_data = [
        ["SR. NO", "ROLE", "NAME", "ACADEMIC YEAR", "COLLEGE / INSTITUTION"],
        ["1", "Team Leader", "Mukund Vinayak", "Final Year", "[Your College / Institution Name]"],
        ["2", "Member 1", "[Member 1 Name]", "Final Year", "[Your College / Institution Name]"],
        ["3", "Member 2", "[Member 2 Name]", "Final Year", "[Your College / Institution Name]"],
        ["4", "Member 3", "[Member 3 Name]", "Final Year", "[Your College / Institution Name]"],
    ]
    t1 = Table(team_data, colWidths=[0.6*inch, 1.2*inch, 2.0*inch, 1.4*inch, 4.0*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2C59')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F7FA')]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Contact Email:</b> email@example.com &nbsp;&nbsp;|&nbsp;&nbsp; <b>Contact Phone:</b> +91 XXXXXXXXXX", body_style))
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 2: Problem Statement Addressed
    # ---------------------------------------------------------
    add_header("Problem Statement Addressed", "Context, Signal Constraints & Practical Significance")
    p2 = [
        "<b>Selected Problem:</b> AI-Based Restoration of Degraded Images for Semiconductor Inspection (Track 1 / KLA PS01)",
        "<b>Task Brief:</b> Train a deep learning restoration network to map 128x128 degraded grayscale NumPy arrays to clean 256x256 ground truth arrays (joint 2x super-resolution and denoising).",
        "<b>Degradation Components:</b> Combined non-stationary multiplicative speckle noise, additive Gaussian noise, and 2x spatial downsampling.",
        "<b>Mandatory Signal Range Rule:</b> Ground truth targets are normalized to [0, 1]. Degraded input arrays contain valid out-of-bounds values (<0 or >1) which must <b>NOT</b> be clipped prior to model entry.",
        "<b>Industrial Importance:</b> Wafer inspection tools in semiconductor fabs require ultra-fast, high-precision image restoration to detect sub-10nm manufacturing defects without creating inspection bottlenecks."
    ]
    for bullet in p2:
        story.append(Paragraph(f"• {bullet}", bullet_style))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 3: Idea Description & Solution Overview
    # ---------------------------------------------------------
    add_header("Idea Description & Solution Overview", "Degradation-Aware Frequency Restormer (DAF-Restormer)")
    p3 = [
        "<b>Core Concept:</b> DAF-Restormer is a novel hybrid architecture that integrates dynamic degradation prompt encoding, spatial noise-strength estimation, and phase-preserving spectral frequency refinement into a Restormer backbone.",
        "<b>Dynamic Context Learning:</b> Predicts a 64-dimensional degradation embedding and a 128x128 local noise map directly from the corrupted input without relying on synthetic or fixed noise assumptions.",
        "<b>Two-Stage Perceptual Recipe:</b> Pre-trained for 10 epochs using Charbonnier + SSIM fidelity loss, followed by 2 epochs of low-rate LPIPS-AlexNet perceptual fine-tuning (DAF-Restormer-P).",
        "<b>Grain Elimination Outcome:</b> Achieves a <b>22% reduction in LPIPS score (0.2719 → 0.2122)</b>, effectively suppressing persistent micro-grain artifacts while maintaining crisp edge boundaries."
    ]
    for bullet in p3:
        story.append(Paragraph(f"• {bullet}", bullet_style))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 4: Proposed Solution Details
    # ---------------------------------------------------------
    add_header("Proposed Solution Details", "Architecture, Module Breakdown & Training Strategy")
    sol_data = [
        ["Component", "Technical Mechanism & Purpose"],
        ["Degradation Encoder", "Extracts global 64-dim prompt vector and 128x128 spatial noise map from input high-pass features."],
        ["Prompted Transformer", "Multi-DConv Head Transposed Attention (MDTA) + GDFN modulated by degradation prompt vectors."],
        ["Frequency Refinement", "Prompt-conditioned FFT magnitude gating in Fourier space that filters periodic grain while preserving phase."],
        ["ICNR PixelShuffle Head", "2x upsampling initialized with ICNR to eliminate sub-pixel checkerboard variance."],
        ["Progressive Supervision", "Auxiliary 128x128 clean-LR reconstruction head and heteroscedastic uncertainty estimation."]
    ]
    t4 = Table(sol_data, colWidths=[2.2*inch, 7.0*inch])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2C59')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F7FA')]),
    ]))
    story.append(t4)
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 5: Innovation & Uniqueness
    # ---------------------------------------------------------
    add_header("Innovation & Uniqueness", "Technical Novelties vs. Standard Restoration Networks")
    p5 = [
        "<b>1. Dynamic Degradation Prompting:</b> Replaces static noise models with a learned prompt encoder that adapts feature modulation per image.",
        "<b>2. Phase-Preserving Frequency Gating:</b> Operates in the Fourier domain to selectively damp high-frequency grain noise without introducing phase distortion or blur.",
        "<b>3. Perceptual Grain Suppression:</b> Downsampled LPIPS perceptual loss specifically eliminates grain artifacts that disrupt downstream semiconductor defect inspection.",
        "<b>4. Dual Deployment Flexibility:</b> Supports 1-View low-latency mode (34.3 img/s) and 8-View TTA accuracy mode (top PSNR/SSIM)."
    ]
    for bullet in p5:
        story.append(Paragraph(bullet, bullet_style))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 6: Results & Visual Impact
    # ---------------------------------------------------------
    add_header("Results & Quantitative Benchmarks", "Tesla T4 GPU Evaluation on 320 Held-out Official Validation Pairs")
    res_data = [
        ["Model Variant", "PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓", "T4 Latency (ms) ↓", "Throughput (img/s) ↑"],
        ["Bicubic Baseline", "23.279", "0.5443", "0.4312", "0.128 ms", "276.2 img/s"],
        ["NAFNet (253K params)", "25.932", "0.6467", "0.3952", "10.040 ms", "174.6 img/s"],
        ["SwinIR (405K params)", "26.377", "0.6588", "0.3641", "21.564 ms", "44.3 img/s"],
        ["Restormer Control (1.61M)", "28.393", "0.7709", "0.2719", "13.660 ms", "124.5 img/s"],
        ["DAF-Restormer-P (1-View)", "28.436", "0.7705", "0.2122", "35.800 ms", "34.3 img/s"],
        ["DAF-Restormer-P (8-View TTA)", "28.471", "0.7712", "0.2285", "264.723 ms", "3.8 img/s"],
    ]
    t6 = Table(res_data, colWidths=[2.5*inch, 1.3*inch, 1.2*inch, 1.2*inch, 1.5*inch, 1.5*inch])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2C59')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#F5F7FA')]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#E6F0FA')),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ]))
    story.append(t6)
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 7: Technology & Feasibility
    # ---------------------------------------------------------
    add_header("Technology Stack & Feasibility", "Hardware Requirements, Software Dependencies & Compute Hygiene")
    p7 = [
        "<b>Software Stack:</b> PyTorch 2.x, PyTorch AMP (FP16), Python 3.12, NumPy, SciPy, PyWavelets, Pytest.",
        "<b>Model Scale:</b> 4,172,518 parameters (~16.7 MB model checkpoint footprint).",
        "<b>Hardware Compatibility:</b> Evaluated on NVIDIA Tesla T4 and RTX GPUs; fully verified for NVIDIA H100 execution.",
        "<b>VRAM Efficiency:</b> Peak memory usage is ~2.3 GB VRAM during validation, easily operating within standard GPU limits.",
        "<b>Deployment Readiness:</b> Verified standalone <code>infer.py</code> CLI producing compliant <code>float32</code> [256, 256] arrays."
    ]
    for bullet in p7:
        story.append(Paragraph(f"• {bullet}", bullet_style))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 8: GitHub & Video Link
    # ---------------------------------------------------------
    add_header("GitHub Repository & Demo Video", "Submission Links")
    story.append(Paragraph("<b>GitHub Repository URL:</b>", highlight_style))
    story.append(Paragraph("<u>https://github.com/IAM-MUKUND/image-restoration-project</u>", body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Prototype / Simulation Video URL:</b>", highlight_style))
    story.append(Paragraph("<u>[Paste your YouTube / Loom Video Link here]</u>", body_style))
    story.append(PageBreak())

    # ---------------------------------------------------------
    # SLIDE 9: Research & References
    # ---------------------------------------------------------
    add_header("Research & References", "Theoretical Basis & Literature Citations")
    p9 = [
        "<b>1. Channel-Transposed Self-Attention:</b> Zamir et al., <i>'Restormer: Efficient Transformer for High-Resolution Image Restoration'</i>, CVPR 2022.",
        "<b>2. Sub-Pixel Convolution & ICNR:</b> Aitken et al., <i>'Checkerboard Artifact Free Sub-Pixel Convolution'</i>, arXiv:1707.02937, 2017.",
        "<b>3. Practical Degradation Modelling:</b> Zhang et al., <i>'Designing a Practical Degradation Model for Deep Image Super-Resolution'</i>, ICCV 2021."
    ]
    for bullet in p9:
        story.append(Paragraph(bullet, bullet_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    print(f"Successfully built PDF presentation at: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
