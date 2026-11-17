import warnings
import os
import logging

# Silence console warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("easyocr").setLevel(logging.ERROR)

import gradio as gr
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Optional multi-page PDF support via PyMuPDF
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Load EasyOCR with CRNN + BiLSTM + CTC recognition network
reader = easyocr.Reader(
    lang_list=['en'],
    gpu=False,
    recog_network='standard',
    verbose=False
)


def enhance_image_for_deep_ocr(img_rgb, upscale_factor=2.0):
    """
    Enhances faint, distant, or low-contrast text (e.g., vehicle license plates,
    shadows, small document text) using upscaling, CLAHE contrast enhancement,
    and unsharp masking.
    """
    if upscale_factor > 1.0:
        h, w = img_rgb.shape[:2]
        if max(h, w) < 2200:
            img_rgb = cv2.resize(
                img_rgb,
                None,
                fx=upscale_factor,
                fy=upscale_factor,
                interpolation=cv2.INTER_CUBIC
            )

    # Apply CLAHE to the luminance channel in LAB color space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # Sharpen edges using Unsharp Masking
    gaussian = cv2.GaussianBlur(enhanced_rgb, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_rgb, 1.5, gaussian, -0.5, 0)
    return sharpened


def compute_iou(boxA, boxB):
    """Calculates Intersection-over-Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)


def deep_scan_single_image(pil_img, conf_thresh, deep_mode, draw_boxes):
    """
    Runs sensitive CRAFT + BiLSTM text detection. When deep_mode is enabled,
    combines detections from both the raw image and an enhanced/upscaled pass.
    """
    orig_rgb = np.array(pil_img.convert("RGB"))
    orig_h, orig_w = orig_rgb.shape[:2]

    ocr_kwargs = dict(
        detail=1,
        paragraph=False,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
        canvas_size=2560,
        mag_ratio=1.5
    )

    # Pass 1: Original image
    raw_detections = []
    for (bbox, text, prob) in reader.readtext(orig_rgb, **ocr_kwargs):
        if prob >= conf_thresh and text.strip():
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            rect = [min(xs), min(ys), max(xs), max(ys)]
            raw_detections.append((rect, text.strip(), prob))

    # Pass 2: CLAHE-enhanced + upscaled + sharpened image
    if deep_mode:
        scale = 2.0 if max(orig_h, orig_w) < 2200 else 1.0
        enhanced_rgb = enhance_image_for_deep_ocr(orig_rgb, upscale_factor=scale)
        for (bbox, text, prob) in reader.readtext(enhanced_rgb, **ocr_kwargs):
            if prob >= conf_thresh and text.strip():
                xs = [p[0] / scale for p in bbox]
                ys = [p[1] / scale for p in bbox]
                rect = [min(xs), min(ys), max(xs), max(ys)]
                raw_detections.append((rect, text.strip(), prob))

    # Deduplicate overlapping boxes keeping the highest confidence text
    raw_detections.sort(key=lambda x: x[2], reverse=True)
    final_detections = []
    for rect, text, prob in raw_detections:
        is_duplicate = False
        for kept_rect, _, _ in final_detections:
            if compute_iou(rect, kept_rect) > 0.45:
                is_duplicate = True
                break
        if not is_duplicate:
            final_detections.append((rect, text, prob))

    # Sort in natural reading order: top-to-bottom, then left-to-right
    final_detections.sort(key=lambda item: (round(item[0][1] / 20), item[0][0]))

    annotated = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    lines = []

    for rect, text, _ in final_detections:
        lines.append(text)
        if draw_boxes:
            draw.rectangle(rect, outline="#00FF00", width=3)

    return (annotated if draw_boxes else pil_img), lines


def process_input(camera_or_img, doc_file, conf_thresh, deep_mode, draw_boxes):
    pages_to_scan = []

    if doc_file is not None:
        file_path = doc_file.name if hasattr(doc_file, "name") else str(doc_file)
        if file_path.lower().endswith(".pdf"):
            if not PDF_SUPPORT:
                return None, "PDF support requires PyMuPDF (pymupdf)."
            pdf_doc = fitz.open(file_path)
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                pix = page.get_pixmap(dpi=250)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages_to_scan.append((f"Page {page_num + 1}", img))
        else:
            pages_to_scan.append(("Uploaded File", Image.open(file_path).convert("RGB")))

    elif camera_or_img is not None:
        pages_to_scan.append(("Captured Image", Image.fromarray(camera_or_img).convert("RGB")))

    else:
        return None, "Please upload an image, upload a PDF document, or capture a photo using your camera."

    all_extracted_text = []
    first_annotated_img = None

    for label, pil_img in pages_to_scan:
        annotated_img, extracted_lines = deep_scan_single_image(
            pil_img, conf_thresh, deep_mode, draw_boxes
        )
        if first_annotated_img is None:
            first_annotated_img = annotated_img

        if len(pages_to_scan) > 1:
            all_extracted_text.append(f"--- {label} ---")

        if extracted_lines:
            all_extracted_text.extend(extracted_lines)
        else:
            all_extracted_text.append("[No readable text detected on this page]")

    return first_annotated_img, "\n".join(all_extracted_text)


# Build Gradio 6 Compatible Interface
with gr.Blocks(title="Deep BiLSTM OCR Scanner") as demo:
    gr.Markdown("# 🔍 Deep BiLSTM Text & Number Plate Extractor")
    gr.Markdown(
        "Extracts complete text from **dense documents, multi-page PDFs, low-contrast images, "
        "and road vehicle license plates** using multi-pass contrast enhancement + "
        "**CRNN-BiLSTM** sequence recognition."
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(
                label="📷 Upload Image OR Use Phone Front/Back Camera",
                sources=["upload", "webcam"],
                type="numpy"
            )
            file_input = gr.File(
                label="📄 Or Upload Full Document / Multi-Page PDF",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
            )
            deep_toggle = gr.Checkbox(
                value=True,
                label="Enable Deep Multi-Pass Analysis (CLAHE Contrast + 2x Upscale for License Plates & Faint Text)"
            )
            box_toggle = gr.Checkbox(
                value=True,
                label="Highlight All Detected Text Regions"
            )
            conf_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.15,
                step=0.05,
                label="Confidence Threshold (Lower = Detect More Faint/Distant Text)"
            )
            run_btn = gr.Button("🚀 Run Deep Text Extraction", variant="primary")

        with gr.Column():
            out_image = gr.Image(label="Annotated Detection Map", type="pil")
            out_text = gr.Textbox(
                label="Complete Extracted Text",
                lines=16,
                placeholder="All detected text lines will appear here..."
            )

    run_btn.click(
        fn=process_input,
        inputs=[img_input, file_input, conf_slider, deep_toggle, box_toggle],
        outputs=[out_image, out_text]
    )
    img_input.change(
        fn=process_input,
        inputs=[img_input, file_input, conf_slider, deep_toggle, box_toggle],
        outputs=[out_image, out_text]
    )
    file_input.change(
        fn=process_input,
        inputs=[img_input, file_input, conf_slider, deep_toggle, box_toggle],
        outputs=[out_image, out_text]
    )

demo.launch()