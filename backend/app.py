from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import io
import uuid
import time
import warnings
import logging

from model_utils import predict_environment
from rag_utils import generate_health_advisory, answer_followup_question
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# Suppress unnecessary logs/warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "..", "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["LAST_ENV_CLASS"] = None

# ------------------------------
# Utility: clean old uploads
# ------------------------------
def cleanup_old_uploads(folder, max_age_seconds=7200):
    now = time.time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            if now - os.path.getmtime(file_path) > max_age_seconds:
                try:
                    os.remove(file_path)
                except:
                    pass

# ------------------------------
# Routes: Pages
# ------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

# ------------------------------
# Route: Analyze image
# ------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    cleanup_old_uploads(app.config["UPLOAD_FOLDER"])

    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Save uploaded image
    filename = str(uuid.uuid4()) + "_" + image_file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(filepath)

    # 1️⃣ Predict environment
    label, confidence = predict_environment(filepath)

    # 2️⃣ Generate RAG info
    diseases, preventive_measures, health_guidelines = generate_health_advisory(label)

    # Store last detected class for follow-up questions
    app.config["LAST_ENV_CLASS"] = label

    # 3️⃣ Return JSON
    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "diseases": diseases,
        "preventive_measures": preventive_measures,
        "health_guidelines": health_guidelines,
        "image": filename,
        "rag_answer": "Optional RAG summary"
    })

# ------------------------------
# Route: Follow-up question
# ------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    environment_class = app.config.get("LAST_ENV_CLASS", "")

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if not environment_class:
        return jsonify({"error": "Analyze an image first to establish context."}), 400

    answer = answer_followup_question(environment_class, question)
    return jsonify({"answer": answer})

# ------------------------------
# Route: Serve uploaded files
# ------------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ------------------------------
# Route: Download PDF report
# ------------------------------
@app.route("/download_report", methods=["POST"])
def download_report():
    data = request.get_json(silent=True) or {}
    image_filename = os.path.basename((data.get("image") or "").strip())
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename) if image_filename else None

    pdf_buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    section_style = styles['Heading2']
    text_style = styles['BodyText']
    elements = []

    # Title & Header
    elements.append(Paragraph("Environmental AI Healthcare System", title_style))
    elements.append(Paragraph("Environmental Health Assessment Report", section_style))
    elements.append(Spacer(1, 20))

    # Image
    if image_path and os.path.exists(image_path):
        try:
            image_reader = ImageReader(image_path)
            img_width, img_height = image_reader.getSize()
            max_width, max_height = 6.2 * inch, 3.4 * inch
            scale = min(max_width / img_width, max_height / img_height, 1)
            elements.append(RLImage(image_path, width=img_width * scale, height=img_height * scale))
            elements.append(Spacer(1, 14))
        except:
            elements.append(Paragraph("Uploaded image could not be embedded.", styles['Normal']))
            elements.append(Spacer(1, 10))

    # Data sections
    prediction = data.get("prediction", "N/A")
    confidence = data.get("confidence", "N/A")
    diseases = data.get("diseases", []) or []
    preventive_measures = data.get("preventive_measures", []) or []
    health_guidelines = data.get("health_guidelines", []) or []

    elements.append(Paragraph(f"<b>Detected Environment:</b> {prediction}", text_style))
    elements.append(Paragraph(f"<b>Model Confidence:</b> {confidence}%", text_style))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Health Risks", section_style))
    for item in diseases:
        elements.append(Paragraph(f"• {item}", text_style))

    elements.append(Paragraph("Preventive Measures", section_style))
    for item in preventive_measures:
        elements.append(Paragraph(f"• {item}", text_style))

    elements.append(Spacer(1, 8))
    elements.append(Paragraph("Health Guidelines", section_style))
    for item in health_guidelines:
        elements.append(Paragraph(f"• {item}", text_style))

    elements.append(Spacer(1, 25))
    elements.append(Paragraph(
        "This report was automatically generated by the Environmental AI Healthcare System "
        "using CNN-based environmental detection and Retrieval-Augmented Generation.",
        styles['Italic']
    ))

    # Build PDF
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    doc.build(elements)
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="environmental_health_report.pdf"
    )

# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)