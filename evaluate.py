# === Importing Libraries ===
import os
import base64
import tempfile

from constants import constants
from openai import OpenAI
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# === Constants ===
SOP_EVALUATION_PATH = "/home/aditya/Hiring Task/task/dristi/SOP_Evaluation_Test Series_Mentorship_Hindi (1).pdf"

# === Streamlit Page Setup ===
st.set_page_config(page_title="UPSC RAG Evaluator", layout="wide")
st.title("🧠 UPSC Answer Sheet Evaluator (RAG + GPT-4o)")
st.markdown(
    "Upload a scanned UPSC answer sheet PDF and get detailed feedback based on SOP."
)

# === File Upload ===
uploaded_file = st.file_uploader("📄 Upload UPSC Answer Sheet PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        pdf_path = tmp_pdf.name

    images = convert_from_path(pdf_path, dpi=300)
    st.success(f"✅ {len(images)} pages detected.")

    # === OpenAI and Embedding Setup ===
    client = OpenAI(api_key=constants.OPENAI_API_KEY)
    embedder = OpenAIEmbeddings(openai_api_key=constants.OPENAI_API_KEY)

    # === Load and Split SOP Document ===
    reader = PdfReader(SOP_EVALUATION_PATH)
    sop_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    sop_chunks = splitter.split_text(sop_text)
    sop_embeddings = embedder.embed_documents(sop_chunks)

    # === Initialize Qdrant Vector Store ===
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name="sop_chunks",
        vectors_config=VectorParams(
            size=len(sop_embeddings[0]), distance=Distance.COSINE
        ),
    )
    qdrant.upload_collection(
        collection_name="sop_chunks",
        vectors=sop_embeddings,
        payload=[{"text": chunk} for chunk in sop_chunks],
    )

    # === Function to Retrieve Relevant SOP Context ===
    def get_sop_context(text):
        """Returns top matching SOP chunks for a given query."""
        query_vec = embedder.embed_query(text)
        results = qdrant.search("sop_chunks", query_vector=query_vec, limit=3)
        return "\n".join([r.payload["text"] for r in results])

    # === Evaluate a Section ===
    def evaluate_section(section_name, section_text):
        """Generates feedback for a given answer section based on SOP."""
        sop_context = get_sop_context(section_text)
        prompt = f"""
You are a UPSC evaluator.

Evaluation Criteria:
- Content Adequacy
- Structure (IBC)
- Depth of Argument & Analysis
- Use of Examples / Data
- Presentation
- Language and Grammar
- Alignment with directive keyword

SOP Reference:
{sop_context}

Evaluate the section: {section_name}
Text:
{section_text}

Provide:
- 3 feedback points
- Marks out of 5
Format:
### {section_name}
- Comment 1
- Comment 2
- Comment 3
Marks Awarded: X.X / 5
"""
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return res.choices[0].message.content

    # === Evaluate Presentation ===
    def evaluate_presentation(full_text):
        """Evaluates handwriting and layout presentation quality."""
        sop_context = get_sop_context(
            "presentation quality in handwritten UPSC answers"
        )
        prompt = f"""
Evaluate presentation aspects based on the SOP:
{sop_context}

Answer Text:
{full_text}

Return:
### Presentation
Strengths:
- ...
- ...
Improvements:
- ...
- ...
- ...
Marks Awarded: X.X / 5
"""
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return res.choices[0].message.content

    # === Generate General Strength Feedback ===
    def generate_summary_feedback():
        """Provides summary strength points for the student."""
        prompt = """
Provide 3 strength points for a UPSC student's handwritten answer based on general structure, contextual relevance, and clarity.
Format:
### Strengths
- ...
- ...
- ...
"""
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        return res.choices[0].message.content

    # === Extract Student Name from First Page ===
    def extract_name(full_text):
        """Extracts student name if found in the answer content."""
        prompt = f"""
The following is the OCR-extracted handwritten Hindi content from a UPSC answer sheet:
{full_text}

If there is a student name mentioned (usually on the first page), extract it in plain format like "Name: Rahul Kumar". If no name found, respond with "Not found".
"""
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
        )
        return res.choices[0].message.content.strip()

    # === Evaluate Button Click ===
    if st.button("🔍 Extract, Segment & Evaluate"):
        full_text = ""
        with st.spinner("Extracting handwritten content..."):
            for i, image in enumerate(images):
                img_path = f"page_{i+1}.png"
                image.save(img_path)
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                os.remove(img_path)

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract handwritten Hindi text from a UPSC answer sheet page.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Extract all handwritten Hindi text. Only return plain text.",
                                },
                            ],
                        },
                    ],
                    max_tokens=2048,
                )
                full_text += response.choices[0].message.content + "\n"

        # === Name Extraction ===
        student_name = extract_name(full_text)
        student_name = (
            student_name.replace("Name:", "").strip()
            if "not found" not in student_name.lower()
            else "Student"
        )

        # === Segment Answer into IBC ===
        segment_prompt = f"""Segment the following answer into Introduction, Body, and Conclusion:
{full_text}
Format:
### Introduction
...
### Body
...
### Conclusion
...
"""
        seg_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": segment_prompt}],
            max_tokens=2048,
        )
        segmented = seg_resp.choices[0].message.content

        with st.expander("✂️ Segmented Answer", expanded=True):
            st.markdown(segmented)

        # === Begin Evaluation ===
        report = "\n\n"
        total_score = 0

        for section in ["Introduction", "Body", "Conclusion"]:
            if f"### {section}" in segmented:
                section_text = (
                    segmented.split(f"### {section}")[1].split("###")[0].strip()
                )
                eval_result = evaluate_section(section, section_text)
                report += eval_result + "\n\n"
                try:
                    score = float(
                        [
                            line
                            for line in eval_result.splitlines()
                            if "Marks Awarded" in line
                        ][0].split()[2]
                    )
                    total_score += score
                except:
                    pass

        # === Add Presentation and Summary Evaluation ===
        presentation_eval = evaluate_presentation(full_text)
        report += presentation_eval + "\n\n"
        try:
            score = float(
                [
                    line
                    for line in presentation_eval.splitlines()
                    if "Marks Awarded" in line
                ][0].split()[2]
            )
            total_score += score
        except:
            pass

        report = (
            f"Dear {student_name},\nTotal Marks: {round(total_score, 2)} / 20\n\n"
            + report
        )
        report += generate_summary_feedback()
        report += "\n\nAll the Best! Keep improving and striving for excellence."

        # === Display Report ===
        with st.expander("📋 Final Evaluation Report", expanded=True):
            st.markdown(report)

        # Clean the report by removing unnecessary characters
        cleaned_report_txt = report.replace("*", "").replace("#", "")

        # Define path to save .txt file in backend
        txt_file_path = os.path.join(os.getcwd(), "report.txt")

        # Write the report content to the file using UTF-8 encoding to preserve Hindi
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_report_txt)

        # Provide download link in Streamlit
        with open(txt_file_path, "rb") as f:
            b64_txt = base64.b64encode(f.read()).decode()
            download_link = f'<a href="data:text/plain;base64,{b64_txt}" download="UPSC_Evaluation_Report.txt">📥 Download Evaluation Report (.txt)</a>'
            st.markdown(download_link, unsafe_allow_html=True)

        # === Clean Up ===
        os.remove(pdf_path)
