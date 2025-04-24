import streamlit as st
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
import openai
import spacy

# Initialize OpenAI and SpaCy
openai.api_key = 'YOUR_OPENAI_API_KEY'
nlp = spacy.load('en_core_web_sm')

def pdf_to_text(pdf_path):
    """Converts PDF document to plain text using OCR."""
    pages = convert_from_path(pdf_path)
    text = ''
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text

def extract_key_information(text):
    """Use NLP to extract key information from text."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def summarize_text(text):
    """Summarizes text using OpenAI's GPT."""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Summarize the following document: {text}",
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()

def process_files(folder_path):
    """Process all PDF files in a folder."""
    folder = Path(folder_path)
    pdf_files = list(folder.glob('*.pdf'))
    results = []
    for pdf_file in pdf_files:
        text = pdf_to_text(pdf_file)
        key_info = extract_key_information(text)
        summary = summarize_text(text)
        results.append({
            'filename': pdf_file.name,
            'key_info': key_info,
            'summary': summary
        })
    return results

# Streamlit app
st.title("Tender Document Processor")
folder_path = st.text_input("Folder Path", "")

if st.button("Process Files"):
    if folder_path:
        results = process_files(folder_path)
        for result in results:
            st.subheader(f"File: {result['filename']}")
            st.markdown("**Key Information:**")
            for info in result['key_info']:
                st.write(f"- {info[0]} ({info[1]})")
            st.markdown("**Summary:**")
            st.write(result['summary'])
    else:
        st.write("Please provide a valid folder path.")
