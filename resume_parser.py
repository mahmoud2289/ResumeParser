# !pip install pymupdf docx2txt spacy transformers torch
# !python -m spacy download en_core_web_sm

import fitz  # PyMuPDF
import docx2txt
import spacy
import re
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from pathlib import Path

# -------------------------------
# Load spaCy model
# -------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Load NER model
# -------------------------------
model_name = "yashpwr/resume-ner-bert-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)


class ResumeParser:
    def __init__(self):
        self.section_headers = {
            'education': ['education', 'academic', 'qualification', 'degree'],
            'experience': ['experience', 'employment', 'work history', 'professional experience'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise'],
            'projects': ['projects', 'project experience'],
            'certifications': ['certifications', 'certificates', 'licenses']
        }

    # Step 1a: Extract text from PDF
    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text

    # Step 1b: Extract text from DOCX
    def extract_text_from_docx(self, docx_path):
        return docx2txt.process(docx_path)

    # Step 1c: Clean and preprocess text
    def clean_and_preprocess(self, text):
        doc = nlp(text)
        text = re.sub(r'\s+', ' ', text).strip()

        processed_tokens = []
        original_tokens = []

        for token in doc:
            original_tokens.append(token.text)
            if not token.is_stop and not token.is_punct:
                processed_tokens.append(token.lemma_.lower())

        return {
            'original_text': text,
            'processed_tokens': processed_tokens,
            'original_tokens': original_tokens
        }

    # Step 1d: Extract contact info
    def extract_contact_info(self, text):
        contact_info = {}

        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'

        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        github = re.findall(github_pattern, text, re.IGNORECASE)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        potential_name = lines[0] if lines else None

        contact_info['name'] = potential_name
        contact_info['email'] = emails[0] if emails else None
        contact_info['phone'] = phones[0] if phones else None
        contact_info['linkedin'] = linkedin[0] if linkedin else None
        contact_info['github'] = github[0] if github else None

        return contact_info

    # Step 1e: Apply NER
    def extract_entities_with_ner(self, text, confidence_threshold=0.3):
        if not text:
            return {}

        max_length = 512
        stride = 50
        all_entities = []
        text_len = len(text)

        for chunk_start in range(0, text_len, max_length - stride):
            chunk = text[chunk_start: chunk_start + max_length]
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                return_offsets_mapping=True
            )
            offset_mapping = inputs.pop("offset_mapping")[0].tolist()

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)[0]
                probs = torch.softmax(logits, dim=2)[0]

            current_entity = None

            for i, pred_id in enumerate(preds):
                start_off, end_off = offset_mapping[i]
                if start_off == 0 and end_off == 0:
                    continue

                label = model.config.id2label[pred_id.item()]
                confidence = probs[i][pred_id].item()

                if label.startswith("B-"):
                    if current_entity and current_entity['avg_confidence'] >= confidence_threshold:
                        ent_text = chunk[current_entity['start']: current_entity['end']]
                        all_entities.append({
                            'label': current_entity['label'],
                            'text': ent_text,
                            'confidence': float(current_entity['avg_confidence'])
                        })
                    current_entity = {
                        'label': label[2:],
                        'start': start_off,
                        'end': end_off,
                        'confidences': [confidence],
                        'avg_confidence': confidence
                    }
                elif label.startswith("I-") and current_entity:
                    if label[2:] == current_entity['label']:
                        current_entity['end'] = end_off
                        current_entity['confidences'].append(confidence)
                        current_entity['avg_confidence'] = float(np.mean(current_entity['confidences']))
                else:
                    if current_entity and current_entity['avg_confidence'] >= confidence_threshold:
                        ent_text = chunk[current_entity['start']: current_entity['end']]
                        all_entities.append({
                            'label': current_entity['label'],
                            'text': ent_text,
                            'confidence': float(current_entity['avg_confidence'])
                        })
                    current_entity = None

            if current_entity and current_entity['avg_confidence'] >= confidence_threshold:
                ent_text = chunk[current_entity['start']: current_entity['end']]
                all_entities.append({
                    'label': current_entity['label'],
                    'text': ent_text,
                    'confidence': float(current_entity['avg_confidence'])
                })

        # Deduplicate entities
        dedup = {}
        for ent in all_entities:
            key = (ent['label'], ent['text'].strip())
            if not key[1]:
                continue
            if key not in dedup or ent['confidence'] > dedup[key]['confidence']:
                dedup[key] = {'label': ent['label'], 'text': ent['text'].strip(), 'confidence': ent['confidence']}

        grouped_entities = {}
        for v in dedup.values():
            grouped_entities.setdefault(v['label'], []).append({
                'text': v['text'],
                'confidence': v['confidence']
            })

        return grouped_entities

    # Step 1f: Segment sections
    def segment_sections(self, text, contact_info):
        lines = text.split('\n')
        sections = {}
        current_section = 'other'
        current_content = []

        skip_values = set(filter(None, contact_info.values()))

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped in skip_values:
                continue

            section_found = False
            for section_type, keywords in self.section_headers.items():
                pattern = r'^\s*(?:' + '|'.join(re.escape(k) for k in keywords) + r')\s*:?\s*$'
                if re.match(pattern, line, re.IGNORECASE):
                    if current_content:
                        sections.setdefault(current_section, []).append('\n'.join(current_content).strip())
                    current_section = section_type
                    current_content = []
                    section_found = True
                    break

            if not section_found:
                current_content.append(line)

        if current_content:
            sections.setdefault(current_section, []).append('\n'.join(current_content).strip())

        # further split into clean lists
        structured_sections = {}
        for sec, contents in sections.items():
            structured_sections[sec] = []
            for block in contents:
                items = [i.strip("-• ").strip() for i in block.split("\n") if i.strip()]
                structured_sections[sec].extend(items)

        return structured_sections

    # Step 1g: Full pipeline
    def parse_resume(self, file_path, confidence_threshold=0.3):
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.pdf':
            raw_text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            raw_text = self.extract_text_from_docx(file_path)
        elif file_path.suffix.lower() in ['.txt']:
            raw_text = file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        preprocessed = self.clean_and_preprocess(raw_text)
        contact_info = self.extract_contact_info(raw_text)
        entities = self.extract_entities_with_ner(raw_text, confidence_threshold)
        sections = self.segment_sections(raw_text, contact_info)

        structured_resume = {
            'file_name': file_path.name,
            'contact_info': contact_info,
            #'entities': entities,
            'sections': sections,
            #'metadata': {
                #'total_tokens': len(preprocessed['processed_tokens']),
                #'processed': True
            #}
        }

        return structured_resume


# -------------------------------
# Example usage with a file
# -------------------------------
def main():
    parser = ResumeParser()

    # Replace this with your actual resume path
    file_path = "resume2.pdf"  # or .docx / .txt
    structured_resume = parser.parse_resume(file_path)

    print(json.dumps(structured_resume, indent=2, ensure_ascii=False))
    return structured_resume


if __name__ == "__main__":
    result = main()
