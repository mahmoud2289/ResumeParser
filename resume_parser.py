import fitz  # PyMuPDF
import docx2txt
import spacy
import re
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import phonenumbers
from fuzzywuzzy import fuzz
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCache:
    # caching for model to avoid reloading
    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_spacy_model(self):
        if 'spacy' not in self._models:
            try:
                self._models['spacy'] = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model")
            except OSError:
                logger.info("Downloading spaCy model")
                import os
                os.system("python -m spacy download en_core_web_sm")
                self._models['spacy'] = spacy.load("en_core_web_sm")
        return self._models['spacy']

    def get_ner_model(self):
        if 'ner_tokenizer' not in self._models:
            model_name = "yashpwr/resume-ner-bert-v2"
            self._models['ner_tokenizer'] = AutoTokenizer.from_pretrained(model_name)
            self._models['ner_model'] = AutoModelForTokenClassification.from_pretrained(model_name)
            logger.info("Loaded NER model")
        return self._models['ner_tokenizer'], self._models['ner_model']

class ResumeParser:
    def __init__(self):
        self.model_cache = ModelCache()
        self.nlp = self.model_cache.get_spacy_model()
        self.tokenizer, self.model = self.model_cache.get_ner_model()

        self.section_headers = {
            'education': ['education', 'academic', 'qualification', 'degree', 'university', 'college'],
            'experience': ['experience', 'employment', 'work history', 'professional experience',
                          'work experience', 'career history', 'professional background'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise', 'technologies',
                      'technical competencies', 'core competencies', 'tech stack'],
            'projects': ['projects', 'project experience', 'personal projects', 'key projects'],
            'certifications': ['certifications', 'certificates', 'licenses', 'credentials'],
            'summary': ['summary', 'profile', 'objective', 'about', 'introduction', 'professional summary'],
            'awards': ['awards', 'honors', 'achievements', 'recognition', 'accomplishments'],
            'publications': ['publications', 'research', 'papers', 'articles'],
            'languages': ['languages', 'language proficiency', 'spoken languages']
        }

        # Entities to exclude from NER output
        self.excluded_entity_types = {
            'Email Address',  # Already in contact_info
            'Phone',          # Already in contact_info
            'Degree',         # Already in education section
            'Name'            # Already in contact_info
        }

        # Date patterns (excluding phone numbers)
        self.date_patterns = [
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
            r'\b\d{1,2}/\d{4}\b',
            r'\b(19|20)\d{2}\b',  # Only 1900-2099 to avoid phone numbers
            r'\b(Present|Current|Now|Ongoing)\b'
        ]

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text")
            doc.close()

            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path}")
                return ""

            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def extract_text_from_docx(self, docx_path: Path) -> str:
        try:
            text = docx2txt.process(str(docx_path))
            if not text.strip():
                logger.warning(f"No text extracted from {docx_path}")
                return ""
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

    def clean_and_preprocess(self, text: str) -> Dict:
        # Clean and preprocess text with better tokenization
        if not text:
            return {
                'original_text': '',
                'processed_tokens': [],
                'original_tokens': []
            }

        try:
            doc = self.nlp(text)
            text = re.sub(r'\s+', ' ', text).strip()

            processed_tokens = []
            original_tokens = []

            for token in doc:
                original_tokens.append(token.text)
                if not token.is_stop and not token.is_punct and token.text.strip():
                    processed_tokens.append(token.lemma_.lower())

            return {
                'original_text': text,
                'processed_tokens': processed_tokens,
                'original_tokens': original_tokens
            }
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return {
                'original_text': text,
                'processed_tokens': [],
                'original_tokens': []
            }

    def normalize_phone(self, phone: str) -> Optional[str]:
        # Normalize phone number to international format
        try:
            parsed = phonenumbers.parse(phone, "US")  # Default to US
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        except:
            pass
        return phone  # Return original if can't parse

    def extract_name(self, text: str) -> Optional[str]:
        try:
            # Split into lines first to avoid grabbing multiple lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            # Try NER on first few lines only
            first_lines = '\n'.join(lines[:3])
            doc = self.nlp(first_lines)

            # Look for PERSON entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    # Validate it's actually a name
                    if 2 <= len(name.split()) <= 4 and len(name) < 50:
                        # Make sure it doesn't contain newlines or email-like strings
                        if '\n' not in name and '@' not in name and ':' not in name:
                            return name

            # Check first few lines with better validation
            for line in lines[:5]:
                # Skip lines with contact info markers
                if any(marker in line.lower() for marker in ['email', 'phone', '@', 'http', 'linkedin']):
                    continue

                # Check if it looks like a name
                words = line.split()
                if 2 <= len(words) <= 4:
                    # All words should start with capital letters
                    if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                        # No digits, no special chars except spaces and hyphens
                        if not re.search(r'[@\d:;]', line):
                            return line

            # Last resort: just return first line if it's reasonable
            if lines and len(lines[0]) < 50 and not re.search(r'[@:]', lines[0]):
                return lines[0]

            return None
        except Exception as e:
            logger.error(f"Error extracting name: {e}")
            return None

    def extract_contact_info(self, text: str) -> Dict:
        contact_info = {}

        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else None
        contact_info['email_confidence'] = 1.0 if emails else 0.0

        # Phone
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = self.normalize_phone(phones[0])
            contact_info['phone_confidence'] = 0.9
        else:
            contact_info['phone'] = None
            contact_info['phone_confidence'] = 0.0

        # LinkedIn
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        contact_info['linkedin'] = linkedin[0] if linkedin else None

        # GitHub
        github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'
        github = re.findall(github_pattern, text, re.IGNORECASE)
        contact_info['github'] = github[0] if github else None

        # Name with better extraction
        contact_info['name'] = self.extract_name(text)
        contact_info['name_confidence'] = 0.8 if contact_info['name'] else 0.0

        return contact_info

    def extract_dates(self, text: str) -> List[Dict]:
        # Extract dates and date ranges from text (excluding phone numbers)
        dates = []
        seen_positions = set()

        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group()
                position = match.start()

                # Skip if this position was already captured
                if position in seen_positions:
                    continue

                # Validate it's actually a year (not part of phone number)
                if date_text.isdigit():
                    year = int(date_text)
                    if not (1950 <= year <= 2030):
                        continue

                dates.append({
                    'text': date_text,
                    'position': position
                })
                seen_positions.add(position)

        return dates

    def extract_gpa(self, text: str) -> Optional[float]:
        # Extract GPA from education section
        gpa_patterns = [
            r'GPA[:\s]*(\d+\.\d+)',
            r'Grade[:\s]*(\d+\.\d+)',
            r'(\d+\.\d+)\s*/\s*4\.0',
            r'(\d+\.\d+)\s*/\s*5\.0'  # Some universities use 5.0 scale
        ]
        for pattern in gpa_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    gpa = float(match.group(1))
                    # Validate GPA range
                    if 0 <= gpa <= 5.0:
                        return gpa
                except:
                    pass
        return None

    def extract_degree_type(self, text: str) -> Optional[str]:
        degree_patterns = [
            (r'\b(Ph\.?D\.?|PhD|Doctorate)\b', 'PhD'),
            (r'\b(M\.?S\.?|M\.?Sc\.?|Master of Science)\b', 'Masters'),
            (r'\b(M\.?B\.?A\.?|Master of Business)\b', 'MBA'),
            (r'\b(M\.?A\.?|Master of Arts)\b', 'Masters'),
            (r'\b(B\.?S\.?|B\.?Sc\.?|Bachelor of Science)\b', 'Bachelors'),
            (r'\b(B\.?A\.?|Bachelor of Arts)\b', 'Bachelors'),
            (r'\b(Associate)\b', 'Associate'),
        ]

        for pattern, degree_type in degree_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return degree_type
        return None

    def parse_skills(self, skills_text: str) -> List[str]:
        # Parse skills section into individual skills
        # Common separators: comma, semicolon, pipe, bullet points
        separators = r'[,;|•\n]'
        skills = re.split(separators, skills_text)

        # Clean and filter skills
        parsed_skills = []
        for skill in skills:
            skill = skill.strip()
            # Remove bullet points and numbers
            skill = re.sub(r'^[\-\*\d\.]+\s*', '', skill).strip()
            # Skip empty or very long entries
            if skill and 2 <= len(skill) <= 50:
                parsed_skills.append(skill)

        return parsed_skills

    def validate_entity(self, label: str, text: str) -> bool:
        text = text.strip()

        # Exclude redundant entity types
        if label in self.excluded_entity_types:
            return False

        # Filter out nonsense entities
        if len(text) < 2 or len(text) > 100:
            return False

        # Check for specific label validations
        if label == "Companies worked at":
            # Should not contain too many special chars or newlines
            if '\n' in text or text.count('-') > 2:
                return False
            # Should not be a full sentence describing work
            if len(text.split()) > 8:
                return False
            # Should not contain common action verbs (indicates it's a job description)
            action_verbs = ['developed', 'designed', 'built', 'created', 'managed', 'led', 'worked']
            if any(verb in text.lower() for verb in action_verbs):
                return False
            return True

        elif label == "Skills":
            # Should not be a full sentence
            if len(text.split()) > 15:
                return False
            # Should not contain verbs (indicates description not skill)
            if re.search(r'\b(developed|designed|built|created|worked|using|with)\b', text.lower()):
                return False
            return True

        elif label == "Designation":
            # Should be reasonable length job title
            if len(text.split()) > 10:
                return False
            # Should not start with action verbs
            if re.match(r'^(developed|designed|built|created|managed)', text, re.IGNORECASE):
                return False
            return True

        elif label == "College Name":
            # Should be reasonable length
            if len(text.split()) > 10:
                return False
            return True

        elif label == "Graduation Year":
            # Should be a valid year
            year_match = re.search(r'(19|20)\d{2}', text)
            if year_match:
                year = int(year_match.group())
                return 1950 <= year <= 2030
            return False

        # Default: accept the entity
        return True

    def extract_entities_with_ner(self, text: str, confidence_threshold: float = 0.45) -> Dict:
        if not text:
            return {}

        max_length = 512
        stride = 128
        all_entities = []
        text_len = len(text)

        try:
            for chunk_start in range(0, text_len, max_length - stride):
                chunk = text[chunk_start: chunk_start + max_length]
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_offsets_mapping=True
                )
                offset_mapping = inputs.pop("offset_mapping")[0].tolist()

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=2)[0]
                    probs = torch.softmax(logits, dim=2)[0]

                current_entity = None

                for i, pred_id in enumerate(preds):
                    start_off, end_off = offset_mapping[i]
                    if start_off == 0 and end_off == 0:
                        continue

                    label = self.model.config.id2label[pred_id.item()]
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

            # Deduplicate and group entities with validation
            dedup = {}
            for ent in all_entities:
                key = (ent['label'], ent['text'].strip())
                if not key[1]:
                    continue

                # Validate entity before adding
                if not self.validate_entity(key[0], key[1]):
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
        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return {}

    def fuzzy_match_section(self, line: str, threshold: int = 80) -> Optional[str]:
        # Fuzzy match section headers for better detection
        line_lower = line.lower().strip()

        for section_type, keywords in self.section_headers.items():
            for keyword in keywords:
                ratio = fuzz.ratio(line_lower, keyword)
                if ratio >= threshold:
                    return section_type
        return None

    def group_experience_entries(self, experience_items: List[Dict]) -> List[Dict]:
        grouped = []
        current_job = None

        for item in experience_items:
            text = item['text']
            has_dates = 'dates' in item and item['dates']

            # Check if this looks like a job title line (has dates or follows pattern)
            is_job_title = (
                has_dates or
                re.search(r'\bat\b|\bwith\b|\b@\b', text, re.IGNORECASE) or
                (len(text.split()) <= 15 and not text.startswith(('-', '•', '*')))
            )

            if is_job_title and (has_dates or current_job is None):
                # Start a new job entry
                if current_job:
                    grouped.append(current_job)

                current_job = {
                    'title': text,
                    'dates': item.get('dates', []),
                    'responsibilities': []
                }
            elif current_job:
                # Add as bullet point to current job
                current_job['responsibilities'].append(text)
            else:
                # Create standalone entry
                grouped.append({
                    'title': text,
                    'dates': item.get('dates', []),
                    'responsibilities': []
                })

        # Don't forget the last job
        if current_job:
            grouped.append(current_job)

        return grouped

    def segment_sections(self, text: str, contact_info: Dict) -> Dict:
        lines = text.split('\n')
        sections = {}
        current_section = 'other'
        current_content = []

        # Build set of values to skip (contact info and common prefixes)
        skip_values = set()
        for key, value in contact_info.items():
            if value and isinstance(value, str):
                skip_values.add(value)
                # Also add variations without "Email:" or "Phone:" prefixes
                if ':' in value:
                    skip_values.add(value.split(':', 1)[1].strip())

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Skip contact info lines
            if any(skip_val in line for skip_val in skip_values):
                continue

            # Skip lines that are just "Email:" or "Phone:" labels
            if re.match(r'^(email|phone|linkedin|github):\s*$', stripped, re.IGNORECASE):
                continue

            # Try exact match first
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

            # Try fuzzy match if exact failed
            if not section_found and len(stripped.split()) <= 3:
                fuzzy_section = self.fuzzy_match_section(stripped)
                if fuzzy_section:
                    if current_content:
                        sections.setdefault(current_section, []).append('\n'.join(current_content).strip())
                    current_section = fuzzy_section
                    current_content = []
                    section_found = True

            if not section_found:
                current_content.append(line)

        if current_content:
            sections.setdefault(current_section, []).append('\n'.join(current_content).strip())

        # Structure sections better with enhanced parsing
        structured_sections = {}
        for sec, contents in sections.items():
            structured_sections[sec] = []
            for block in contents:
                # Preserve bullet structure
                items = [i.strip() for i in re.split(r'\n(?=[•\-\*]|\d+\.)', block) if i.strip()]
                for item in items:
                    clean_item = re.sub(r'^[•\-\*]\s*', '', item).strip()
                    if clean_item:
                        # Extract dates if present
                        dates = self.extract_dates(clean_item)
                        entry = {'text': clean_item}
                        if dates:
                            entry['dates'] = [d['text'] for d in dates]

                        # Enhanced education parsing - extract GPA only
                        if sec == 'education':
                            gpa = self.extract_gpa(clean_item)
                            if gpa:
                                entry['gpa'] = gpa

                        # Enhanced skills parsing
                        if sec == 'skills':
                            # Parse individual skills from text
                            individual_skills = self.parse_skills(clean_item)
                            if individual_skills:
                                entry['parsed_skills'] = individual_skills

                        structured_sections[sec].append(entry)

        # Group experience entries better
        if 'experience' in structured_sections:
            structured_sections['experience'] = self.group_experience_entries(
                structured_sections['experience']
            )

        return structured_sections

    def calculate_years_of_experience(self, sections: Dict) -> float:
        # Calculate total years of experience from experience section
        if 'experience' not in sections:
            return 0.0

        total_months = 0
        current_year = datetime.now().year

        for entry in sections['experience']:
            # Handle both old format and new grouped format
            dates = entry.get('dates', [])

            if len(dates) >= 2:
                try:
                    # Parse start and end dates
                    start_match = re.search(r'(19|20)\d{2}', dates[0])
                    if not start_match:
                        continue
                    start_year = int(start_match.group())

                    if 'present' in dates[-1].lower() or 'current' in dates[-1].lower():
                        end_year = current_year
                    else:
                        end_match = re.search(r'(19|20)\d{2}', dates[-1])
                        if not end_match:
                            continue
                        end_year = int(end_match.group())

                    total_months += (end_year - start_year) * 12
                except Exception as e:
                    logger.debug(f"Could not parse dates: {dates}, error: {e}")
                    pass

        return round(total_months / 12, 1)

    def generate_summary_stats(self, structured_resume: Dict) -> Dict:
        stats = {}

        sections = structured_resume.get('sections', {})

        # Count items in each section
        stats['education_count'] = len(sections.get('education', []))
        stats['experience_count'] = len(sections.get('experience', []))
        stats['certifications_count'] = len(sections.get('certifications', []))
        stats['projects_count'] = len(sections.get('projects', []))

        # Skills count
        skills_section = sections.get('skills', [])
        if skills_section:
            total_skills = 0
            for skill_entry in skills_section:
                if 'parsed_skills' in skill_entry:
                    total_skills += len(skill_entry['parsed_skills'])
            stats['skills_count'] = total_skills
        else:
            stats['skills_count'] = 0

        # Check completeness
        stats['has_email'] = structured_resume['contact_info'].get('email') is not None
        stats['has_phone'] = structured_resume['contact_info'].get('phone') is not None
        stats['has_linkedin'] = structured_resume['contact_info'].get('linkedin') is not None
        stats['has_github'] = structured_resume['contact_info'].get('github') is not None

        # Completeness score (0-100)
        completeness_factors = [
            stats['has_email'],
            stats['has_phone'],
            stats['education_count'] > 0,
            stats['experience_count'] > 0,
            stats['skills_count'] > 0,
            structured_resume['contact_info'].get('name') is not None
        ]
        stats['completeness_score'] = round((sum(completeness_factors) / len(completeness_factors)) * 100, 1)

        return stats
    # Full pipeline
    def parse_resume(self, file_path: str, confidence_threshold: float = 0.45) -> Dict:
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            logger.info(f"Parsing resume: {file_path.name}")

            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                raw_text = self.extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                raw_text = self.extract_text_from_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                raw_text = file_path.read_text(encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            if not raw_text.strip():
                raise ValueError("No text could be extracted from the file")

            # Process resume
            preprocessed = self.clean_and_preprocess(raw_text)
            contact_info = self.extract_contact_info(raw_text)
            entities = self.extract_entities_with_ner(raw_text, confidence_threshold)
            sections = self.segment_sections(raw_text, contact_info)
            years_exp = self.calculate_years_of_experience(sections)

            structured_resume = {
                'file_name': file_path.name,
                'contact_info': contact_info,
                'sections': sections,
                'entities': entities,
                'metadata': {
                    'total_tokens': len(preprocessed['processed_tokens']),
                    'years_of_experience': years_exp,
                    'processed': True,
                    'timestamp': datetime.now().isoformat()
                }
            }

            # Add summary statistics
            structured_resume['summary_stats'] = self.generate_summary_stats(structured_resume)

            logger.info(f"Successfully parsed resume: {file_path.name}")
            return structured_resume

        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            return {
                'file_name': str(file_path),
                'error': str(e),
                'processed': False
            }

def main():
    parser = ResumeParser()

    file_path = "resume1.pdf"

    try:
        structured_resume = parser.parse_resume(file_path)
        print(json.dumps(structured_resume, indent=2, ensure_ascii=False))
        return structured_resume
    except Exception as e:
        logger.error(f"Failed to parse resume: {e}")
        return None

if __name__ == "__main__":
    result = main()