Summary

This Resume Parser is a Python NLP pipeline that extracts structured information from resumes in PDF, DOCX/DOC, and TXT formats, returning a clean JSON object.
How it works

    Text extraction — Raw text is pulled from the file using PyMuPDF (fitz) for PDFs, docx2txt for DOCX/DOC, and standard file reading for TXT.
    Preprocessing — The text is tokenized with spaCy (en_core_web_sm), which removes stop words, punctuation, and lemmatizes tokens.
    Contact & header extraction — Regex patterns extract email, phone, LinkedIn, GitHub, location, personal website, and name from the first lines of the resume.
    Section segmentation — Lines are matched against a fixed set of section header keywords (e.g., experience, education, skills) with fuzzy matching fallback via fuzzywuzzy.
    NER — A BERT-based token classifier (yashpwr/resume-ner-bert-v2) runs over the text in sliding 512-token windows (stride=128) with a confidence threshold of 0.45 to extract entities with labels: Skills, Designation, College Name, and Graduation Year. Entities labeled Email Address, Phone, Degree, Name, and Companies worked at are explicitly excluded.

Output structure

The final dictionary returned by parse_resume() contains:
Field	Description
file_name	Name of the parsed file
summary	Professional summary/profile paragraph (if found)
sections	Structured sections (see below)
processed	True on success, False on error

Included sections: education (with GPA and honors), experience (title + responsibilities list), skills, soft_skills, projects, certifications, awards, publications, volunteering

Explicitly excluded from output: contact_info (extracted internally but not returned), languages, interests, dates (stripped from all items), company (stripped from experience entries), summary section duplicate, other catch-all section, and metadata/summary_stats (commented out).
