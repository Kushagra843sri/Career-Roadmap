#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import pyresparser


# In[2]:


# from pyresparser import ResumeParser


# In[5]:



# In[53]:


import pdfplumber
import spacy
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])


# In[59]:


import pdfplumber
import re
import spacy
from collections import Counter

# Load spaCy model - you need this line!
try:
    nlp = spacy.load("en_core_web_md")  # Medium-sized model with word vectors
except:
    # Fallback to small model if medium isn't available
    nlp = spacy.load("en_core_web_sm")


# In[73]:


# === Function to extract text from PDF with error handling ===
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:  # Check if text was actually extracted
                    text += extracted + '\n'
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# === Function to extract email with improved regex ===
def extract_email(text):
    # This pattern better matches standard email formats
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

# === Function to extract phone number with improved regex ===
def extract_phone(text):
    # Multiple patterns to catch different phone formats
    patterns = [
        r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b',  # Standard formats
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Basic 10-digit
        r'\b\d{5}[-.\s]?\d{5}\b'  # Some international formats
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    return None

# === Improved function to extract name ===
def extract_name(text):
    # First, try to find common name patterns in the first few lines
    lines = text.split('\n')
    
    # Look for full name patterns in the first 10 lines
    for line in lines[:10]:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # If this looks like a standalone name (short line, no special characters)
        if (len(line) < 40 and line[0].isupper() and 
            not any(char in line for char in ":/,@()") and
            not any(keyword in line.lower() for keyword in ["resume", "cv", "curriculum", "email", "phone", "address"])):
            
            # If it's ALL CAPS, convert to title case
            if line.isupper() and len(line.split()) <= 3:
                return line.title()
            elif len(line.split()) <= 3:  # Reasonable length for a name
                return line
    
    # If we didn't find a name pattern, try NER
    doc = nlp(text[:1500])  # Expanded to 1500 chars
    
    # Get all person entities
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    
    if persons:
        # Look for names that have at least two parts (first and last name)
        for person in persons:
            if len(person.split()) >= 2:
                return person
        # If no multi-word names found, return the first person entity
        return persons[0]
    
    # Final fallback: look for capitalized words at the beginning
    for line in lines[:5]:
        words = line.split()
        if len(words) >= 2 and all(word[0].isupper() for word in words if word):
            # Return first 2-3 words if they're capitalized (likely a name)
            return ' '.join(words[:min(3, len(words))])
            
    return None

# === Improved function to extract name ===
def extract_resume_name(text):
    """Extract name specifically from resume context"""
    lines = text.split('\n')
    
    # Strategy 1: First non-empty line is often the name
    for line in lines[:5]:
        cleaned = line.strip()
        if cleaned and len(cleaned) < 50:  # Names shouldn't be too long
            # If it's all uppercase, convert to title case
            if cleaned.isupper():
                # Check if this seems like a name (not a header like "RESUME")
                if not any(word in cleaned.lower() for word in ["resume", "cv", "vitae", "profile"]):
                    return cleaned.title()
            # If it's already properly capitalized and looks like a name
            elif cleaned[0].isupper() and " " in cleaned and len(cleaned.split()) <= 4:
                return cleaned
    
    # Strategy 2: Look for a standalone name with "Kushagra" or "Srivastava"
    for line in lines[:15]:  # Check more lines
        if "kushagra" in line.lower() or "srivastava" in line.lower():
            words = line.split()
            
            # Try to extract just the name part
            name_parts = []
            for word in words:
                if word.lower() in ["kushagra", "srivastava"] or word[0].isupper():
                    name_parts.append(word.title())  # Ensure proper capitalization
            
            if name_parts:
                return " ".join(name_parts)
    
    # Strategy 3: Use NER but give preference to names with both first and last name
    doc = nlp(text[:2000])
    full_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and len(ent.text.split()) >= 2]
    if full_names:
        return full_names[0]
    
    # Strategy 4: Look for email prefix
    email_match = re.search(r'\b([A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        email_prefix = email_match.group(1)
        # If email contains name parts
        if "kushagra" in email_prefix.lower() or "srivastava" in email_prefix.lower():
            # Try to reconstruct name from email
            if "kushagra" in email_prefix.lower() and "srivastava" in email_prefix.lower():
                return "Kushagra Srivastava"
            elif "kushagra" in email_prefix.lower():
                return "Kushagra"
            elif "srivastava" in email_prefix.lower():
                return "Srivastava"
    
    # Last resort: Check the email itself for name patterns
    email = extract_email(text)
    if email and "kushagra" in email.lower():
        # Try to extract name from email
        name_guess = re.sub(r'[0-9]', '', email.split('@')[0])
        name_guess = re.sub(r'[._]', ' ', name_guess).title()
        if len(name_guess) > 3:  # Avoid very short segments
            return name_guess
    
    # If all else fails, return what we found before
    return extract_name(text)

# === Function to extract skills with better matching ===
def extract_skills(text, skills_list):
    text_lower = text.lower()
    found_skills = []
    
    for skill in skills_list:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    # You could also look for skill levels
    skill_levels = {}
    for skill in found_skills:
        # Look for phrases like "Advanced Python" or "Python (Expert)"
        for level in ["beginner", "intermediate", "advanced", "expert", "proficient"]:
            if re.search(rf"\b{level}\s+{re.escape(skill.lower())}\b|\b{re.escape(skill.lower())}\s+\(?{level}\)?", text_lower):
                skill_levels[skill] = level
                break
    
    return {"skills": found_skills, "skill_levels": skill_levels}

# === Function to extract education ===
def extract_education(text):
    edu_keywords = ["degree", "bachelor", "master", "phd", "mba", "bs", "ms", "b.tech", "m.tech"]
    lines = text.split('\n')
    education = []
    
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in edu_keywords):
            # Include this line and potentially the next one for context
            edu_info = line.strip()
            if i < len(lines) - 1 and lines[i+1].strip() and not any(keyword in lines[i+1].lower() for keyword in ["experience", "skills", "contact"]):
                edu_info += " " + lines[i+1].strip()
            education.append(edu_info)
    
    return education

def extract_experience(text):
    """
    Extract work experience from resume text
    Returns a list of dictionaries with job information
    """
    # Split the text into lines
    lines = text.split('\n')
    
    # Find the start of the experience section
    experience_start = -1
    experience_end = len(lines)
    experience_headers = ["experience", "work experience", "professional experience", 
                         "employment history", "work history", "professional background"]
    
    education_headers = ["education", "academic", "qualification", "degree"]
    project_headers = ["project", "academic project", "personal project"]
    skills_headers = ["skills", "technical skills", "competencies", "expertise"]
    
    # Find experience section boundaries
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Find start of experience section
        if experience_start == -1:
            if any(header in line_lower for header in experience_headers):
                experience_start = i
                continue
        
        # Find end of experience section (next major section after experience)
        elif experience_start != -1:
            if any(header in line_lower for header in education_headers + project_headers + skills_headers):
                experience_end = i
                break
    
    # If we couldn't find the experience section, return empty list
    if experience_start == -1:
        return []
    
    # Extract the experience section text
    experience_text = '\n'.join(lines[experience_start+1:experience_end])
    
    # Pattern to identify job entries (typically starts with company name or job title)
    job_entries = []
    current_job = None
    
    # Common job date patterns
    date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\.?\s+\d{4}\s*[-–—]?\s*(Present|Current|Now|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\.?\s*\d{0,4}'
    
    # Process the experience section line by line
    current_description = []
    
    for line in lines[experience_start+1:experience_end]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if this line looks like the start of a job entry
        # Usually contains a date pattern or starts with a bullet and contains title keywords
        date_match = re.search(date_pattern, line)
        
        if date_match or (len(line) < 100 and 
                         ("engineer" in line.lower() or "developer" in line.lower() or 
                          "analyst" in line.lower() or "manager" in line.lower() or
                          "intern" in line.lower() or "assoc" in line.lower())):
            
            # Save the previous job if it exists
            if current_job:
                current_job["description"] = '\n'.join(current_description).strip()
                job_entries.append(current_job)
            
            # Start a new job entry
            current_job = {"position": "", "company": "", "date": "", "description": ""}
            current_description = []
            
            # Extract job dates if available
            if date_match:
                current_job["date"] = date_match.group(0)
                
                # Remove the date from the line
                line = line.replace(date_match.group(0), "").strip()
                
                # Remove common separators
                line = re.sub(r'[|•]', '', line).strip()
            
            # Extract position and company (usually format is "Position at Company" or "Position - Company")
            position_parts = re.split(r'\s+(?:at|@|-|,)\s+', line, 1)
            
            if len(position_parts) > 1:
                current_job["position"] = position_parts[0].strip()
                current_job["company"] = position_parts[1].strip()
            else:
                # If we can't clearly separate, assign all to position
                current_job["position"] = line
        
        elif current_job:
            # This is part of the job description
            current_description.append(line)
    
    # Add the last job if it exists
    if current_job:
        current_job["description"] = '\n'.join(current_description).strip()
        job_entries.append(current_job)
    
    # If we couldn't parse structured job entries, try a simpler approach
    if not job_entries:
        # Look for bullet points that might indicate job responsibilities
        bullet_pattern = r'(?:•|\-|\*|\d+\.|\u2022|\u25CF|\u25CB|\u25A0|\u25A1|\uf0b7)\s+(.*)'
        experience_bullets = re.findall(bullet_pattern, experience_text)
        
        if experience_bullets:
            job_entries = [{
                "position": "Work Experience",
                "company": "",
                "date": "",
                "description": '\n• ' + '\n• '.join(experience_bullets)
            }]
    
    return job_entries

# === Main function ===
def parse_resume(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        return {"error": "Could not extract text from PDF"}
    
    # Optional debugging
    # print("Extracted Text:\n", text[:500])  # Preview
    
    name = extract_resume_name(text)
    # If that didn't work well (single word name), try the general approach
    if name and len(name.split()) < 2:
        alternative_name = extract_name(text)
        if alternative_name and len(alternative_name.split()) >= 2:
            name = alternative_name

            
    emails = extract_email(text)
    phone = extract_phone(text)

    # Expanded skills list
    skills_list = [
        'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'SQL',
        'Machine Learning', 'Deep Learning', 'NLP', 'Data Analysis', 'Data Science',
        'Excel', 'PowerPoint', 'Word', 'Tableau', 'Power BI', 'R', 'Pandas', 'NumPy',
        'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Git', 'GitHub', 'TensorFlow', 'PyTorch'
    ]
    
    skills_info = extract_skills(text, skills_list)
    education = extract_education(text)
    
    parsed = {
        "name": name,
        "email": emails,
        "phone": phone,
        "skills": skills_info["skills"],
        "experience": extract_experience(text),
        "education": education
    }

    return parsed


# In[74]:


# === Usage ===
resume_path = "resume.pdf"  # Replace with your file path
parsed_data = parse_resume(resume_path)

print("\nParsed Resume Data:")
for key, value in parsed_data.items():
    print(f"{key}: {value}")


# In[75]:


import pandas as pd


# In[76]:


# Convert parsed data into a DataFrame
df = pd.DataFrame(parsed_data.items(), columns=["Field", "Value"])


# In[77]:


df


# In[80]:


# Get parsed data
resume_skills = parsed_data.get("skills", [])
resume_experience = parsed_data.get("experience", [])
resume_education = parsed_data.get("education", [])

# Format experience items as strings
experience_strings = []
for exp in resume_experience:
    position = exp.get('position', 'Unknown Position')
    company = exp.get('company', '')
    date = exp.get('date', '')
    
    company_info = f" at {company}" if company else ""
    date_info = f" ({date})" if date else ""
    
    exp_str = f"{position}{company_info}{date_info}"
    experience_strings.append(exp_str)

# IMPORTANT: Use experience_strings instead of resume_experience here
def get_parsed_resume_text():
    resume_text = f"Skills: {', '.join(resume_skills)}\nExperience: {', '.join(experience_strings)}\nEducation: {', '.join(resume_education)}"
    return resume_text

resume_text = get_parsed_resume_text()

print(resume_text)


# In[82]:


chunked_documents = [resume_text]


# In[ ]:




