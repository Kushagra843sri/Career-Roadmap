import streamlit as st
from resume_parse import parse_resume  # Assuming this is a custom module for resume parsing
from pinecone_integration import vector_store  # Assuming this is a custom module for Pinecone integration
from pinecone_integration import chain  # Assuming this is a custom chain for generating career advice or learning roadmap

# Initialize Streamlit app
st.title("Personalized Career Path Generator")

# Upload PDF resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    # Save uploaded resume to a temporary file
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Parse the resume
    resume_data = parse_resume("temp_resume.pdf")

    # Extract relevant information directly from the dictionary
    skills = resume_data.get('skills', [])
    experience = resume_data.get('experience', [])
    education = resume_data.get('education', [])

    # Convert each item to string if it's a dict or extract specific field like 'name' or 'description' if available
    skills_text = ', '.join([skill if isinstance(skill, str) else str(skill) for skill in skills])
    experience_text = ', '.join([exp if isinstance(exp, str) else str(exp) for exp in experience])
    education_text = ', '.join([edu if isinstance(edu, str) else str(edu) for edu in education])
    
    # Construct resume text using the extracted data
    resume_text = f"Skills: {skills_text}\nExperience: {experience_text}\nEducation: {education_text}"
    
    # Perform similarity search on the resume text (Assuming the `vector_store` is already set up)
    results = vector_store.similarity_search(resume_text, k=3)
    
    # Get job roles from the search results
    job_roles = [result.page_content.split(":")[0] for result in results]
    
    # Generate a learning roadmap based on the resume and job roles
    roadmap = chain.invoke({"resume": resume_text, "job_roles": ", ".join(job_roles)})
    
    # Display the results in Streamlit
    st.write("Recommended Job Roles:", job_roles)
    st.write("Learning Roadmap:", roadmap.content)
