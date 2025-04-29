#!/usr/bin/env python
# coding: utf-8

# In[1]:
import subprocess
import sys
import pinecone

try:
    import pinecone_integration
except ImportError:
    print("Pinecone not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pinecone"])

# In[2]:


from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os


# In[5]:


load_dotenv()


# In[8]:





# In[11]:


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_cloud = os.getenv("PINECONE_CLOUD")  # usually "aws" or "gcp"
pinecone_region = os.getenv("PINECONE_REGION")  # like "us-west-2"


# In[12]:
from pinecone import Pinecone, ServerlessSpec

# ✅ Initialize Pinecone client correctly
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# In[14]:

index_name = "career-paths"

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # or whatever dimension you're using
        metric="cosine",  # or "euclidean" / "dotproduct"
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )



# In[15]:

from langchain.vectorstores import Pinecone


# In[16]:


# using openai embedding to convert job roles into vector and store in pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# In[17]:


embeddings = OpenAIEmbeddings()


# In[18]:


import pandas as pd


# In[19]:


# loading the dataset
df = pd.read_csv("cleaned_job_skills.csv")


# In[20]:


df.head()


# In[24]:


documents = df["Job_Role"]+": "+ df["Skills/Description"]+": "+df["Company"]


# Our documents payload is too large for Pinecone's input size limits.So, we will do chunking to break the size of
# document in order to fit it in pinecone

# In[28]:


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# chunk each document
chunked_documents = []
for doc in documents:
    chunks = text_splitter.split_text(doc)
    chunked_documents.extend(chunks)


# In[32]:


from tqdm import tqdm
import time

# Create the vector store object from the existing index.
vector_store = PineconeVectorStore.from_existing_index(
    index_name="career-paths",
    embedding=embeddings
)

batch_size = 50  # Adjust as needed
for i in tqdm(range(0, len(chunked_documents), batch_size)):
    batch = chunked_documents[i:i + batch_size]
    for attempt in range(3):  # Retry up to 3 times
        try:
            vector_store.add_texts(batch)
            break  # Success; exit the retry loop for this batch.
        except Exception as e:
            print(f"⚠️ Batch {i}-{i+batch_size} failed on attempt {attempt + 1}: {e}")
            time.sleep(5)  # Wait before retrying.
    else:
        print(f"❌ Batch {i}-{i+batch_size} failed after 3 retries.")


# In[35]:


import nbformat


# In[36]:


with open('resume_parse.ipynb', 'r') as f:
    notebook_content = nbformat.read(f, as_version=4)


# In[37]:
from resume_parse import get_parsed_resume_text

resume_text = get_parsed_resume_text()

query = resume_text


# In[38]:


results = vector_store.similarity_search(query, k=3)  # Get top 3 matches


# In[39]:


# Extract job roles
job_roles = [result.page_content.split(":")[0] for result in results]
print("Recommended Job Roles:", job_roles)


# #### Now we will set up our llm in order to generate roadmap to qualify for these jobs

# In[40]:


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# In[41]:


# Initialize LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")


# In[42]:


# Define prompt
prompt_template = PromptTemplate(
    template="""
    Based on the following resume:
    {resume}

    And the recommended job roles: {job_roles}

    Generate a personalized learning roadmap to help the user transition to one of these roles.
    Include specific courses, certifications, and projects they should pursue.
    """,
    input_variables=["resume", "job_roles"]
)


# In[43]:


# Create chain
chain = prompt_template | llm


# In[44]:


# Run the chain
roadmap = chain.invoke({"resume": resume_text, "job_roles": ", ".join(job_roles)})
print("Learning Roadmap:\n", roadmap.content)


# In[ ]:




# In[ ]:




