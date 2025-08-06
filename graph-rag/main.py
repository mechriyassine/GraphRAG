from neo4j import GraphDatabase
import json, os
import faiss
import numpy as np
import google.generativeai as genai
from pydf import PdfReader
from dotenv import load_dotenv

load_dotenv() # Load environment variables


# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Gemini API setup
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY) 
model = genai.GenerativeModel("gemini-2.5-flash")

# 1. DB chek
def check_db():
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS count")
        for record in result:
            print(f"Number of nodes in the database: {record['count']}")

# 2. Add relationship
def add_relationships(person, relation, target):
    """Add a relationship between two entities in the Neo4j database."""
    clean_relation = relation.upper().replace(" ", "_").replace("-", "_").replace(".", "")
    clean_relation =''.join(c for c in clean_relation if c.isalnum() or c == '_')
    
    with driver.session() as session:
        session.run(f"""
            MERGE (p:Entity {{name: $person}})
            MERGE (t:Entity {{name: $target}})
            MERGE (p)-[r:{clean_relation}]->(t)
        """, {'person': person, 'target': target}
        )
        print(f"Added {person} -[{clean_relation}]-> {target}")
        
        
# 3. Extract text content from a PDF file
def read_pdf(file_path):
    """Read a PDF file and return its text content."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip() # Remove leading and trailing whitespace, including extra newlines
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    
# 4. Extract relationships (LLM)
def extract_relationships(text):
    """Use LLM to extract relationships from the text like:
    [{'person':'Mohamed', 'relation':'MANAGES', and 'target':'Ali'}]
    """
    prompt = f"""Extract relationships from the following text.ONLY return a valid JSON array.
    Use simple relations : MANAGES, REPORTS_TO, SUPERVISES, ASSISTS, WORKS_UNDER.
    
    Text:
    {text}
    """
    
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        if raw_text.startswith("'''"):
            raw_text = raw_text.strip("'").replace("json\n", "")
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# 5. Chunk test fro FAISS
def chunk_text(text, chunk_size=512):
    """Chunk text into smaller pieces for FAISS indexing."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) <= chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
# def chunk_text(text, chunk_size=512):
#     """Chunk text into smaller pieces for FAISS indexing."""
#     return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# 6. Embedding + FAISS index
# def get_embeddings(texts):
#     """Get embeddings for a list of texts using the Gemini API."""
#     try:
#         response = model.generate_embeddings(texts)
#         return [embedding.vector for embedding in response.embeddings]
#     except Exception as e:
#         print(f"Error generating embeddings: {e}")
#         return []
    
def get_embeddings(texts):
    embed = genai.embed_content(model="models/embedding-011",content=text)
    return embed["embedding"]