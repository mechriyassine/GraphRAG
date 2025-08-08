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
    #  Chunking (keep as is or adapt)
def chunk_text(text, chunk_size=512):
    """Chunk text into smaller pieces for FAISS indexing."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Batch embedding function
def get_embeddings(texts):
    """Get embeddings for a list of texts (batch) using the Gemini API."""
    try:
        response = model.generate_embeddings(texts)  # batch embedding call
        return [embedding.vector for embedding in response.embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

    # Create FAISS index with batch embeddings
def create_faiss_index(chunks):
    """Create a FAISS index for semantic search."""
    embeddings = get_embeddings(chunks)  # Get embeddings for all chunks at once
    if not embeddings:
        raise ValueError("No embeddings generated.")
    
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings_np)  # Add all embeddings to the index
    print(f"Added {index.ntotal} chunks to FAISS index.")
    
    return index, dimension

    #  Search using FAISS
def search_vector(query, index, chunks, k=3):
    """Find most relevant text chunks for a query using FAISS."""
    query_embedding = get_embeddings([query])  # Embed query as a batch of one
    if not query_embedding:
        return []
    
    query_vector = np.array(query_embedding).astype('float32')
    D, I = index.search(query_vector, k)  # Search top k nearest neighbors
    return [chunks[i] for i in I[0]]

    
# def get_embeddings(texts):
#     embed = genai.embed_content(model="models/embedding-011",content=text)
#     return embed["embedding"]
# def create_faiss_index(chunks):
#     """Create a FAISS index for semantic search."""
#     embeddings = np.array([get_embeddings(chunk) for chunk in chunks]).astype('float32')
#     dimension = embeddings.shape[1]
    
#     index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
#     index.add(embeddings)  # Add embeddings to the index
#     print(f"Added {index.ntotal} chunks to FAISS index.")
    
#     return index,dimension

# def search_vector(query, index, chunks, k=3):
#     """Find most relevant text chunks for a query using FAISS."""
#     query_vector = np.array([get_embeddings(query)]).astype('float32')
#     D, I = index.search(query_vector, k)  # Search for k nearest neighbors
#     return [chunks[i] for i in I[0]]

# 7. Graph Query Helpers
def query_graph(query, params=None):
    """Run a Cypher query on the Neo4j database."""
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]
    
def debug_relationships ():
    """Print all relationships in graph inspection."""
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS relation, m.name ORDER BY n.name")
        relationships = [record.data() for record in result]
        
        print(" DEBUG: Relationships in the graph:")
        for rel in relationships:
            print(f"{rel['n.name']} -[{rel['relation']}]-> {rel['m.name']}")
        return relationships

# 8. GraphRag answer generation

def graph_rag_answer(question, index, chunks):
    """
    Combines semantic search + graph relationships + LLM to generate an answer.
    """
    # step 1 : Semantic context
    vector_context = "\n".join(search_vector(question, index, chunks))
    
    # step 2 : Pull a ll relationships
    graph_data= query_graph("MATCH (n)-[r]->(m) RETURN n.name, type(r) AS relation, m.name LIMIT 50")
    graph_context = "\n".join([f"{row['n.name']} -[{row['relation']}]-> {row['m.name']}" for row in graph_data])
    
    # step 3 : Combine contexts
    prompt = f"""
    Use the following graph relationships + context to answer the question:
    
    Context:
    {vector_context}
    
    Graph:
    {graph_context}
    
    Question: {question}
    """
    
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text.strip()

# Main pipeline
def main():
    # 1. Test Neo4j connection
    test_connection ()
    
    # 2. Ingest sample text (replace with real_pdf() for real docs)
    sample_text ="""
    Mohamed manages Ali, who works under Fatima.
    Ali assists Fatima in her projects.
    Fatima supervises both Mohamed and Ali.
    """
    # 3. Extract relationships
    relations = extract_relationships(sample_text)
    for rel in relations:
        add_relationships(rel['person'], rel['relation'], rel['target'])
        
    # 4. Build FAISS index
    chunks = chunk_text(sample_text)
    index, dimension = create_faiss_index(chunks)
    
    # 5. Debug graph
    debug_relationships()
    
    # 6. Ask a question
    answer = graph_rag_answer("Who manages Ali?", index, chunks)
    print(f"Answer: {answer}")
    
if __name__ == "__main__":
    main() # Close Neo4j driver connection