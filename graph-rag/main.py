from neo4j import GraphDatabase
import json, os
import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# LLaMA model setup
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")
llm = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048)  # Adjust n_ctx based on model

# Sentence Transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, local embedding model

# 1. DB check
def check_db():
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) AS count")
        for record in result:
            print(f"Number of nodes in the database: {record['count']}")

# 2. Add relationship
def add_relationships(person, relation, target):
    """Add a relationship between two entities in the Neo4j database."""
    clean_relation = relation.upper().replace(" ", "_").replace("-", "_").replace(".", "")
    clean_relation = ''.join(c for c in clean_relation if c.isalnum() or c == '_')
    
    with driver.session() as session:
        session.run(f"""
            MERGE (p:Entity {{name: $person}})
            MERGE (t:Entity {{name: $target}})
            MERGE (p)-[r:{clean_relation}]->(t)
        """, {'person': person, 'target': target})
        print(f"Added {person} -[{clean_relation}]-> {target}")

# 3. Extract text content from a PDF file
def read_pdf(file_path):
    """Read a PDF file and return its text content."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# 4. Extract relationships (LLM)
def extract_relationships(text):
    """Use LLaMA to extract relationships from the text like:
    [{'person':'Mohamed', 'relation':'MANAGES', 'target':'Ali'}]
    """
    prompt = f"""Extract relationships from the following text. ONLY return a valid JSON array.
    Use simple relations: MANAGES, REPORTS_TO, SUPERVISES, ASSISTS, WORKS_UNDER.
    
    Text:
    {text}
    """
    
    try:
        response = llm(prompt, max_tokens=512, stop=["\n\n"])  # Adjust max_tokens as needed
        raw_text = response['choices'][0]['text'].strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()  # Remove ```json and ``` markers
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# 5. Chunk text for FAISS
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

# 6. Embedding + FAISS index
def get_embeddings(texts):
    """Get embeddings for a list of texts (batch) using SentenceTransformer."""
    try:
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def create_faiss_index(chunks):
    """Create a FAISS index for semantic search."""
    embeddings = get_embeddings(chunks)  # Get embeddings for all chunks at once
    if not embeddings.size:
        raise ValueError("No embeddings generated.")
    
    embeddings_np = embeddings.astype('float32')
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings_np)  # Add all embeddings to the index
    print(f"Added {index.ntotal} chunks to FAISS index.")
    
    return index, dimension

def search_vector(query, index, chunks, k=3):
    """Find most relevant text chunks for a query using FAISS."""
    query_embedding = get_embeddings([query])  # Embed query as a batch of one
    if not query_embedding.size:
        return []
    
    query_vector = query_embedding.astype('float32')
    D, I = index.search(query_vector, k)  # Search top k nearest neighbors
    return [chunks[i] for i in I[0]]

# 7. Graph Query Helpers
def query_graph(query, params=None):
    """Run a Cypher query on the Neo4j database."""
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]

def debug_relationships():
    """Print all relationships in graph inspection."""
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS relation, m.name AS target ORDER BY n.name")
        relationships = [record.data() for record in result]
        
        print("DEBUG: Relationships in the graph:")
        for rel in relationships:
            print(f"{rel['source']} -[{rel['relation']}]-> {rel['target']}")
        return relationships

# 8. GraphRAG answer generation
def graph_rag_answer(question, index, chunks):
    """
    Combines semantic search + graph relationships + LLM to generate an answer.
    """
    # Step 1: Semantic context
    vector_context = "\n".join(search_vector(question, index, chunks))
    
    # Step 2: Pull all relationships
    graph_data = query_graph("MATCH (n)-[r]->(m) RETURN n.name, type(r) AS relation, m.name LIMIT 50")
    graph_context = "\n".join([f"{row['n.name']} -[{row['relation']}]-> {row['m.name']}" for row in graph_data])
    
    # Step 3: Combine contexts
    prompt = f"""
    Use the following graph relationships + context to answer the question:
    
    Context:
    {vector_context}
    
    Graph:
    {graph_context}
    
    Question: {question}
    """
    
    response = llm(prompt, max_tokens=512, stop=["\n\n"])  # Adjust max_tokens as needed
    return response['choices'][0]['text'].strip()

# Main pipeline
def main():
    # 1. Test Neo4j connection
    check_db()
    
    # 2. Ingest sample text
    sample_text = """
    Mohamed manages Ali, who works under Fatima.
    Ali assists Fatima in her projects.
    Fatima supervises both Mohamed and Ali.
    """
    
    # 3. Extract relationships
    relations = extract_relationships(sample_text)
    if relations:
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
    try:
        main()
    finally:
        driver.close()  # Close Neo4j driver connection