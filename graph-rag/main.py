from neo4j import GraphDatabase
import json
import os
import faiss
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables

class GraphRAGSystem:
    def __init__(self):
        """Initialize the GraphRAG system with all components."""
        self.setup_neo4j()
        self.setup_llm()
        self.setup_embedder()
        self.index = None
        self.chunks = []
        
    def setup_neo4j(self):
        """Setup Neo4j connection."""
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Neo4j credentials not found in environment variables")
            
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        logger.info("Neo4j connection established")
        
    def setup_llm(self):
        """Setup LLaMA model."""
        llama_model_path = os.getenv("LLAMA_MODEL_PATH")
        if not llama_model_path or not Path(llama_model_path).exists():
            raise ValueError("Valid LLAMA_MODEL_PATH not found")
            
        self.llm = Llama(
            model_path=llama_model_path, 
            n_ctx=4096,  # Increased context window
            n_threads=4,  # Optimize for multi-threading
            verbose=False
        )
        logger.info("LLaMA model loaded successfully")
        
    def setup_embedder(self):
        """Setup sentence transformer for embeddings."""
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer loaded")

    def check_db_health(self) -> Dict:
        """Check Neo4j database health and statistics."""
        with self.driver.session() as session:
            stats = {}
            
            # Node count
            result = session.run("MATCH (n) RETURN count(n) AS node_count")
            stats['nodes'] = result.single()['node_count']
            
            # Relationship count
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
            stats['relationships'] = result.single()['rel_count']
            
            # Node labels
            result = session.run("CALL db.labels()")
            stats['labels'] = [record['label'] for record in result]
            
            # Relationship types
            result = session.run("CALL db.relationshipTypes()")
            stats['relationship_types'] = [record['relationshipType'] for record in result]
            
            logger.info(f"Database stats: {stats}")
            return stats

    def clean_relation_name(self, relation: str) -> str:
        """Clean and normalize relationship names for Neo4j."""
        # Convert to uppercase and replace special characters
        clean_relation = relation.upper().strip()
        clean_relation = re.sub(r'[^\w\s]', '', clean_relation)  # Remove special chars
        clean_relation = re.sub(r'\s+', '_', clean_relation)     # Replace spaces with underscores
        
        # Ensure it starts with a letter or underscore
        if clean_relation and not (clean_relation[0].isalpha() or clean_relation[0] == '_'):
            clean_relation = f"REL_{clean_relation}"
            
        return clean_relation if clean_relation else "RELATED_TO"

    def add_entity_with_properties(self, name: str, properties: Dict = None):
        """Add an entity with optional properties to the graph."""
        properties = properties or {}
        properties['name'] = name
        
        with self.driver.session() as session:
            session.run(
                "MERGE (e:Entity {name: $name}) SET e += $properties",
                {'name': name, 'properties': properties}
            )

    def add_relationship(self, source: str, relation: str, target: str, properties: Dict = None):
        """Add a relationship between two entities with optional properties."""
        clean_relation = self.clean_relation_name(relation)
        properties = properties or {}
        
        with self.driver.session() as session:
            query = f"""
                MERGE (s:Entity {{name: $source}})
                MERGE (t:Entity {{name: $target}})
                MERGE (s)-[r:{clean_relation}]->(t)
                SET r += $properties
                RETURN s.name, type(r), t.name
            """
            result = session.run(query, {
                'source': source, 
                'target': target, 
                'properties': properties
            })
            
            # Verify the relationship was created
            record = result.single()
            if record:
                logger.info(f"✅ Added: {source} -[{clean_relation}]-> {target}")
            else:
                logger.warning(f"⚠️ Failed to add: {source} -[{clean_relation}]-> {target}")
                
        # Double-check by counting relationships
        self._verify_relationship_count()
    
    def _verify_relationship_count(self):
        """Internal method to verify relationships are being saved."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                count = result.single()['count']
                logger.debug(f"Current relationship count: {count}")
        except Exception as e:
            logger.warning(f"Could not verify relationship count: {e}")

    def read_pdf(self, file_path: str) -> Optional[str]:
        """Extract text from PDF with better error handling."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
                
            reader = PdfReader(str(path))
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return None

    def extract_relationships_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[Dict]]:
        """Extract relationships using LLaMA with retry logic."""
        prompt = f"""Extract relationships from the following text. Return ONLY a valid JSON array with this exact format:
[{{"person": "Name1", "relation": "MANAGES", "target": "Name2"}}, {{"person": "Name2", "relation": "REPORTS_TO", "target": "Name3"}}]

Use these relationship types: MANAGES, REPORTS_TO, SUPERVISES, ASSISTS, WORKS_WITH, LEADS, COLLABORATES_WITH, MENTORS, SUPPORTS.

Text:
{text[:1500]}...

JSON:"""
        
        for attempt in range(max_retries):
            try:
                response = self.llm(
                    prompt, 
                    max_tokens=1024,
                    temperature=0.1,  # Low temperature for consistency
                    stop=["Text:", "---", "\n\n\n"]
                )
                
                raw_text = response['choices'][0]['text'].strip()
                
                # Clean up response
                if "```json" in raw_text:
                    json_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
                    if json_match:
                        raw_text = json_match.group(1)
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:].strip()
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3].strip()
                
                # Try to find JSON array in the response
                json_match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    relationships = json.loads(json_str)
                    
                    # Validate structure
                    if isinstance(relationships, list):
                        valid_relationships = []
                        for rel in relationships:
                            if (isinstance(rel, dict) and 
                                all(key in rel for key in ['person', 'relation', 'target'])):
                                valid_relationships.append(rel)
                        
                        if valid_relationships:
                            logger.info(f"Extracted {len(valid_relationships)} relationships on attempt {attempt + 1}")
                            return valid_relationships
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Failed to extract relationships after all retries")
                    
        return None

    def chunk_text_smart(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Intelligently chunk text with overlapping windows."""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence exceeds chunk size, save current chunk
            if len(current_chunk + sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap//10:]) if len(words) > overlap//10 else ""
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings in batches for memory efficiency."""
        if not texts:
            return np.array([])
            
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(
                batch, 
                convert_to_numpy=True, 
                show_progress_bar=True,
                batch_size=batch_size
            )
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)

    def create_faiss_index(self, chunks: List[str]) -> Tuple[faiss.Index, int]:
        """Create and populate FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided for indexing")
            
        embeddings = self.get_embeddings_batch(chunks)
        embeddings_np = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
        dimension = embeddings_np.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)
        
        logger.info(f"Created FAISS index with {index.ntotal} vectors of dimension {dimension}")
        
        self.index = index
        self.chunks = chunks
        
        return index, dimension

    def save_index(self, index_path: str, chunks_path: str):
        """Save FAISS index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to save")
            
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
            
        logger.info(f"Saved index to {index_path} and chunks to {chunks_path}")

    def load_index(self, index_path: str, chunks_path: str):
        """Load FAISS index and chunks from disk."""
        if not Path(index_path).exists() or not Path(chunks_path).exists():
            raise FileNotFoundError("Index or chunks file not found")
            
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
            
        logger.info(f"Loaded index from {index_path} and chunks from {chunks_path}")

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant chunks using semantic similarity."""
        if self.index is None or not self.chunks:
            return []
            
        query_embedding = self.get_embeddings_batch([query])
        query_vector = query_embedding.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
                
        return results

    def query_graph_advanced(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute advanced Cypher queries with error handling."""
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []

    def find_graph_context(self, entities: List[str], max_depth: int = 2) -> str:
        """Find relevant graph context for given entities."""
        if not entities:
            return ""
            
        # Build query to find paths between entities
        entity_conditions = " OR ".join([f"n.name CONTAINS '{entity}'" for entity in entities])
        
        query = f"""
        MATCH (n)-[r*1..{max_depth}]-(m)
        WHERE ({entity_conditions}) OR m.name IN {entities}
        RETURN DISTINCT n.name AS source, type(r[0]) AS relation, m.name AS target
        LIMIT 50
        """
        
        results = self.query_graph_advanced(query)
        
        context_lines = []
        for row in results:
            if all(key in row for key in ['source', 'relation', 'target']):
                context_lines.append(f"{row['source']} -[{row['relation']}]-> {row['target']}")
                
        return "\n".join(context_lines)

    def extract_entities_from_question(self, question: str) -> List[str]:
        """Extract potential entity names from the question."""
        # Simple entity extraction - could be enhanced with NER
        words = re.findall(r'\b[A-Z][a-z]+\b', question)  # Capitalized words
        return list(set(words))

    def generate_answer(self, question: str, max_context_length: int = 2000) -> str:
        """Generate comprehensive answer using GraphRAG approach."""
        # Step 1: Semantic search for relevant context
        semantic_results = self.semantic_search(question, k=3)
        semantic_context = "\n".join([chunk for chunk, _ in semantic_results[:3]])
        
        # Step 2: Extract entities and find graph context
        entities = self.extract_entities_from_question(question)
        graph_context = self.find_graph_context(entities)
        
        # Step 3: Combine contexts, respecting length limits
        combined_context = ""
        if semantic_context:
            combined_context += f"Relevant Content:\n{semantic_context[:max_context_length//2]}\n\n"
        if graph_context:
            combined_context += f"Knowledge Graph:\n{graph_context[:max_context_length//2]}\n\n"
        
        # Step 4: Generate answer
        prompt = f"""Based on the following information, provide a comprehensive and accurate answer to the question.

{combined_context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm(
                prompt, 
                max_tokens=512,
                temperature=0.2,
                stop=["Question:", "\n\n\n"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."

    def process_document(self, file_path: str) -> bool:
        """Process a document end-to-end."""
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        if file_path.lower().endswith('.pdf'):
            text = self.read_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return False
        
        if not text:
            logger.error("No text extracted from document")
            return False
        
        # Extract and add relationships
        relationships = self.extract_relationships_with_retry(text)
        if relationships:
            for rel in relationships:
                self.add_relationship(
                    rel['person'], 
                    rel['relation'], 
                    rel['target']
                )
        
        # Create vector index
        chunks = self.chunk_text_smart(text)
        if chunks:
            self.create_faiss_index(chunks)
            
        logger.info(f"Successfully processed document with {len(chunks)} chunks and {len(relationships or [])} relationships")
        return True

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'driver'):
            self.driver.close()
        logger.info("GraphRAG system closed")

    def process_pdf_folder(self, folder_path: str = "pdfs") -> bool:
        """Process all PDF files in the specified folder."""
        pdf_folder = Path(folder_path)
        
        # Create folder if it doesn't exist
        if not pdf_folder.exists():
            pdf_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created folder: {pdf_folder}")
            print(f"📁 Created folder '{folder_path}' - please add your PDF files there and run again!")
            return False
        
        # Find all PDF files
        pdf_files = list(pdf_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder}")
            print(f"📁 No PDF files found in '{folder_path}' folder")
            print(f"💡 Please add some PDF files to the '{folder_path}' folder and run again!")
            return False
        
        print(f"📚 Found {len(pdf_files)} PDF files:")
        for pdf_file in pdf_files:
            print(f"  • {pdf_file.name}")
        
        # Process each PDF
        processed_count = 0
        all_chunks = []
        
        for pdf_file in pdf_files:
            print(f"\n🔄 Processing: {pdf_file.name}")
            
            # Extract text from PDF
            text = self.read_pdf(str(pdf_file))
            if not text:
                print(f"❌ Could not extract text from {pdf_file.name}")
                continue
            
            print(f"📄 Extracted {len(text)} characters from {pdf_file.name}")
            
            # Extract relationships
            relationships = self.extract_relationships_with_retry(text)
            if relationships:
                print(f"🔗 Found {len(relationships)} relationships:")
                for rel in relationships:
                    print(f"  • {rel['person']} --{rel['relation']}--> {rel['target']}")
                    self.add_relationship(rel['person'], rel['relation'], rel['target'])
            else:
                print("⚠️ No relationships found in this document")
            
            # Create chunks for this document
            chunks = self.chunk_text_smart(text)
            all_chunks.extend(chunks)
            processed_count += 1
            
            print(f"✅ Processed {pdf_file.name} - {len(chunks)} chunks created")
        
        # Create combined vector index from all documents
        if all_chunks:
            print(f"\n🔗 Creating combined vector index from {len(all_chunks)} chunks...")
            self.create_faiss_index(all_chunks)
            print("✅ Vector index created successfully!")
        
        print(f"\n📊 Processing complete:")
        print(f"  • Processed: {processed_count}/{len(pdf_files)} PDF files")
        print(f"  • Total chunks: {len(all_chunks)}")
        
        return processed_count > 0

# Example usage
def main():
    system = GraphRAGSystem()
    
    try:
        # Check database health
        stats = system.check_db_health()
        print(f"📊 Database Statistics: {stats}")
        
        # Clear existing data (optional - remove if you want to keep accumulating data)
        clear_existing = input("\n🗑️  Clear existing data? (y/N): ").lower().strip() == 'y'
        if clear_existing:
            with system.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Cleared existing data")
        
        # Process all PDFs in the "pdfs" folder
        print("\n🚀 Processing PDF files...")
        success = system.process_pdf_folder("pdfs")  # You can change "pdfs" to any folder name
        
        if success:
            # Check what we have in the database now
            final_stats = system.check_db_health()
            print(f"\n📊 Final Database Statistics: {final_stats}")
            
            # Save the knowledge base
            system.save_index("pdf_knowledge.faiss", "pdf_chunks.pkl")
            print("💾 Saved knowledge base to pdf_knowledge.faiss and pdf_chunks.pkl")
            
            # Interactive Q&A session
            print(f"\n🤖 Ready for questions! (Type 'quit' to exit)")
            while True:
                question = input("\n❓ Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if question:
                    print("🔍 Searching...")
                    answer = system.generate_answer(question)
                    print(f"💡 Answer: {answer}")
        
        else:
            print("❌ No PDFs processed. Please add PDF files to the 'pdfs' folder.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
    finally:
        system.close()

if __name__ == "__main__":
    main()