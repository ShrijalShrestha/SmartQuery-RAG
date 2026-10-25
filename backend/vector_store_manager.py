"""
Simple vector store management module for embedding and vector database operations.
Handles Pinecone integration with hybrid search (dense + sparse vectors).
"""
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
import uuid

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

from config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store operations including embedding and Pinecone integration with hybrid search."""
    
    def __init__(self):
        # Dense embeddings using HuggingFace through LangChain
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Sparse embeddings for hybrid search
        self.bm25_encoder = BM25Encoder().default()
        
        self.pinecone_client = None
        self.pinecone_index = None
        self.hybrid_retriever = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize embedding model, Pinecone client, and hybrid retriever."""
        try:
            # Initialize Pinecone client
            logger.info(f"Initializing Pinecone client for region: {config.PINECONE_REGION}")
            self.pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
            
            # Connect to or create the index
            self._ensure_index_exists()
            self.pinecone_index = self.pinecone_client.Index(config.PINECONE_INDEX_NAME)
            
            # Initialize hybrid retriever
            self._initialize_hybrid_retriever()
            
            logger.info("Vector store manager with hybrid search initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store manager: {e}")
            raise
    
    def _initialize_hybrid_retriever(self):
        """Initialize the hybrid retriever after index is ready."""
        try:
            self.hybrid_retriever = PineconeHybridSearchRetriever(
                embeddings=self.embedding_model,
                sparse_encoder=self.bm25_encoder,
                index=self.pinecone_index,
                top_k=15,
                alpha=0.3  # Favor dense semantic search (0.3 dense vs 0.7 sparse) for conceptual matches
            )
            logger.info("Hybrid retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {e}")
            # Fallback to dense-only if hybrid fails
            self.hybrid_retriever = None
    
    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists, create if it doesn't."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if config.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating Pinecone index: {config.PINECONE_INDEX_NAME}")
                # Create serverless spec with proper region handling
                spec = ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION
                )
                
                self.pinecone_client.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=config.EMBEDDING_DIMENSION,
                    metric=config.PINECONE_METRIC,
                    spec=spec
                )
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(10)
                logger.info("Index created successfully")
            else:
                logger.info(f"Index '{config.PINECONE_INDEX_NAME}' already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            # Log more details about the error
            if "Malformed domain" in str(e):
                logger.error(f"Pinecone region issue. Check PINECONE_REGION='{config.PINECONE_REGION}' and PINECONE_API_KEY")
                logger.error("Common valid regions: us-east-1, us-west-2, eu-west-1, asia-southeast-1")
            raise
    
    def get_existing_vector_store(self) -> bool:
        """
        Check if a vector store already exists and has vectors.
        
        Returns:
            True if exists and has vectors, False otherwise
        """
        try:
            # Check if index exists and has vectors
            index_stats = self.pinecone_index.describe_index_stats()
            total_vectors = index_stats.get('total_vector_count', 0)
            
            if total_vectors > 0:
                logger.info(f"Found existing vector store with {total_vectors} vectors")
                return True
            else:
                logger.info("Vector store exists but is empty")
                return False
                
        except Exception as e:
            logger.error(f"Error checking existing vector store: {e}")
            return False
    
    def fit_bm25_encoder(self, documents: List[Document]):
        """Fit the BM25 encoder on the document corpus for better sparse representations."""
        try:
            logger.info("Fitting BM25 encoder on document corpus...")
            texts = [doc.page_content for doc in documents]
            self.bm25_encoder.fit(texts)
            logger.info("BM25 encoder fitted successfully")
        except Exception as e:
            logger.warning(f"Failed to fit BM25 encoder: {e}")
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Create vectors from documents and store them in Pinecone with hybrid search support.
        
        Args:
            documents: List of Document objects to embed and store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Creating vector store with {len(documents)} documents using hybrid approach")
            
            # Fit BM25 encoder on the document corpus
            self.fit_bm25_encoder(documents)
            
            # Use hybrid retriever to add documents if available
            if self.hybrid_retriever:
                logger.info("Adding documents using hybrid retriever...")
                # The hybrid retriever handles both dense and sparse embeddings internally
                self.hybrid_retriever.add_texts(
                    texts=[doc.page_content for doc in documents],
                    metadatas=[doc.metadata for doc in documents]
                )
                logger.info("Documents added successfully using hybrid approach")
                
                # Wait a bit for eventual consistency
                logger.info("Waiting for index consistency...")
                time.sleep(3)
                
                # Verify vectors were added
                total_vectors = self.get_total_vector_count()
                if total_vectors > 0:
                    logger.info(f"Vector store created successfully with {total_vectors} vectors")
                    return True
                else:
                    logger.warning("No vectors found after upsert, falling back to manual approach")
                    return self._create_vector_store_manual(documents)
            else:
                # Fallback to manual dense-only approach
                logger.warning("Hybrid retriever not available, using dense-only approach")
                return self._create_vector_store_manual(documents)
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            # Try manual approach as fallback
            return self._create_vector_store_manual(documents)
    
    def _create_vector_store_manual(self, documents: List[Document]) -> bool:
        """Manual vector store creation (fallback method)."""
        try:
            logger.info(f"Creating vector store manually with {len(documents)} documents")
            
            # Prepare data for upsert
            vectors_to_upsert = []
            
            for i, doc in enumerate(documents):
                # Generate embedding using HuggingFaceEmbeddings
                embedding = self.embedding_model.embed_query(doc.page_content)
                
                # Create unique ID
                vector_id = str(uuid.uuid4())
                
                # Prepare metadata
                metadata = {
                    'text': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', f'chunk_{i}'),
                    'chunk_index': doc.metadata.get('chunk_index', i)
                }
                
                # Add other metadata if available
                for key, value in doc.metadata.items():
                    if key not in ['text', 'source', 'chunk_id', 'chunk_index']:
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                self.pinecone_index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
            
            logger.info("Vector store created successfully")
            
            # Wait for eventual consistency
            self._wait_for_index_readiness(len(documents))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector store manually: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Tuple[Document, float]]:
        """
        Perform hybrid similarity search in the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            logger.info(f"Performing hybrid similarity search for: {query[:50]}...")

            # Use hybrid retriever if available
            if self.hybrid_retriever:
                return self._hybrid_search(query, k, score_threshold)
            else:
                # Fallback to dense-only search
                logger.warning("Hybrid retriever not available, using dense-only search")
                return self._dense_only_search(query, k, score_threshold)

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def _hybrid_search(self, query: str, k: int, score_threshold: float) -> List[Tuple[Document, float]]:
        """Perform hybrid search using the hybrid retriever."""
        try:
            # Use the hybrid retriever for better results
            documents = self.hybrid_retriever.get_relevant_documents(query)
            
            # Filter by score threshold if needed and add scores
            results = []
            for doc in documents[:k]:
                # Ensure page_content is populated from metadata if empty
                if not doc.page_content or not doc.page_content.strip():
                    doc.page_content = doc.metadata.get('text') or doc.metadata.get('context') or ""
                
                # Note: LangChain retrievers don't always return scores
                score = getattr(doc, 'score', None) or doc.metadata.get('score', 0.8)
                
                if float(score) >= float(score_threshold):
                    results.append((doc, float(score)))
            
            logger.info(f"Hybrid search found {len(results)} results above threshold {score_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._dense_only_search(query, k, score_threshold)
    
    def _dense_only_search(self, query: str, k: int, score_threshold: float) -> List[Tuple[Document, float]]:
        """Fallback dense-only search method."""
        try:
            logger.info(f"Performing dense-only similarity search for: {query[:50]}...")

            # Generate query embedding using HuggingFaceEmbeddings
            query_embedding = self.embedding_model.embed_query(query)

            # Retry with backoff to allow for eventual consistency after upserts
            attempts = 5
            backoff = 1.0
            last_documents: List[Tuple[Document, float]] = []

            for attempt in range(1, attempts + 1):
                # Query Pinecone
                results = self.pinecone_index.query(
                    vector=query_embedding,
                    top_k=k,
                    include_metadata=True,
                )

                # Handle response for both dict-like and attr-like access
                matches = []
                try:
                    if isinstance(results, dict):
                        matches = results.get('matches', [])
                    else:
                        matches = getattr(results, 'matches', []) or []
                except Exception:
                    matches = []

                # Convert results to Document objects
                documents: List[Tuple[Document, float]] = []
                for m in matches:
                    try:
                        score = m.get('score') if isinstance(m, dict) else getattr(m, 'score', 0.0)
                        if score is None:
                            score = 0.0
                        if float(score) < float(score_threshold):
                            continue

                        metadata = m.get('metadata') if isinstance(m, dict) else getattr(m, 'metadata', {})
                        metadata = metadata or {}
                        # Extract text from metadata (stored during upsert)
                        text = metadata.pop('text', '')

                        doc = Document(
                            page_content=text,
                            metadata=metadata,
                        )
                        documents.append((doc, float(score)))
                    except Exception:
                        continue

                if documents:
                    logger.info(f"Found {len(documents)} results above threshold {score_threshold}")
                    return documents

                last_documents = documents
                if attempt < attempts:
                    time.sleep(backoff)
                    backoff *= 1.5  # exponential backoff

            logger.info(f"Found 0 results above threshold {score_threshold} after {attempts} attempts")
            return last_documents

        except Exception as e:
            logger.error(f"Dense-only similarity search failed: {e}")
            return []
    
    def _wait_for_index_readiness(self, expected_docs: int, max_wait: int = 30):
        """Wait for index to be ready after upserts."""
        logger.info("Waiting for index readiness...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                stats = self.pinecone_index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                
                if total_vectors >= expected_docs:
                    logger.info(f"Index ready with {total_vectors} vectors")
                    return True
                    
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Error checking index readiness: {e}")
                time.sleep(2)
        
        logger.warning(f"Index readiness timeout after {max_wait}s")
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            if not self.pinecone_index:
                return {"error": "Pinecone index not initialized"}
            
            # Get index statistics
            index_stats = self.pinecone_index.describe_index_stats()
            
            return {
                "index_name": config.PINECONE_INDEX_NAME,
                "total_vectors": index_stats.get('total_vector_count', 0),
                "dimension": index_stats.get('dimension', 0),
                "embedding_model": config.EMBEDDING_MODEL_NAME,
                "index_fullness": index_stats.get('index_fullness', 0.0),
                "namespaces": index_stats.get('namespaces', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {"error": str(e)}
    
    def check_vector_store_health(self) -> bool:
        """Check if the vector store is healthy and accessible."""
        try:
            if not self.pinecone_index:
                return False
            
            # Try to get index stats
            stats = self.pinecone_index.describe_index_stats()
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
    
    def get_total_vector_count(self) -> int:
        """Get the total number of vectors in the index."""
        try:
            if not self.pinecone_index:
                return 0
            
            index_stats = self.pinecone_index.describe_index_stats()
            return index_stats.get('total_vector_count', 0)
            
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0
    
    def reset_index(self) -> bool:
        """Reset (clear) the entire vector index."""
        try:
            logger.warning("Resetting vector index - all data will be deleted!")
            
            # Delete all vectors in the index
            self.pinecone_index.delete(delete_all=True)
            
            logger.info("Vector index reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset vector index: {e}")
            return False
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including performance metrics."""
        try:
            start_time = time.time()
            
            # Basic stats
            stats = self.get_stats()
            
            # Add performance metrics
            stats.update({
                "health_check_time": time.time() - start_time,
                "embeddings_initialized": self.embedding_model is not None,
                "pinecone_connected": self.pinecone_index is not None
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get detailed stats: {e}")
            return {"error": str(e)}
