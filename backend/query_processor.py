"""
Simple query processing and retrieval module for the RAG system.
Handles question answering and source citation without multi-user functionality.
"""
import logging
from typing import List, Dict, Any, Optional
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from config import config
from vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Handles query processing and retrieval-augmented generation."""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = None
        self.qa_chain = None
        self._initialize_llm()
        self._setup_qa_chain()
    
    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            logger.info(f"Initializing LLM: {config.LLM_MODEL_NAME}")
            
            self.llm = ChatGoogleGenerativeAI(
                model=config.LLM_MODEL_NAME,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.1,  # Low temperature for more consistent responses
                max_tokens=1024,  # Reasonable response length
                top_p=0.9,       # Nucleus sampling for better quality
                timeout=45,       # Prevent hanging indefinitely
                max_retries=2,    # Fast failover if issues arise
            )
            
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_qa_chain(self) -> None:
        """Setup the question-answering chain with custom prompt."""
        try:
            # Universal, high-accuracy RAG prompt applicable to any document type and question style
            prompt_template = """You are an intelligent, precise question-answering assistant.
Your task is to answer the user's question accurately and thoroughly based ONLY on the provided context below.

Context:
{context}

Question: {question}

Instructions:
1. Grounding: Answer strictly using the information present in the context. Do not invent facts or extrapolate beyond what is documented.
2. Understanding:
   - For definition/conceptual questions ("what is X", "who is Y"): State what the document defines, describes, or mentions regarding X or Y.
   - For conditional/procedural questions: Summarize the relevant rules, criteria, steps, or exclusions clearly.
   - For indirect questions: Connect relevant pieces of evidence from the context to form a coherent answer.
3. Partial Information: If the context answers only part of the question, provide what is available.
4. Missing Information: Only if the context contains no relevant details or clues to address the question, respond with:
   "This information is not available in the provided documents."
5. Style: Be direct, factual, and concise (2 to 4 sentences). Avoid fluff like "According to the context".

Answer:"""


            
            self.prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            logger.info("QA chain setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            raise
    
    def process_query(self, query: str, max_results: int = 4, score_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Process a query and return the answer with sources.
        
        Args:
            query: The query string
            max_results: Maximum number of source documents to retrieve
            score_threshold: Minimum similarity score threshold (higher = more selective)
            
        Returns:
            Dictionary containing answer and sources
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Enrich query for definition/conceptual questions if needed
            search_query = query
            q_lower = query.lower().strip()
            if any(q_lower.startswith(p) for p in ["who is", "what is", "define", "meaning of", "who are"]):
                # Append definition-oriented terms to capture policy definition clauses
                search_query = f"{query} definition meaning eligible eligibility terms"

            # First try with higher threshold
            search_results = self.vector_store_manager.similarity_search(
                query=search_query,
                k=max_results,
                score_threshold=score_threshold
            )
            
            # If no results with higher threshold, try with lower threshold or original query
            if not search_results:
                logger.info("No results with higher threshold, trying with lower threshold")
                search_results = self.vector_store_manager.similarity_search(
                    query=query,
                    k=max_results,
                    score_threshold=0.05  # Lower fallback threshold
                )
            
            if not search_results:
                logger.warning("No relevant documents found even with lower threshold")
                return {
                    "answer": "This information is not available in the policy documents provided.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Extract documents and scores from the tuples
            relevant_docs = []
            scores = []
            
            for doc, score in search_results:
                relevant_docs.append(doc)
                scores.append(float(score))
            
            logger.info(f"Retrieved {len(relevant_docs)} documents with scores: {scores}")
            
            # Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Generate answer using LLM
            answer = self._generate_answer(query, context)
            
            # Format sources
            sources = self._format_sources(relevant_docs, scores)
            
            # Calculate confidence based on actual similarity scores and threshold used
            avg_score = sum(scores) / len(scores) if scores else 0.0
            confidence = min(0.95, avg_score * 0.7 + (len(relevant_docs) * 0.05))
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "num_sources": len(relevant_docs)
            }
            
            logger.info(f"Query processed successfully with {len(relevant_docs)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from retrieved documents."""
        try:
            context_parts = []
            
            for i, doc in enumerate(documents, 1):
                # Get source information
                source = doc.metadata.get('source', f'Document {i}')
                chunk_id = doc.metadata.get('chunk_id', f'chunk_{i}')
                
                # Format context part
                context_part = f"[Source {i}: {source}, {chunk_id}]\n{doc.page_content}\n"
                context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return ""
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using the LLM."""
        try:
            if not self.llm:
                return "Language model not available"
            
            # Create prompt
            formatted_prompt = self.prompt.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = self.llm.invoke(formatted_prompt)
            
            # Extract text content
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            # Post-process answer for consistency
            answer = self._post_process_answer(answer.strip())
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process the answer to ensure consistency and conciseness."""
        try:
            # Remove common redundant phrases
            redundant_phrases = [
                "Based on the provided context, ",
                "According to the policy documents, ",
                "From the information provided, ",
                "The policy states that ",
                "As per the policy, "
            ]
            
            for phrase in redundant_phrases:
                if answer.startswith(phrase):
                    answer = answer[len(phrase):].strip()
                    # Capitalize first letter after removing prefix
                    if answer and answer[0].islower():
                        answer = answer[0].upper() + answer[1:]
            
            # Ensure the answer ends with proper punctuation
            if answer and not answer.endswith(('.', '!', '?')):
                answer += '.'
            
            # Replace very long sentences with more concise versions
            if len(answer) > 300:
                # Try to find a natural break point
                sentences = answer.split('. ')
                if len(sentences) > 2:
                    # Keep first 2 sentences if they're substantial
                    first_two = '. '.join(sentences[:2])
                    if len(first_two) > 100:  # Ensure we're keeping substantial content
                        answer = first_two + '.'
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in post-processing answer: {e}")
            return answer  # Return original if post-processing fails
    
    def _format_sources(self, documents: List[Document], scores: List[float] = None) -> List[Dict[str, Any]]:
        """Format source documents for response with flat structure for frontend."""
        try:
            sources = []
            
            logger.info(f"Formatting {len(documents)} documents with scores: {scores}")
            
            for i, doc in enumerate(documents):
                # Get the actual similarity score
                score = scores[i] if scores and i < len(scores) else 0.0
                
                # Ensure score is a valid float between 0 and 1
                try:
                    normalized_score = float(score) if score is not None else 0.0
                    # Clamp score between 0 and 1
                    normalized_score = max(0.0, min(1.0, normalized_score))
                except (ValueError, TypeError):
                    normalized_score = 0.0
                
                # Extract source name with fallbacks
                source_name = (
                    doc.metadata.get("source") or 
                    doc.metadata.get("filename") or 
                    doc.metadata.get("file_name") or 
                    f"Document_{i+1}"
                )
                
                # Create content and preview
                content = doc.page_content or ""
                content_preview = content[:200].strip()
                if len(content) > 200:
                    content_preview += "..."
                
                # Create flat structure that matches frontend expectations
                source_info = {
                    "source": source_name,           # Frontend expects this field
                    "filename": source_name,         # Fallback field
                    "score": normalized_score,       # Frontend expects this field (0.0 to 1.0)
                    "content": content,              # Full content
                    "content_preview": content_preview or "No preview available",  # Frontend expects this
                    "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i+1}'),
                    "chunk_index": doc.metadata.get('chunk_index', i),
                    # Keep original metadata as well for compatibility
                    "metadata": doc.metadata or {}
                }
                
                logger.debug(f"Formatted source {i+1}: {source_name}, score: {normalized_score:.3f}")
                sources.append(source_info)
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to format sources: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query processor statistics."""
        try:
            return {
                "llm_model": config.LLM_MODEL_NAME,
                "llm_initialized": self.llm is not None,
                "qa_chain_ready": self.qa_chain is not None,
                "prompt_template_ready": hasattr(self, 'prompt')
            }
            
        except Exception as e:
            logger.error(f"Failed to get query processor stats: {e}")
            return {"error": str(e)}
