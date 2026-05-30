"""
RAG (Retrieval-Augmented Generation) Pipeline Implementation
Combines vector search with LLM for intelligent question answering
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for question answering
    
    FLOW:
    =====
    Step 1: Query Embedding
        Input: User question
        Process: Convert query to embedding vector
        Output: Query embedding
    
    Step 2: Vector Search/Retrieval
        Input: Query embedding
        Process: Search vector store for similar documents
        Output: Top-K relevant document chunks with scores
    
    Step 3: Context Construction
        Input: Retrieved documents
        Process: Format context with metadata
        Output: Structured context
    
    Step 4: Prompt Construction
        Input: Query + Context
        Process: Build prompt for LLM
        Output: Formatted prompt
    
    Step 5: LLM Generation
        Input: Prompt with context
        Process: Generate answer using LLM
        Output: Answer text
    
    Result: Question with sources and confidence
    """
    
    def __init__(self, vector_store, embedding_model, llm_service):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Vector store instance for retrieval
            embedding_model: Embedding model for query encoding
            llm_service: LLM service for answer generation
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_service = llm_service
    
    # ============= STEP 1 & 2: RETRIEVE CONTEXT =============
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Step 1 & 2: Embed query and retrieve similar documents
        
        Args:
            query: User question/query
            top_k: Number of documents to retrieve (default: 3 for context)
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        logger.info(f"Step 1-2: Retrieving top {top_k} relevant documents for query")
        logger.info(f"Query: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed(query)
            logger.info(f"✓ Query embedded into {len(query_embedding)}-dimensional vector")
            
            # Search vector store for similar documents
            results = self.vector_store.search(query, top_k)
            logger.info(f"✓ Retrieved {len(results)} relevant documents")
            
            # Log relevance scores
            for i, result in enumerate(results, 1):
                score = result.get('score', 0)
                filename = result.get('filename', 'Unknown')
                logger.info(f"  {i}. {filename} (similarity: {score:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    # ============= STEP 3: CONSTRUCT CONTEXT =============
    def construct_context(self, context_docs: List[Dict]) -> str:
        """
        Step 3: Format retrieved documents into context
        
        Args:
            context_docs: Retrieved document chunks
            
        Returns:
            Formatted context string
        """
        logger.info(f"Step 3: Constructing context from {len(context_docs)} documents")
        
        if not context_docs:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            filename = doc.get('filename', 'Unknown')
            chunk_index = doc.get('chunk_index', 0)
            text = doc.get('text', doc.get('chunk_text', ''))
            score = doc.get('score', 0)
            
            context_parts.append(f"""
Source {i}: {filename} (Chunk {chunk_index}) - Relevance: {score:.1%}
---
{text}
""")
        
        context_text = "\n".join(context_parts)
        logger.info(f"✓ Context constructed ({len(context_text)} characters)")
        
        return context_text
    
    # ============= STEP 4: CONSTRUCT PROMPT =============
    def construct_prompt(self, query: str, context: str) -> str:
        """
        Step 4: Build LLM prompt with context
        
        Prompt structure:
        - System: Define role and context awareness
        - Context: Retrieved relevant documents
        - Instruction: How to answer
        - Question: User query
        
        Args:
            query: User question
            context: Formatted context from documents
            
        Returns:
            Formatted prompt for LLM
        """
        logger.info("Step 4: Constructing LLM prompt")
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

INSTRUCTIONS:
- Use ONLY the information from the provided context to answer the question
- If the context doesn't contain the answer, say "I cannot find this information in the provided documents"
- Include sources when referencing specific information
- Be concise and clear

CONTEXT FROM DOCUMENTS:
{context}

QUESTION:
{query}

ANSWER:"""
        
        logger.info(f"✓ Prompt constructed ({len(prompt)} characters)")
        
        return prompt
    
    # ============= STEP 5: GENERATE ANSWER =============
    def generate_answer(self, prompt: str) -> str:
        """
        Step 5: Generate answer using LLM
        
        Args:
            prompt: Formatted prompt with context
            
        Returns:
            LLM generated answer
        """
        logger.info("Step 5: Generating answer from LLM")
        
        try:
            answer = self.llm_service.generate(prompt)
            logger.info(f"✓ Answer generated ({len(answer)} characters)")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    # ============= COMPLETE RAG PIPELINE =============
    def answer_question(self, query: str, top_k: int = 3) -> Dict:
        """
        Complete RAG Pipeline: Question → Answer with Sources
        
        Executes all 5 steps in sequence
        
        Args:
            query: User question
            top_k: Number of context documents to retrieve
            
        Returns:
            Dictionary with:
            - question: Original query
            - answer: Generated answer
            - sources: List of source documents
            - confidence: Confidence score
            - timestamp: When answer was generated
        """
        logger.info("=" * 70)
        logger.info("STARTING RAG PIPELINE")
        logger.info("=" * 70)
        
        try:
            # Step 1 & 2: Retrieve context
            context_docs = self.retrieve_context(query, top_k)
            
            if not context_docs:
                logger.warning("No relevant documents found")
                return {
                    "success": False,
                    "query": query,
                    "answer": "No relevant documents found to answer this question.",
                    "sources": [],
                    "confidence": 0.0,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            
            # Step 3: Construct context
            context = self.construct_context(context_docs)
            
            # Step 4: Construct prompt
            prompt = self.construct_prompt(query, context)
            
            # Step 5: Generate answer
            answer = self.generate_answer(prompt)
            
            # Extract sources
            sources = [
                {
                    "document_id": doc.get("document_id"),
                    "filename": doc.get("filename"),
                    "chunk_index": doc.get("chunk_index"),
                    "relevance_score": doc.get("score", 0),
                }
                for doc in context_docs
            ]
            
            # Calculate confidence based on relevance scores
            confidence = sum(doc.get("score", 0) for doc in context_docs) / len(context_docs) if context_docs else 0
            
            result = {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": sources,
                "confidence": min(confidence, 1.0),  # Ensure confidence is between 0 and 1
                "num_sources": len(sources),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            logger.info("=" * 70)
            logger.info("✓ RAG PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Confidence: {result['confidence']:.1%}")
            logger.info(f"Sources used: {len(sources)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    # ============= BATCH QUESTION ANSWERING =============
    def answer_multiple_questions(self, questions: List[str], top_k: int = 3) -> List[Dict]:
        """
        Answer multiple questions in batch
        
        Args:
            questions: List of questions
            top_k: Number of context documents per question
            
        Returns:
            List of answers
        """
        logger.info(f"Answering {len(questions)} questions in batch")
        
        answers = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Question {i}/{len(questions)}")
            answer = self.answer_question(question, top_k)
            answers.append(answer)
        
        return answers
    
    # ============= RETRIEVAL ONLY (for debugging) =============
    def retrieve_only(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve documents without generating answer (for debugging/analysis)
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieved documents
        """
        logger.info("Retrieving documents (no answer generation)")
        
        context_docs = self.retrieve_context(query, top_k)
        
        return {
            "query": query,
            "num_results": len(context_docs),
            "results": context_docs,
        }
