import PyPDF2
import re

import PyPDF2
import io

def parse_pdf_to_text(uploaded_file):
    """
    Parse the PDF from a Streamlit UploadedFile object and extract its text.
    """
    text = []
    
    # Read the file-like object directly from Streamlit
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
    
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)

def create_chunks(text, chunk_size=800, chunk_overlap=100):
    """
    Split the text into overlapping chunks of specified size.
    """
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunk_str = " ".join(chunk)
        chunks.append(chunk_str)
        start += (chunk_size - chunk_overlap)
    return chunks

import torch
from transformers import AutoTokenizer, AutoModel

class SimpleEmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_text(self, texts):
        """
        Given a list of texts, return a list of embedding vectors (one per text).
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            outputs = self.model(**inputs)
            # "Pool" the token embeddings (mean-pool)
            # shape of outputs.last_hidden_state -> [batch_size, seq_len, hidden_dim]
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs.attention_mask

            # Expand mask so that we can do a proper mean
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # Sum all token embeddings
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            # Count how many tokens
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            # Mean pooling
            embeddings = sum_embeddings / sum_mask
            
            # Convert to numpy
            embeddings = embeddings.cpu().numpy()
        return embeddings


import numpy as np

def normalize_vector(v):
    return v / (np.linalg.norm(v) + 1e-10)

def cosine_similarity(vec_a, vec_b):
    """
    Compute the cosine similarity between two vectors.
    """
    vec_a = normalize_vector(vec_a)
    vec_b = normalize_vector(vec_b)
    return np.dot(vec_a, vec_b)

class PDFIndexer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index_data = []  # list of (chunk_text, embedding)

    def index_chunks(self, chunks):
        embeddings = self.embedding_model.embed_text(chunks)
        # store them for later retrieval
        self.index_data = [
            (chunk, embeddings[i]) 
            for i, chunk in enumerate(chunks)
        ]

    def retrieve(self, query, top_k=3):
        """
        Given a user query, embed it, compute similarity vs each chunk.
        Return the top_k chunks with highest similarity.
        """
        query_embedding = self.embedding_model.embed_text([query])[0]
        # compute sim with each chunk
        scored = []
        for (chunk_text, chunk_emb) in self.index_data:
            sim = cosine_similarity(query_embedding, chunk_emb)
            scored.append((chunk_text, sim))
        # sort by sim, descending
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        return scored[:top_k]


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleGenerator:
    def __init__(self, model_name="facebook/opt-1.3b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_answer(self, query, retrieved_texts, max_new_tokens=100):
        """
        Generate an answer given a user query and retrieved text chunks.
        We cap the total generated token length to avoid hanging.
        """
        # Combine retrieved chunks into context
        context = "\n\n".join([f"Chunk {i+1}:\n{txt}" for i, txt in enumerate(retrieved_texts)])
        prompt = (
            f"Here are some relevant excerpts from a PDF:\n"
            f"{context}\n\n"
            f"Using the above information, summarize the context and answer the question:\n{query}\n\n"
            f"Answer:"
        )

        # Tokenize the prompt; limit to a max of 256 tokens to ensure brevity
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        )
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_length = inputs['input_ids'].shape[1]

        # Compute total max length (prompt + new tokens), capped at 512 tokens
        total_length = min(prompt_length + max_new_tokens, 512)

        # Print debugging info (optional)
        print(f"Prompt length: {prompt_length}, Total max length for generation: {total_length}")

        # Generate the answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=total_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output if it is present
        if generated_text.startswith(prompt):
            final_answer = generated_text[len(prompt):].strip()
        else:
            final_answer = generated_text.strip()

        return final_answer

def evaluate_with_judgement(queries):
    """
    Evaluate the RAG system by showing for each query:
      - The retrieved text chunks.
      - The final generated answer.
    Allows a human judge to mark whether the answer is helpful.
    
    Returns a list of dictionaries containing the query, retrieved chunks,
    generated answer, and the human judgement.
    """
    results = []
    # Ensure that the index and generator are available
    if 'indexer' not in st.session_state or st.session_state['indexer'] is None:
        st.error("Please process the PDF and build the index first!")
        return results
    if 'generator_model' not in st.session_state:
        st.session_state['generator_model'] = SimpleGenerator("facebook/opt-350m")
    
    for query in queries:
        st.markdown(f"### Query: {query}")
        
        # Retrieve the top-k chunks
        retrieved = st.session_state['indexer'].retrieve(query, top_k=3)
        retrieved_chunks = [item[0] for item in retrieved]
        
        st.markdown("**Retrieved Excerpts:**")
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(f"**Chunk {idx}:** {chunk}")
        
        # Generate answer using the retrieved context
        generated_answer = st.session_state['generator_model'].generate_answer(query, retrieved_chunks)
        st.markdown("**Generated Answer:**")
        st.markdown(generated_answer)
        
        # Provide a human judgement option
        judgement = st.radio("Is this answer helpful?", ("Yes", "No"), key=query)
        results.append({
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": generated_answer,
            "judgement": judgement,
        })
        st.write("---")
    return results



import streamlit as st

def main():
    st.title("RAG Demo from Scratch")
    
    # 1) File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        pdf_text = parse_pdf_to_text(uploaded_file)
    else:
        st.write("Please upload a PDF file.")
        pdf_text = ""

    # 2) Model/Indexer placeholders
    if 'embedding_model' not in st.session_state:
        st.session_state['embedding_model'] = SimpleEmbeddingModel(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    if 'indexer' not in st.session_state:
        st.session_state['indexer'] = None

    if pdf_text:
        # Create a button to build the index
        if st.button("Process PDF and Build index"):
            chunks = create_chunks(pdf_text, chunk_size=800, chunk_overlap=100)
            st.session_state['indexer'] = PDFIndexer(st.session_state['embedding_model'])
            st.session_state['indexer'].index_chunks(chunks)
            st.success("Index built successfully!")

    # 3) Question/Answer
    if st.session_state.get('indexer', None) is not None:
        user_query = st.text_input("Ask a question about the PDF")
        if user_query:
            if 'generator_model' not in st.session_state:
                st.session_state['generator_model'] = SimpleGenerator("facebook/opt-350m")
            
            retrieved = st.session_state['indexer'].retrieve(user_query, top_k=3)
            retrieved_chunks = [item[0] for item in retrieved]
            
            answer = st.session_state['generator_model'].generate_answer(
                user_query, retrieved_chunks
            )
            st.markdown("**Answer:** " + answer)
    
    # 4) Performance Evaluation with Human Judgement
    st.header("Performance Evaluation (Human Judgement)")

    # Define some test queries
    test_queries = [
        "What is distillation?",
        "How does DeepSeek-R1 improve reasoning?",
        "What are the main contributions of DeepSeek-R1?"
    ]

    if st.button("Evaluate Performance"):
        evaluation_results = evaluate_with_judgement(test_queries)
        st.write("Evaluation Results:")
        st.json(evaluation_results)


if __name__ == "__main__":
    main()
