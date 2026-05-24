import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- PAGE TITLE --------------------

st.title("📚 Multi PDF AI Chatbot")

st.write("Upload PDFs and ask questions.")

# -------------------- FILE UPLOAD --------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# -------------------- PROCESS PDFs --------------------

if uploaded_files:

    text = ""

    for pdf in uploaded_files:

        reader = PdfReader(pdf)

        for page in reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text

    # -------------------- TEXT CHUNKING --------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    # -------------------- EMBEDDINGS --------------------

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------- VECTOR STORE --------------------

    vector_store = FAISS.from_texts(
        chunks,
        embedding_model
    )

    # -------------------- LOAD MODEL --------------------

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )

    # -------------------- USER QUESTION --------------------

    query = st.text_input("Ask a Question")

    if query:

        # Retrieve relevant chunks

        docs = vector_store.similarity_search(query, k=3)

        retrieved_text = ""

        for doc in docs:
            retrieved_text += doc.page_content + "\n"

        # Prompt

        prompt = f"""
        Answer the question based on the context below.

        Context:
        {retrieved_text}

        Question:
        {query}
        """

        # Generate Answer

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=150
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Display Answer

        st.subheader("Answer")

        st.write(answer)
