import fitz  # PyMuPDF
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

pdf_texts = {
    'python': extract_text_from_pdf('docs-pdf/tutorial.pdf'),
}

# Step 2: Initialize ChromaDB client and create collection
client = chromadb.Client()
collection = client.create_collection("all-my-documents")

# Function to split text into chunks
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Step 3: Chunk the text and add to collection
chunks = {}
ids = []
documents = []
metadatas = []

for key, text in pdf_texts.items():
    chunked_texts = chunk_text(text)
    chunks[key] = chunked_texts
    for i, chunk in enumerate(chunked_texts):
        doc_id = f"{key}_{i}"
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({"source": key})

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# Step 4: Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, k=5):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results['documents']

# Step 5: Load the LLaMA-3 8B model and tokenizer
llama_model_name = "meta-llama/Meta-Llama-3-8B"  # Replace with the actual model name
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

def generate_answer(query):
    relevant_documents = retrieve_relevant_chunks(query)
    context = ""
    for doc in relevant_documents:
        context += doc + "\n"
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = llama_tokenizer.encode(input_text, return_tensors='pt')
    outputs = llama_model.generate(inputs, max_length=512, num_return_sequences=1)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
while True:
    query = input("Enter your query: ")
    answer = generate_answer(query)
    print(answer)
