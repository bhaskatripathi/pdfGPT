
# pdfGPT
What is PDF GPT ?
1. PDF GPT allows you to chat with a PDF file using GPT functionalities.
2. The application intelligently breaks the document into smaller chunks and employs a powerful Deep Averaging Network Encoder to generate embeddings.
3. PDF GPT utilizes Open AI as its data layer to generate a summary for each chunk.
4. PDF GPT uses a KNN algorithm to return the top-n embedding from each chunk and uses custom logic to generate a response. The application also leverages important document sections to generate precise responses, and can even provide the page number where the information is located, adding credibility to the responses and helping to locate pertinent information quickly.

### Demo
Demo URL: https://huggingface.co/spaces/bhaskartripathi/pdfChatter

### Description
This pipeline allows users to input a URL to a PDF document, preprocess the text, and use semantic search to generate answers to user questions. The pipeline follows the following steps:

1. **Input**: User inputs a URL to a PDF document.
2. **Download** PDF: The pipeline downloads the PDF from the input URL.
3. **Load PDF**: The PDF is loaded into the pipeline.
4. **Preprocess**: The text of the PDF is preprocessed to prepare for semantic search.
5. **Text Chunks**: The text of the PDF is split into smaller text chunks for semantic search.
6. **Embedding**: Generate an embedding of each text chunks usingDeep Averaging Network Encoder.
7. **Query:** User inputs a question.
8. **Get Top Results**: The top semantic search results for the user's question are returned.
9. **Generate Answer**: The pipeline generates an answer based on the top semantic search results.
10. **Output**: The answer is outputted

**NOTE**: Please star this project if you like it!

### Flowchart
```mermaid
flowchart TB
A[Input] --> B[URL]
A -- Upload File manually --> C[Parse PDF]
B --> D[Parse PDF] -- Preprocess --> E[Dynamic Text Chunks]
C -- Preprocess --> E[Dynamic Text Chunks with citation history]
E --Fit-->F[Generate text embedding with Deep Averaging Network Encoder on each chunk]
F -- Query --> G[Get Top Results]
G -- K-Nearest Neighbour --> K[Get Nearest Neighbour - matching citation references]
K -- Generate Prompt --> H[Generate Answer]
H -- Output --> I[Output]


