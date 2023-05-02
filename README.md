
# pdfGPT
### Problem Description : 
1. When you pass a large text to Open AI, it suffers from a 4K token limit. It cannot take an entire pdf file as an input
2. Open AI sometimes becomes overtly chatty and returns irrelevant response not directly related to your query. This is because Open AI uses poor embeddings.
3. ChatGPT cannot directly talk to external data. Some solutions use Langchain but it is token hungry if not implemented correctly.
4. There are a number of solutions like https://www.chatpdf.com, https://www.bespacific.com/chat-with-any-pdf, https://www.filechat.io they have poor content quality and are prone to hallucination problem. One good way to avoid hallucinations and improve truthfulness is to use improved embeddings. To solve this problem, I propose to improve embeddings with Universal Sentence Encoder family of algorithms (Read more here: https://tfhub.dev/google/collections/universal-sentence-encoder/1). 

### Solution: What is PDF GPT ?
1. PDF GPT allows you to chat with an uploaded PDF file using GPT functionalities.
2. The application intelligently breaks the document into smaller chunks and employs a powerful Deep Averaging Network Encoder to generate embeddings.
3. A semantic search is first performed on your pdf content and the most relevant embeddings are passed to the Open AI.
4. A custom logic generates precise responses. The returned response can even cite the page number in square brackets([]) where the information is located, adding credibility to the responses and helping to locate pertinent information quickly. The Responses are much better than the naive responses by Open AI.
5. Andrej Karpathy mentioned in this post that KNN algorithm is most appropriate for similar problems: https://twitter.com/karpathy/status/1647025230546886658

### Demo
Demo URL: https://bit.ly/41ZXBJM

**NOTE**: Please star this project if you like it!

### How to use it?

1. Run `lc-serve deploy local api` on one terminal to expose the app as API using langchain-serve.
2. Run `python app.py` on another terminal.
3. Open `http://localhost:7860` on your browser and interact with the app.
4. Deploy the app on Cloud using `lc-serve deploy jcloud api` & share your app with the world.

    <details>
    <summary>Show command output</summary>

    ```text
    ╭──────────────┬──────────────────────────────────────────────────────────────────────────────────────╮
    │ App ID       │                                 langchain-3ff4ab2c9d                                 │
    ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
    │ Phase        │                                       Serving                                        │
    ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
    │ Endpoint     │                      https://langchain-3ff4ab2c9d.wolf.jina.ai                       │
    ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
    │ App logs     │                               dashboards.wolf.jina.ai                                │
    ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
    │ Swagger UI   │                    https://langchain-3ff4ab2c9d.wolf.jina.ai/docs                    │
    ├──────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
    │ OpenAPI JSON │                https://langchain-3ff4ab2c9d.wolf.jina.ai/openapi.json                │
    ╰──────────────┴──────────────────────────────────────────────────────────────────────────────────────╯
    ```
    
    </details>

Read more about **langchain-serve** [here](https://github.com/jina-ai/langchain-serve).

### UML
```mermaid
sequenceDiagram
    participant User
    participant System

    User->>System: Enter API Key
    User->>System: Upload PDF/PDF URL
    User->>System: Ask Question
    User->>System: Submit Call to Action

    System->>System: Blank field Validations
    System->>System: Convert PDF to Text
    System->>System: Decompose Text to Chunks (150 word length)
    System->>System: Check if embeddings file exists
    System->>System: If file exists, load embeddings and set the fitted attribute to True
    System->>System: If file doesn't exist, generate embeddings, fit the recommender, save embeddings to file and set fitted attribute to True
    System->>System: Perform Semantic Search and return Top 5 Chunks with KNN
    System->>System: Load Open AI prompt
    System->>System: Embed Top 5 Chunks in Open AI Prompt
    System->>System: Generate Answer with Davinci

    System-->>User: Return Answer
```

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
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bhaskatripathi/pdfGPT&type=Date)](https://star-history.com/#bhaskatripathi/pdfGPT&Date)

## Also Try:
This app creates schematic architecture diagrams, UML, flowcharts, Gantt charts and many more. You simple need to mention the usecase in natural language and it will create the desired diagram.
https://github.com/bhaskatripathi/Text2Diagram


