# import libraries
import os
import re
import shutil
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile
import fitz
import numpy as np
import openai
import tensorflow_hub as hub
from fastapi import UploadFile
from lcserve import serving
from sklearn.neighbors import NearestNeighbors

# download pdf from given url


recommender = None


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


# preprocess text


def preprocess(text):
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text


# convert pdf to text list


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    # if end page is not specified set it to total pages
    if end_page is None:
        end_page = total_pages

    text_list = []

    # loop through all the pages and get the text
    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


# convert text list to chunks of words with page numbers


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(" ") for t in texts]
    page_nums = []
    chunks = []

    # loop through each word and create chunks
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            # if last chunk is smaller than word length and not last page then add it to next page
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = " ".join(chunk).strip()
            chunk = f'[Page no. {idx + start_page}] "{chunk}"'
            chunks.append(chunk)
    return chunks


# semantic search class


class SemanticSearch:
    def __init__(self):
        self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.fitted = False

    # fit the data
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    # call the model
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        return [self.data[i] for i in neighbors] if return_data else neighbors

    # get text embedding
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i : (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        return np.vstack(embeddings)


# load recommender


def load_recommender(path, start_page=1):
    global recommender
    if recommender is None:
        recommender = SemanticSearch()

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return "Corpus Loaded."


# generate text using openAI


def generate_text(openAI_key, prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return completions.choices[0].text


# generate answer for a given question


def generate_answer(question, openAI_key):
    topn_chunks = recommender(question)
    prompt = "" + "search results:\n\n"
    for c in topn_chunks:
        prompt += c + "\n\n"

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "
    )

    prompt += f"Query: {question}\nAnswer:"
    return generate_text(openAI_key, prompt, "text-davinci-003")



# global instance of semantic search
recommender = SemanticSearch()

# load openAI key



def load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError(
            "[ERROR]: Please pass your OPENAI_API_KEY. Get your key here : https://platform.openai.com/account/api-keys"
        )
    return key


# ask url


@serving
def ask_url(url: str, question: str):
    download_pdf(url, "corpus.pdf")
    load_recommender("corpus.pdf")
    openAI_key = load_openai_key()
    return generate_answer(question, openAI_key)


# ask file


@serving
async def ask_file(file: UploadFile, question: str) -> str:
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    load_recommender(str(tmp_path))
    openAI_key = load_openai_key()
    return generate_answer(question, openAI_key)
