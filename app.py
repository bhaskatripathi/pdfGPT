"""
This module provides functions for working with PDF files and URLs. It uses the urllib.request library
to download files from URLs, and the fitz library to extract text from PDF files. And GPT3 modules to generate
text completions.
"""

import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Union, IO


def download_pdf(url: str, output_path: str) -> None:
    """
    Downloads a PDF file from the given URL and saves it to the specified output path.

    Args:
    url (str): The URL of the PDF file to be downloaded.
    output_path (str): The file path where the downloaded PDF file will be saved.

    Returns:
    None
    """
    urllib.request.urlretrieve(url, output_path)


def preprocess(text: str) -> str:
    """
    Preprocesses the given text by replacing newline characters with spaces and removing extra whitespaces.

    Args:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text with newline characters replaced by spaces and extra whitespaces removed.

    Example:
        >>> preprocess("Hello\\n   world!")
        'Hello world!'
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text


def pdf_to_text(path: str, start_page: int = 1, end_page: Optional[int] = None) -> list[str]:
    """
    Converts a PDF file to a list of text strings.

    Args:
        path (str): The path to the PDF file.
        start_page (int): The page number to start extracting text from (default is 1).
        end_page (int): Page number to stop extracting text at (default is None, which means extract text from all ]
        pages)

    Returns:
        list: A list of text strings extracted from the PDF file.
    """
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts: list[str], word_length: int = 150, start_page: int = 1) -> list[str]:
    """
    Splits a list of texts into chunks of specified length and formats them as strings.

    Args:
    - texts: A list of strings to be split into chunks.
    - word_length: An integer representing the maximum number of words in each chunk. Default is 150.
    - start_page: An integer representing the starting page number. Default is 1.

    Returns:
    - A list of formatted string chunks, where each chunk contains a page number, enclosed in square brackets,
    followed by the chunk of text enclosed in double quotes.

    Example:
    >>> texts = ['This is a sample text for testing the function.', 'It should split the text into chunks of 5 words.']
    >>> text_to_chunks(texts, word_length=5, start_page=3)
    ['[3] "This is a sample text for"', '[3] "testing the function. It should"',
        '[4] "split the text into chunks of"','[4] "5 words."']
    """
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                    len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch(object):
    """
    This class provides functionality for semantic search.
    """

    def __init__(self) -> None:
        """
        Initializes an instance of the class.

        Attributes:
        -----------
        use : tensorflow_hub.KerasLayer
            A pre-trained Universal Sentence Encoder model from TensorFlow Hub.
        fitted : bool
            A flag indicating whether the model has been fitted to data or not.
        """
        self.use = hub.load(
            'https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False

    def fit(self, data: list[str], batch: int = 1000, n_neighbors: int = 5) -> None:
        """
        Fits the nearest neighbor model to the given data.

        Args:
            data (list[str]): A list of strings to fit the model on.
            batch (int): The batch size to use when computing text embeddings. Defaults to 1000.
            n_neighbors (int): The number of nearest neighbors to find for each query. Defaults to 5.

        Returns:
            None
        """
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text: str, return_data: bool = True) -> Union[list[str], np.ndarray]:
        """
        Finds nearest neighbors to a given text in the embedding space.

        Args:
            text (str): The input text to find nearest neighbors for.
            return_data (bool): Whether to return the actual data points corresponding to the nearest neighbors.
                If False, returns only the indices of the nearest neighbors. Defaults to True.

        Returns:
            Union[List[str], np.ndarray]: If return_data is True, returns a list of strings representing the
            nearest neighbors. If return_data is False, returns a numpy array of shape (n_neighbors,)
            containing the indices of the nearest neighbors.
        """
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts: list[str], batch: int = 1000) -> np.ndarray:
        """
        Generates embeddings for a list of texts using the Universal Sentence Encoder.

        Args:
            texts (List[str]): A list of strings to generate embeddings for.
            batch (int): The batch size to use when generating embeddings. Defaults to 1000.

        Returns:
            np.ndarray: An array of shape (n_texts, embedding_size) containing the embeddings for each text.
        """
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def load_recommender(path: str, start_page: int = 1) -> str:
    """
    Loads embeddings from file if available, otherwise generates embeddings and saves them to file.

    Args:
    path (str): The path of the PDF file.
    start_page (int): The page number to start generating embeddings from. Default is 1.

    Returns:
    str: A message indicating whether embeddings were loaded from file or generated and saved to file.
    """
    global recommender
    pdf_file = os.path.basename(path)
    embeddings_file = f"{pdf_file}_{start_page}.npy"

    if os.path.isfile(embeddings_file):
        embeddings = np.load(embeddings_file)
        recommender.embeddings = embeddings
        recommender.fitted = True
        return "Embeddings loaded from file"

    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    np.save(embeddings_file, recommender.embeddings)
    return 'Corpus Loaded.'


def generate_text(openai_key: str, prompt: str, engine: str = "text-davinci-003") -> str:
    """
    Generates text using OpenAI's GPT-3 language model.

    Parameters:
    openai_key (str): The API key for accessing OpenAI's API.
    prompt (str): The starting text prompt to generate the text from.
    engine (str): The ID of the language model to use. Defaults to "text-davinci-003".

    Returns:
    str: The generated text based on the given prompt.
    """
    openai.api_key = openai_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_text2(openai_key: str, prompt: str, engine: str = "gpt-3.5-turbo-0301") -> str:
    """
    Generates text using OpenAI's GPT-3 language model.

    Args:
        openai_key (str): The API key for accessing OpenAI's GPT-3 language model.
        prompt (str): The user's prompt to generate a response to.
        engine (str, optional): The name of the GPT-3 engine to use. Defaults to "gpt-3.5-turbo-0301".

    Returns:
        str: The generated text response from the GPT-3 language model.
    """
    openai.api_key = openai_key
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]

    completions = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].message['content']
    return message


def generate_answer(question: str, openai_key: str) -> str:
    """
    Generates an answer to the given question using OpenAI's GPT-3 language model.

    Args:
        question (str): The question to answer.
        openai_key (str): The API key for accessing OpenAI's GPT-3 API.

    Returns:
        str: The generated answer to the question.
    """
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'

    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
        "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "\
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
        "with the same name, create separate answers for each. Only include information found in the results and "\
        "don't add any additional information. Make sure the answer is correct and don't output false content. "\
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "\
        "search results which has nothing to do with the question. Only answer what is asked. The "\
        "answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: "

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openai_key, prompt, "text-davinci-003")
    return answer


def question_answer(url: str, file: IO[str], question: str, openai_key: str) -> str:
    """
    Generates an answer to a given question using OpenAI's GPT-3 model.

    Parameters:
    -----------
    url : str
        The URL of a webpage to extract text from. If provided, the text will be saved as a PDF and used
        as input for the model.
    file : file-like object
        A file object containing a PDF document to use as input for the model. If provided, the text will
        be extracted from the PDF and used as input for the model.
    question : str
        The question to generate an answer for.
    openai_key : str
        An API key for accessing OpenAI's GPT-3 model.

    Returns:
    --------
    str
        The generated answer to the given question.

    Raises:
    -------
    ValueError
        If both `url` and `file` are empty or if both are provided.
        If `question` is empty.
    """
    if openai_key.strip() == '':
        return '[ERROR]: Please enter you Open AI Key. Get your key here : https://platform.openai.com/account/api-keys'
    if url.strip() == '' and file == None:
        return '[ERROR]: Both URL and PDF is empty. Provide at least one.'

    if url.strip() != '' and file != None:
        return '[ERROR]: Both URL and PDF is provided. Please provide only one (either URL or PDF).'

    if url.strip() != '':
        glob_url = url
        download_pdf(glob_url, 'corpus.pdf')
        load_recommender('corpus.pdf')

    else:
        old_file_name = file.name
        file_name = file.name
        file_name = file_name[:-12] + file_name[-4:]
        os.rename(old_file_name, file_name)
        load_recommender(file_name)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    return generate_answer(question, openai_key)


recommender = SemanticSearch()

title = 'PDF GPT'
description = """ What is PDF GPT ?
1. The problem is that Open AI has a 4K token limit and cannot take an entire PDF file as input. Additionally,
it sometimes returns irrelevant responses due to poor embeddings. ChatGPT cannot directly talk to external data.
The solution is PDF GPT, which allows you to chat with an uploaded PDF file using GPT functionalities.
The application breaks the document into smaller chunks and generates embeddings using a powerful Deep Averaging
Network Encoder. A semantic search is performed on your query, and the top relevant chunks are used to generate a
response.
2. The returned response can even cite the page number in square brackets([]) where the information is located,
adding credibility to the responses and helping to locate pertinent information quickly. The Responses are much
better than the naive responses by Open AI."""

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():

        with gr.Group():
            gr.Markdown(
                '<p style="text-align:center">'
                'Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a>'
                '</p>'
            )
            openAI_key = gr.Textbox(label='Enter your OpenAI API key here')
            url = gr.Textbox(label='Enter PDF URL here')
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(
                label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf'])
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn.click(question_answer, inputs=[
            url, file, question, openAI_key], outputs=[answer])

demo.launch()
