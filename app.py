import json
from tempfile import _TemporaryFileWrapper

import gradio as gr
import requests


def ask_api(
    lcserve_host: str,
    url: str,
    file: _TemporaryFileWrapper,
    question: str,
    openAI_key: str,
) -> str:
    if not lcserve_host.startswith('http'):
        return '[ERROR]: Invalid API Host'

    if url.strip() == '' and file == None:
        return '[ERROR]: Both URL and PDF is empty. Provide at least one.'

    if url.strip() != '' and file != None:
        return '[ERROR]: Both URL and PDF is provided. Please provide only one (either URL or PDF).'

    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    _data = {
        'question': question,
        'envs': {
            'OPENAI_API_KEY': openAI_key,
        },
    }

    if url.strip() != '':
        r = requests.post(
            f'{lcserve_host}/ask_url',
            json={'url': url, **_data},
        )

    else:
        with open(file.name, 'rb') as f:
            r = requests.post(
                f'{lcserve_host}/ask_file',
                params={'input_data': json.dumps(_data)},
                files={'file': f},
            )

    if r.status_code != 200:
        raise ValueError(f'[ERROR]: {r.text}')

    return r.json()['result']


title = 'PDF GPT'
description = """ PDF GPT allows you to chat with your PDF file using Universal Sentence Encoder and Open AI. It gives hallucination free response than other tools as the embeddings are better than OpenAI. The returned response can even cite the page number in square brackets([]) where the information is located, adding credibility to the responses and helping to locate pertinent information quickly."""

with gr.Blocks() as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        with gr.Group():
            lcserve_host = gr.Textbox(
                label='Enter your API Host here',
                value='http://localhost:8080',
                placeholder='http://localhost:8080',
            )
            gr.Markdown(
                '<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>'
            )
            openAI_key = gr.Textbox(
                label='Enter your OpenAI API key here', type='password'
            )
            pdf_url = gr.Textbox(label='Enter PDF URL here')
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(
                label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf']
            )
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')

        btn.click(
            ask_api,
            inputs=[lcserve_host, pdf_url, file, question, openAI_key],
            outputs=[answer],
        )

demo.app.server.timeout = 60000 # Set the maximum return time for the results of accessing the upstream server

demo.launch(server_port=7860, enable_queue=True) # `enable_queue=True` to ensure the validity of multi-user requests
