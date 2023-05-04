import json
import requests

import gradio as gr


def ask_api(
    lcserve_host: str,
    url: str,
    file,
    question: str,
    openAI_key: str,
) -> str:
    if not lcserve_host.startswith("http"):
        raise ValueError("Invalid API Host")

    if not any([url.strip(), file]):
        raise ValueError("Either URL or PDF should be provided.")

    if all([url.strip(), file]):
        raise ValueError("Both URL and PDF are provided. Please provide only one.")

    if not question.strip():
        raise ValueError("Question field is empty.")

    _data = {
        "question": question,
        "envs": {"OPENAI_API_KEY": openAI_key},
    }

    if url.strip():
        r = requests.post(f"{lcserve_host}/ask_url", json={"url": url, **_data})

    else:
        with open(file.name, "rb") as f:
            r = requests.post(
                f"{lcserve_host}/ask_file",
                params={"input_data": json.dumps(_data)},
                files={"file": f},
            )

    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise ValueError(
            f"Request failed with status code {r.status_code}: {e}"
        ) from e

    return r.json()["result"]


title = "PDF GPT"
description = """ PDF GPT allows you to chat with your PDF file using Universal Sentence Encoder and Open AI. It gives hallucination free response than other tools as the embeddings are better than OpenAI. The returned response can even cite the page number in square brackets([]) where the information is located, adding credibility to the responses and helping to locate pertinent information quickly."""

with gr.Blocks() as demo:
    gr.Markdown(f"<center><h1>{title}</h1></center>")
    gr.Markdown(description)

    with gr.Row():
        with gr.Group():
            lcserve_host = gr.Textbox(
                label="Enter your API Host here",
                value="http://localhost:8080",
                placeholder="http://localhost:8080",
            )
            gr.Markdown(
                '<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>'
            )
            openAI_key = gr.Textbox(label="Enter your OpenAI API key here", type="password")
            pdf_url = gr.Textbox(label="Enter PDF URL here")
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(label="Upload your PDF/ Research Paper / Book here", file_types=[".pdf"])
            question = gr.Textbox(label="Enter your question here")
            btn = gr.Button(value="Submit")
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label="The answer to your question is :")

        def on_click():
            try:
                ans = ask_api(lcserve_host.value, pdf_url.value, file, question.value, openAI_key.value)
                answer.update(str(ans))
            except ValueError as e:
                answer.update(f"[ERROR]: {str(e)}")

        btn.click(on_click)

    demo.launch(server_port=7860)
