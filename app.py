import json  # importing the JSON module for encoding and decoding data
import requests  # importing the Requests library for making HTTP requests
import gradio as gr  # importing the Gradio library for building web interfaces

# Define a function named ask_api that accepts 5 parameters -
# lcserve_host, url, file, question, openAI_key - and returns a string


def ask_api(
    lcserve_host: str,
    url: str,
    file,
    question: str,
    openAI_key: str,
) -> str:
    # Check if lcserve_host starts with "http"
    if not lcserve_host.startswith("http"):
        # Throw an exception if lcserve_host is invalid
        raise ValueError("Invalid API Host")

    # If neither url nor file is provided, throw an exception
    if not any([url.strip(), file]):
        raise ValueError("Either URL or PDF should be provided.")

    # If both url and file are provided, throw an exception
    if all([url.strip(), file]):
        raise ValueError("Both URL and PDF are provided. Please provide only one.")

    # If question field is empty, throw an exception
    if not question.strip():
        raise ValueError("Question field is empty.")

    # Create a dictionary _data with two keys "question" and "envs"
    _data = {
        "question": question,
        "envs": {"OPENAI_API_KEY": openAI_key},
    }

    # If url is provided, make a POST request to "lcserve_host"/ask_url route with data _data
    if url.strip():
        r = requests.post(f"{lcserve_host}/ask_url", json={"url": url, **_data})

    # Otherwise open the file in binary mode and make a POST request to "lcserve_host"/ask_file route with data _data and the file
    else:
        with open(file.name, "rb") as f:
            r = requests.post(
                f"{lcserve_host}/ask_file",
                params={"input_data": json.dumps(_data)},
                files={"file": f},
            )

    try:
        # Raise an HTTPError if one occurs while making a request to the server
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise ValueError(  # Throw a ValueError if the request fails
            f"Request failed with status code {r.status_code}: {e}"
        ) from e

    # Return the value of the "result" key in the JSON response
    return r.json()["result"]


# Define variables title and description which describe our Gradio interface
title = "PDF GPT"
description = """ PDF GPT allows you to chat with your PDF file using Universal Sentence Encoder and Open AI. It gives hallucination free response than other tools as the embeddings are better than OpenAI. The returned response can even cite the page number in square brackets([]) where the information is located, adding credibility to the responses and helping to locate pertinent information quickly."""

# Define a Gradio Blocks object named demo
with gr.Blocks() as demo:
    # Add a Markdown heading and description to the Gradio interface
    gr.Markdown(f"<center><h1>{title}</h1></center>")
    gr.Markdown(description)

    # Create two side-by-side Groups for input fields and outputs
    with gr.Row():
        with gr.Group():
            # Add a Textbox widget to accept the API host URL from the user
            lcserve_host = gr.Textbox(
                label="Enter your API Host here",
                value="http://localhost:8080",
                placeholder="http://localhost:8080",
            )

            # Add a link to the OpenAI API key webpage and a Password textbox to get the user's API Key
            gr.Markdown(
                '<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>'
            )
            openAI_key = gr.Textbox(
                label="Enter your OpenAI API key here", type="password"
            )

            # Add a Text box that allows users to enter URL of the PDF file they want to chat with
            pdf_url = gr.Textbox(label="Enter PDF URL here")

            # Add a File Upload widget so that users can upload their PDF/Research Paper/Book
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(
                label="Upload your PDF/ Research Paper / Book here", file_types=[".pdf"]
            )

            # Add a field for the user to enter their question
            question = gr.Textbox(label="Enter your question here")

            # Add a submit button for user to trigger their API request
            btn = gr.Button(value="Submit")
            btn.style(full_width=True)


        # Add another group for the output area where the answer will be shown
        with gr.Group():
            answer = gr.Textbox(label="The answer to your question is :")

        # Define function onclick() which will be called when the user clicks the "submit" button
        def on_click():
            try:
                # Call the ask_api function and update the answer in the Gradio UI
                ans = ask_api(
                    lcserve_host.value,
                    pdf_url.value,
                    file,
                    question.value,
                    openAI_key.value,
                )
                answer.update(str(ans))
            except ValueError as e:
                # Update the response with an error message if an error occurs during the API call
                answer.update(f"[ERROR]: {str(e)}")

        btn.click(on_click)

    # Launch the Gradio interface on port number 7860
    demo.launch(server_port=7860)