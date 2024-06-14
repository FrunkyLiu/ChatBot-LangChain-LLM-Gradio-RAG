import gradio as gr
from Gemini import Gemini


def main():
    # Initialize the Gemini chatbot with the document path
    model = Gemini("ABSOLUTE/PATH/TO/YOUR/DATABASE")
    # Create a Gradio ChatInterface for the chatbot
    demo = gr.ChatInterface(
        model.respond,
        examples=[
            "Who attacked Jennifer's village?",
        ],
        cache_examples=False # Disable example caching to avoid pre-computed responses
    )
    demo.launch()

if __name__ == "__main__":
    main()