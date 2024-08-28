from fasthtml import HTML

def create_html():
    """Generate the HTML content for the project page."""
    html = HTML()

    with html:
        html.head(
            html.title("Fine-Tune LLaMA Story Teller"),
            html.meta(charset="UTF-8"),
            html.meta(name="viewport", content="width=device-width, initial-scale=1.0"),
            html.style("""
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f4f4;
                    color: #333;
                }
                header {
                    background: #333;
                    color: #fff;
                    padding: 1em 0;
                    text-align: center;
                }
                .container {
                    width: 80%;
                    margin: auto;
                    overflow: hidden;
                }
                h1, h2, h3 {
                    color: #333;
                }
                code {
                    background: #eee;
                    padding: 2px 4px;
                    border-radius: 3px;
                }
                pre {
                    background: #eee;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
                a {
                    color: #333;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                footer {
                    background: #333;
                    color: #fff;
                    text-align: center;
                    padding: 1em 0;
                    position: fixed;
                    width: 100%;
                    bottom: 0;
                }
            """)
        )
        html.body(
            html.header(
                html.h1("Fine-Tune LLaMA Story Teller")
            ),
            html.div(class_="container",
                html.section(
                    html.h2("Overview"),
                    html.p("This project contains a script for fine-tuning the LLaMA model using a custom dataset and specialized configurations. It leverages advanced techniques like quantization and low-rank adaptation (LoRA) to optimize training and performance.")
                ),
                html.section(
                    html.h2("Dependencies"),
                    html.p("The following dependencies are required:"),
                    html.ul(
                        html.li(code("torch")),
                        html.li(code("datasets")),
                        html.li(code("peft")),
                        html.li(code("transformers")),
                        html.li(code("trl"))
                    ),
                    html.p("Install them using ", code("pip"), ":"),
                    html.pre(html.code("pip install torch datasets peft transformers trl"))
                ),
                html.section(
                    html.h2("Usage"),
                    html.p("To fine-tune the LLaMA model, run the script:"),
                    html.pre(html.code("python finetune_llama_story_teller.py"))
                ),
                html.section(
                    html.h2("Script Details"),
                    html.ol(
                        html.li(html.strong("Loading Data:"), " The ", code("load_data"), " function fetches the dataset specified by ", code("dataset_name"), " and ", code("split_name"), "."),
                        html.li(html.strong("Tokenization:"), " The ", code("initialize_tokenizer"), " function sets up the tokenizer and configures the padding token."),
                        html.li(html.strong("Model Initialization:"), " The ", code("initialize_model"), " function initializes the LLaMA model with 4-bit quantization for efficient computation."),
                        html.li(html.strong("LoRA Configuration:"), " The ", code("configure_lora"), " function sets up LoRA for low-rank adaptation to enhance model performance."),
                        html.li(html.strong("Training Arguments:"), " The ", code("configure_training_arguments"), " function specifies parameters such as batch size, learning rate, and number of epochs."),
                        html.li(html.strong("Training:"), " The ", code("train_model"), " function uses ", code("SFTTrainer"), " to fine-tune the model and push the results to the model hub.")
                    )
                ),
                html.section(
                    html.h2("Configuration"),
                    html.ul(
                        html.li(html.strong("Dataset:"), " The script uses the dataset ", code('"2173ars/finetuning_story"'), ". Ensure that you have access to this dataset or modify the ", code("dataset_name"), " variable accordingly."),
                        html.li(html.strong("Model:"), " The script uses ", code('"meta-llama/Llama-2-7b-hf"'), ". You can replace this with a different model if needed."),
                        html.li(html.strong("Training Parameters:"), " Adjust the training parameters in ", code("configure_training_arguments"), " based on your requirements.")
                    )
                ),
                html.section(
                    html.h2("Contributing"),
                    html.p("Feel free to open issues or submit pull requests if you have improvements or suggestions.")
                ),
                html.section(
                    html.h2("License"),
                    html.p("This project is licensed under the MIT License. See the ", code("LICENSE"), " file for more details.")
                )
            ),
            html.footer(
                html.p("For any questions or issues, please open an issue in the repository or contact the maintainers."),
                html.p("Happy fine-tuning!")
            )
        )

    return str(html)

def main():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        return create_html()

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)

