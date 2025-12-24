import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load the model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def predict(message, history):
    input_ids = tokenizer.encode(
        message + tokenizer.eos_token,
        return_tensors="pt"
    )

    output_ids = model.generate(
        input_ids,
        max_length=2000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        output_ids[:, input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return response

    # 4. Decode the response to text
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Update history for the next turn
    history = chat_history_ids.tolist()
    
    return response, history

# 5. Create the Gradio Web UI
demo = gr.ChatInterface(fn=predict, title="Talkie")

if __name__ == "__main__":
    demo.launch()
