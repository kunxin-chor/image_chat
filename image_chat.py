import gradio as gr
from gemini_utils import chat_with_gemini, generate_gemini_image, TEXT_MODEL, IMAGE_MODEL
# --- UI helpers and Gradio logic ---

# inputs: pending_image_prompt
# outputs: yes button visibility, no button visbility
def on_prompt_change(pending_image_prompt):
    if pending_image_prompt:
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)

def render_ui():
    # --- UI (Gradio v5) ---
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# 💬 Gemini Chat with Optional Images (Text + Streaming Image Model)")

        chatbot = gr.Chatbot(label="Chat")
        msg = gr.Textbox(label="Type your message")

        btn_yes = gr.Button("✅ Generate Image", visible=False)
        btn_no  = gr.Button("❌ Skip", visible=False)

        # State must be created inside the Blocks (v5)
        pending_image_prompt = gr.State("")

        # Main chat flow (returns history, button visibilities, and the pending prompt state)
        msg.submit(chat_fn, inputs=[msg, chatbot],
                  outputs=[chatbot, pending_image_prompt])

        pending_image_prompt.change(
            fn=on_prompt_change,
            inputs=[pending_image_prompt],
            outputs=[btn_yes, btn_no]
        )

        # Approval buttons
        btn_yes.click(generate_image, inputs=[pending_image_prompt, chatbot],
                      outputs=[chatbot, pending_image_prompt])
        btn_no.click(skip_image, inputs=[chatbot],
                    outputs=[chatbot, pending_image_prompt])
        demo.launch(share=False, debug=True)

def chat_fn(message, history):
    history, want_image, _, image_prompt = chat_with_gemini(message, history)
    if want_image and image_prompt:
        return history, image_prompt
    return history, ""



def generate_image(image_prompt, history):
    return generate_gemini_image(image_prompt, history)

def skip_image(history):
    return history, ""

render_ui()