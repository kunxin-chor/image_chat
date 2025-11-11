import re, json, base64
import gradio as gr
from google import genai
from google.genai import types
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os

# --- Keys / Models ---
load_dotenv()
GEMINI_KEY   = os.getenv("GOOGLE_API_KEY", "")
TEXT_MODEL   = "gemini-2.5-flash-lite"                    # chat/reasoning model
IMAGE_MODEL  = "gemini-2.5-flash-image-preview"           # image-gen model

# --- Strict system prompt for JSON control ---
SYSTEM_PROMPT = """
You are a helpful assistant.
Output ONLY a single valid JSON object. Do NOT include code fences or ```json markers.
{{ ... }}
- "reply" (string): your chat reply to display to the user.
- "want_image" (boolean): true only if an image would help or was requested.
- "image_prompt" (string): a short, specific prompt to generate the image if want_image is true, else "".
Example:
{"reply":"Hello there!","want_image":false,"image_prompt":""}
""".strip()

# --- Init Gemini client ---
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# --- Helpers ---
def strip_code_fences(text):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                  flags=re.IGNORECASE | re.MULTILINE)

def safe_parse_json(raw):
    cleaned = strip_code_fences(raw)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    block = m.group(0) if m else cleaned
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return {"reply": raw.strip(), "want_image": False, "image_prompt": ""}

def img_to_md(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"![image](data:image/png;base64,{b64})"

# inputs: pending_image_prompt
# outputs: yes button visibility, no button visbility
def on_prompt_change(pending_image_prompt):
  if pending_image_prompt:
    return  gr.update(visible=True), gr.update(visible=True)
  else:
    return  gr.update(visible=False), gr.update(visible=False)

def render_ui():
  # --- UI (Gradio v5) ---
  with gr.Blocks(theme="soft") as demo:
      gr.Markdown("# 💬 Gemini Chat with Optional Images (Text + Streaming Image Model)")
      gr.Markdown(f"**Text model:** `{TEXT_MODEL}`  |  **Image model:** `{IMAGE_MODEL}`")

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
      demo.launch(share=True, debug=True)

def sanitize_for_llm(text: str) -> str:
    if not text:
        return ""
    image_index = text.find("![image](data:image/png;base64,")
    if image_index != -1:
        text = text[:image_index]
    return text

def chat_fn(message, history):
    if not client:
        return history + [(message, "No GOOGLE_API_KEY configured.")], gr.update(visible=False), gr.update(visible=False), ""

    convo = SYSTEM_PROMPT + "\n\n"
    for user, bot in history:
        convo += f"User: {sanitize_for_llm(user)}\nAssistant: {sanitize_for_llm(bot)}\n"
    convo += f"User: {message}\nAssistant:"
    print(convo)

    resp = client.models.generate_content(model=TEXT_MODEL, contents=convo)

    raw = str(resp.text or "").strip()
    print(resp.text)
    if not raw:
        reply_text = "(No text received from the model.)"
        history.append((message, reply_text))
        return history, gr.update(visible=False), gr.update(visible=False), ""

    parsed = safe_parse_json(raw)
    reply_text   = parsed.get("reply", "").strip()
    want_image   = bool(parsed.get("want_image", False))
    image_prompt = parsed.get("image_prompt", "").strip()

    history.append((message, reply_text or "(Empty reply)"))

    if want_image and image_prompt:
        return history,  image_prompt
    else:
        return history, ""



def generate_image(image_prompt, history):
    if not image_prompt:
        return history, ""

    try:
        stream = client.models.generate_content_stream(
            model=IMAGE_MODEL,
            contents=image_prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT","IMAGE"]),
        )
        img = read_image_from_stream(stream)

        if img:
            md = img_to_md(img)
            # merge into last assistant message
            if history:
                last_user, last_bot = history[-1]
                history[-1] = (last_user, f"{last_bot}\n\n{md}")
            else:
                history = [("", md)]
        else:
            history = history + [("", "(No image data received)")]
        return history, ""
    except Exception as e:
        if history:
            last_user, last_bot = history[-1]
            history[-1] = (last_user, f"{last_bot}\n\n(Image generation failed: {e})")
        else:
            history = [("", f"(Image generation failed: {e})")]
        return history,  ""

def read_image_from_stream(stream):
  img = None
  for chunk in stream:
    cands = getattr(chunk, "candidates", None) or []
    for cand in cands:
        parts = getattr(getattr(cand, "content", None), "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                img = Image.open(BytesIO(inline.data))
                break
    if img:
        break
  return img


# --- Skip image ---
def skip_image(history):
    return history, ""

render_ui()