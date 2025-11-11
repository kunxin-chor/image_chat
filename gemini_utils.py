import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GOOGLE_API_KEY", "")
TEXT_MODEL = "gemini-2.5-flash-lite"
IMAGE_MODEL = "gemini-2.5-flash-image-preview"
SYSTEM_PROMPT = """
You are a helpful assistant.
Output ONLY a single valid JSON object. Do NOT include code fences or ```json markers.
Do NOT add any text before or after the JSON. The JSON must have EXACTLY these keys:
- \"reply\" (string): your chat reply to display to the user.
- \"want_image\" (boolean): true only if an image would help or was requested.
- \"image_prompt\" (string): a short, specific prompt to generate the image if want_image is true, else \"\".
Example:
{"reply":"Hello there!","want_image":false,"image_prompt":""}
""".strip()

client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

def strip_code_fences(text):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)

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

def chat_with_gemini(message, history):
    if not client:
        return history + [(message, "No GOOGLE_API_KEY configured.")], False, False, ""
    convo = SYSTEM_PROMPT + "\n\n"
    for user, bot in history:
        convo += f"User: {user}\nAssistant: {bot}\n"
    convo += f"User: {message}\nAssistant:"
    resp = client.models.generate_content(model=TEXT_MODEL, contents=convo)
    raw = str(resp.text or "").strip()
    if not raw:
        reply_text = "(No text received from the model.)"
        history.append((message, reply_text))
        return history, False, False, ""
    parsed = safe_parse_json(raw)
    reply_text   = parsed.get("reply", "").strip()
    want_image   = bool(parsed.get("want_image", False))
    image_prompt = parsed.get("image_prompt", "").strip()
    history.append((message, reply_text or "(Empty reply)"))
    if want_image and image_prompt:
        return history, True, True, image_prompt
    else:
        return history, False, False, ""

def generate_gemini_image(image_prompt, history):
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
