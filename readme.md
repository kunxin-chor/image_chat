# Dependencies

```
pip -q install --upgrade gradio google-genai pillow python-dotenv 
```

# Setup Instructions

1. **Create a virtual environment:**
   ```
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   - On Windows (Command Prompt):
     ```
     venv\Scripts\activate
     ```
   - On Windows (PowerShell):
     ```
     venv\Scripts\Activate.ps1
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

   If you see an error like `running scripts is disabled on this system`, run PowerShell as Administrator and execute:
   ```
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Then try activating again.
3. **Install dependencies:**
   ```
   pip install --upgrade gradio google-genai pillow
   ```

# Running the App

1. Make sure your virtual environment is activated.
2. Run:
   ```
   python image_chat.py
   ```

# Notes
- Update `requirements.txt` if you add more dependencies.
- For more info, see comments in `image_chat.py`.