# test_imports.py
import groq
import google.generativeai as genai

print(f"✅ Groq version: {groq.__version__}")
print(f"✅ Google GenAI version: {genai.__version__}")
print("✅ All packages installed successfully!")