# Code Review Tool

## Running the Application

To launch the application, use the single entry point:

```bash
python launcher.py
```

All other launcher files have been consolidated into this single file for simplicity.

## Environment Variables

The application will automatically load environment variables from the .env file, including:

- OPENROUTER_API_KEY - Your OpenRouter API key if using cloud models
- OLLAMA_URL - Custom Ollama server URL if not using the default localhost

## Folder Structure

- api/ — LLM API integration (OpenRouter and Ollama)
- core/ — Prompt logic, chunking, caching, config
- ui/ — PySide6 UI components and styling
- workers/ — Background processing and coordination
- utils/ — Analysis tools, plugins, metrics, file operations
