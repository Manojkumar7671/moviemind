# Deploying MovieMind to Hugging Face Spaces

MovieMind is designed to deploy directly to Hugging Face Spaces using the Docker SDK. Because all artifacts (models, FAISS index, and parquet metadata) are precomputed and under 35 MB, serving requires zero training.

## Deployment Steps

1. **Create a New Space on Hugging Face**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Select **Docker** as the Space SDK (Blank template).
   - Set Space hardware to standard CPU (free tier).

2. **Configure Secrets**:
   - In your Space settings, under **Variables and secrets**, add:
     - `OPENROUTER_API_KEY`: Your OpenRouter API key (optional; agent falls back gracefully if absent).
     - `OPENROUTER_MODEL`: `meta-llama/llama-3.1-8b-instruct:free` (or any OpenAI-compatible model).

3. **Port Configuration**:
   - Hugging Face Spaces expects the web server to listen on port `7860`.
   - Update `Dockerfile`'s CMD or override via environment variable:
     ```dockerfile
     ENV PORT=7860
     EXPOSE 7860
     CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
     ```

4. **Push Repository to Space**:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/moviemind
   git push space main
   ```

5. **Verify Live Space**:
   - The UI loads at `https://<your-space>.hf.space/`.
   - Interactive chat, genre onboarding, and the `/ab` simulation dashboard will be live immediately.
