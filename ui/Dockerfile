FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv/ui
RUN pip install --no-cache-dir streamlit==1.64.0 httpx==0.28.1
COPY ui/app.py ui/components.py ./
RUN useradd --create-home --uid 1000 rag
USER rag

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
