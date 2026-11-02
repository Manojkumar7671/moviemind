.PHONY: help install prepare train diversity evaluate test serve docker-build docker-run

help:
	@echo "MovieMind Commands:"
	@echo "  make install    - Install pinned dependencies"
	@echo "  make prepare    - Download and split MovieLens 1M dataset"
	@echo "  make train      - Train Two-Tower retrieval and Wide&Deep ranker"
	@echo "  make diversity  - Run MMR diversity tradeoff analysis"
	@echo "  make evaluate   - Run full pipeline ablation table and plot chart"
	@echo "  make test       - Execute pytest test suite"
	@echo "  make serve      - Launch FastAPI application at http://localhost:8000"
	@echo "  make docker-build - Build Docker container"
	@echo "  make docker-run   - Run container on port 8000"

install:
	pip install -r requirements.txt

prepare:
	python prepare.py

train:
	python retrieval.py
	python ranker.py

diversity:
	python diversity.py

evaluate:
	python evaluate.py

test:
	pytest -v

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t moviemind:latest .

docker-run:
	docker run -p 8000:8000 moviemind:latest
