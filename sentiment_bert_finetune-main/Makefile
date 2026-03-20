.PHONY: help setup download-model run-backend run-frontend docker-up docker-down test lint

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies locally
	pip install -r backend/requirements.txt
	pip install -r frontend/requirements.txt
	pip install pytest

download-model: ## Download the model from HuggingFace Hub
	cd backend && python model.py

run-backend: ## Start the FastAPI backend locally
	cd backend && uvicorn app:app --host 0.0.0.0 --port 8000 --reload

run-frontend: ## Start the Gradio frontend locally
	cd frontend && API_URL=http://127.0.0.1:8000/predict python gradio_ui.py

docker-up: ## Build and start all services with Docker
	docker compose up --build

docker-down: ## Stop all Docker services
	docker compose down

test: ## Run tests
	pytest tests/ -v

lint: ## Run linter
	python -m py_compile backend/app.py
	python -m py_compile backend/main.py
	python -m py_compile frontend/gradio_ui.py
	@echo "All files compile OK"
