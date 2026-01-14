# ============================================
# FILE 9: Makefile (Updated with model download)
# ============================================
.PHONY: build run stop clean test logs shell download-models

# Download models locally
download-models:
	@echo "Downloading models locally..."
	python download_models.py

# Build with model download
build:
	@echo "Building Docker image (will download models)..."
	docker build -t easymocap-api .

# Build for local dev (use local models)
build-local:
	@echo "Building with local models..."
	docker build -f Dockerfile.local -t easymocap-api-local .

run:
	@echo "Starting containers..."
	docker-compose up -d

stop:
	@echo "Stopping containers..."
	docker-compose down

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f

test:
	@echo "Testing API..."
	curl http://localhost:5000/

logs:
	docker-compose logs -f

shell:
	docker-compose exec easymocap-api /bin/bash

rebuild:
	@echo "Rebuilding from scratch..."
	docker-compose down -v
	docker build --no-cache -t easymocap-api .
	docker-compose up -d