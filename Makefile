.PHONY: help build up down restart logs clean test

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build Docker images
	docker-compose build

build-prod: ## Build production Docker images
	docker-compose -f docker-compose.prod.yml build

up: ## Start services in development mode
	docker-compose up -d

up-prod: ## Start services in production mode
	docker-compose -f docker-compose.prod.yml up -d

down: ## Stop services
	docker-compose down

down-prod: ## Stop production services
	docker-compose -f docker-compose.prod.yml down

restart: ## Restart services
	docker-compose restart

logs: ## View logs
	docker-compose logs -f

logs-api: ## View API logs
	docker-compose logs -f api

logs-dashboard: ## View dashboard logs
	docker-compose logs -f dashboard

clean: ## Remove containers and volumes
	docker-compose down -v

clean-all: ## Remove containers, volumes, and images
	docker-compose down -v --rmi all

ps: ## Show running containers
	docker-compose ps

init-db: ## Initialize database
	docker-compose exec api python api/database_init.py

test-api: ## Test API health
	@echo "Testing API..."
	@curl -s http://localhost:8000/ | head -20 || echo "API not responding"

test-dashboard: ## Test dashboard
	@echo "Testing Dashboard..."
	@curl -s http://localhost:3000/ | head -20 || echo "Dashboard not responding"

shell-api: ## Open shell in API container
	docker-compose exec api /bin/bash

shell-dashboard: ## Open shell in dashboard container
	docker-compose exec dashboard /bin/sh
