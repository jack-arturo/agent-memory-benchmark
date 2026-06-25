# One-command AMB reproduction (Docker + GEMINI_API_KEY only).
#
#   make repro-smoke                          # 5-query end-to-end sanity check
#   make repro DATASET=beam SPLIT=1m NAME=automem-sub
#   make repro-test                           # provider/retry unit tests in-image
#
# GEMINI_API_KEY is taken from your shell, else pulled (only that one var) from
# ./.env — so no stray keys leak into the lean AutoMem stack.

COMPOSE       ?= docker compose -f docker-compose.repro.yml
DATASET       ?= locomo
SPLIT         ?= locomo10
NAME          ?= automem-sub
AUTOMEM_IMAGE ?= ghcr.io/verygoodplugins/automem:amb-v1

# Prefer an already-exported key; otherwise extract just GEMINI_API_KEY from .env.
GEMINI_API_KEY ?= $(shell grep -E '^GEMINI_API_KEY=' .env 2>/dev/null | head -1 | cut -d= -f2-)
export GEMINI_API_KEY
export AUTOMEM_IMAGE

define _require_key
	@test -n "$(GEMINI_API_KEY)" || { echo "ERROR: set GEMINI_API_KEY (in your shell or ./.env)"; exit 1; }
endef

.PHONY: repro repro-smoke repro-test repro-build repro-clean help

help: ## List the repro targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

repro-build: ## Build the runner image.
	$(COMPOSE) build

repro: repro-build ## Run a full split end-to-end. Usage: make repro DATASET=beam SPLIT=1m NAME=automem-sub
	$(call _require_key)
	$(COMPOSE) run --rm runner \
		omb run --memory automem --dataset $(DATASET) --split $(SPLIT) --name $(NAME)

repro-smoke: repro-build ## 5-query end-to-end smoke that verifies the one-command path.
	$(call _require_key)
	$(COMPOSE) run --rm runner \
		omb run --memory automem --dataset $(DATASET) --split $(SPLIT) --query-limit 5 --name $(NAME)

repro-test: repro-build ## Run the provider/retry unit tests inside the runner image (no Docker spin).
	$(COMPOSE) run --rm runner \
		uv run --frozen --with pytest pytest tests/test_automem_provider.py tests/test_gemini_retry.py

repro-clean: ## Remove the dataset-cache volume.
	$(COMPOSE) down -v
