# Common dev commands for mlanthology.
# Run `make help` to see available targets.

.DEFAULT_GOAL := help
PYTHON ?= python

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

.PHONY: fetch fetch-quick fetch-pmlr fetch-dblp

fetch:  ## Fetch all sources (uses cache, skips already-fetched volumes)
	$(PYTHON) scripts/fetch_all.py

fetch-quick:  ## Quick fetch: latest volume per venue only
	$(PYTHON) scripts/fetch_all.py --quick

fetch-pmlr:  ## Fetch PMLR only
	$(PYTHON) scripts/fetch_all.py --source pmlr

fetch-dblp:  ## Fetch DBLP backlog only
	$(PYTHON) scripts/fetch_all.py --source dblp

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

.PHONY: build build-sample build-only serve

build:  ## Build Hugo content from data files
	$(PYTHON) scripts/build_content.py

build-sample:  ## Build a small sample for fast UI iteration
	$(PYTHON) scripts/build_content.py --sample

build-only:  ## Rebuild Hugo content without fetching
	$(PYTHON) scripts/fetch_all.py --build-only

serve: build  ## Build content then start Hugo dev server
	hugo server --source hugo --buildDrafts

dev: build-sample  ## Fast dev loop: sample build + Hugo server with live reload
	hugo server --source hugo --buildDrafts --disableFastRender

# ---------------------------------------------------------------------------
# Legacy data
# ---------------------------------------------------------------------------

.PHONY: legacy legacy-dry-run

legacy:  ## Build legacy data for a single venue (VENUE=icml)
ifndef VENUE
	$(error Set VENUE, e.g. make legacy VENUE=icml)
endif
	$(PYTHON) scripts/build_legacy.py --venue $(VENUE)

legacy-dry-run:  ## Show what legacy build would do
	$(PYTHON) scripts/build_legacy.py --dry-run

# ---------------------------------------------------------------------------
# Repair / enrichment
# ---------------------------------------------------------------------------

.PHONY: repair repair-dry-run

repair:  ## Re-normalise author names in all data files
	$(PYTHON) -m scripts.repair_author_names

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
