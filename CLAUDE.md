# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Jekyll-based academic personal website built on the [Academic Pages](https://academicpages.github.io/) template. It is deployed to GitHub Pages at `https://xingjunjie.me/blog`. The owner is Gavin Junjie Xing, a Senior Researcher at Microsoft Research.

## Running locally

```bash
# Install dependencies (first time only)
bundle install

# Serve with live reload
bundle exec jekyll serve -l -H localhost
# → http://localhost:4000
```

Changes to `_config.yml` require a server restart; changes to Markdown and HTML files reload automatically.

**Dev config** (`_config.dev.yml`) overrides the production URL and disables analytics — it is not passed by default; the serve command above uses only `_config.yml`.

**Docker alternative:**
```bash
chmod -R 777 .
docker compose up
# → http://localhost:4000
```

## Content architecture

| Directory | Purpose |
|-----------|---------|
| `_posts/` | Blog posts (date-prefixed Markdown) |
| `_publications/` | Research papers (one `.md` per paper) |
| `_portfolio/` | Project portfolio items |
| `_teaching/` | Course entries |
| `_drafts/` | Unpublished drafts |
| `_pages/` | Static pages (about, CV, talks, etc.) |
| `_data/` | Structured data: `navigation.yml`, `cv.json` |
| `_layouts/` | Jinja/Liquid templates (`single`, `talk`, `cv-layout`, etc.) |
| `_includes/` | Reusable template fragments |
| `_sass/` | SCSS stylesheets |
| `files/` | Static file downloads (PDFs, etc.) |
| `images/` | Site images |

## Publication front matter

Each file in `_publications/` uses this schema:

```yaml
---
title: "Paper Title"
author:
    - First Author
    - Other Authors
publication_status: published   # or: preprint
remarks: "Award name"           # optional
collection: publications
category:                       # leave blank or: books, manuscripts, conferences
permalink: /publication/YYYY-MM-slug
date: YYYY-MM-DD
venue: 'Venue Name'
paperurl: 'https://...'
pdf: 'https://...'
bib: '../files/papers/slug/cite.txt'
code: 'https://github.com/...'  # optional
citation: ''
---
```

## CV

The CV is driven by `_data/cv.json` (JSON Resume format) and rendered by `_layouts/cv-layout.html`. To regenerate `cv.json` from a Markdown CV source, use `scripts/cv_markdown_to_json.py` or `scripts/update_cv_json.sh`.

## Talk location map

`talkmap.ipynb` geocodes talks and generates a map. The GitHub Actions workflow `.github/workflows/scrape_talks.yml` runs this notebook automatically when files under `_talks/` or `talkmap.ipynb` change.

## Key config

- **Site URL / baseurl**: set in `_config.yml` (`url: https://xingjunjie.me`, `baseurl: /blog`)
- **Author profile sidebar**: all fields under `author:` in `_config.yml`
- **Navigation menu**: `_data/navigation.yml`
- **Collections** (publications, talks, teaching, portfolio, news): configured in `_config.yml` under `collections:`
