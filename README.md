---
title: AI Candidate Screening Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# AI Candidate Screening Assistant

AI-powered CV screening system that compares candidate resumes with job descriptions using semantic embeddings and rule-based qualification analysis.

## Features

- PDF CV extraction
- DOCX CV extraction
- Skill and keyword recognition
- Skill aliases such as:
  - Python / Python3
  - Scikit-learn / sklearn
  - Power BI / PowerBI
  - Machine Learning / ML
- Semantic similarity using BAAI/bge-small-en-v1.5
- Evidence-based matching
- Required qualification coverage
- Weighted candidate scoring
- Shortlist / Review / Do Not Shortlist recommendation
- Human-review indicator
- Fast inference without a large generative LLM

## How it works

The system combines:

1. CV text extraction
2. Job requirement extraction
3. Explicit keyword and skill matching
4. Semantic embedding similarity
5. Evidence scoring
6. Weighted final scoring
7. Recommendation generation

The system is designed to reduce CV screening time while maintaining transparent evidence for each matching requirement.

## Supported files

- PDF
- DOCX

