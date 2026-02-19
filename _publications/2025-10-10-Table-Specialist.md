---
title: "Table-LLM-Specialist: Language Model Specialists for Tables using Iterative Fine-tuning"
collection: publications
publication_status: published
author:
    - self
    - Yeye He, Mengyu Zhou, Haoyu Dong, Shi Han, Dongmei Zhang, Surajit Chaudhuri
permalink: /publication/Table-LLM-Specialist-Language-Model-Specialists-for-Tables-using-Iterative-Fine-tuning
excerpt: ''
date: 2025-10-10
venue: 'EMNLP'
paperurl: 'https://aclanthology.org/2025.emnlp-main.1795.pdf'
pdf: 'https://aclanthology.org/2025.emnlp-main.1795.pdf'
# citation: 'Your Name, You. (2009). &quot;Paper Title Number 1.&quot; <i>Journal 1</i>. 1(1).'
bib: '../files/papers/table-llm-specialist/XingHZDHZC25.txt'
---

Language models such as GPT and Llama have
shown remarkable ability on diverse natural language tasks, yet their performance on complex
table tasks (e.g., NL-to-Code, data cleaning,
etc.) continues to be suboptimal. To improve
their performance, task-specific fine-tuning is
often needed, which, however, require expensive human labeling and is prone to over-fitting.
In this work, we propose TABLE-SPECIALIST,
a self-trained fine-tuning paradigm specifically
designed for table tasks. Our insight is that
for each table task, there often exist two dual
versions of the same task, one generative
and one classification in nature. Leveraging
their duality, we propose a Generator-Validator
paradigm to iteratively generate-then-validate
training data from language models, to finetune stronger TABLE-SPECIALIST models that
can specialize in a given task, without using
manually-labeled data.
