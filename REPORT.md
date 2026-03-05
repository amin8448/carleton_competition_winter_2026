# SQL Query Writer Agent — Technical Report

## Overview

This agent translates natural language questions into executable SQL queries
for a bike store database using an open-source LLM (qwen3:32b) via Ollama.
Rather than relying on a single prompt-and-hope approach, the agent implements
a multi-stage pipeline designed to maximize query accuracy, even when faced with
ambiguous or unusual inputs.

## Architecture

The query generation pipeline follows these stages:

1. **Input Validation** — Rejects empty or whitespace-only inputs
2. **Context Assembly** — Builds a rich prompt containing:
   - Full database schema with column types
   - Sample values from every column (3 per column)
   - Five few-shot examples covering different query patterns
3. **SQL Generation** — Sends the assembled prompt to the LLM
4. **Response Cleaning** — Strips thinking blocks, markdown formatting,
   and multi-statement responses
5. **Validation** — Runs `EXPLAIN` on the generated SQL against DuckDB
6. **Self-Correction Loop** — If validation fails, feeds the error message
   back to the LLM and retries (up to 3 attempts)

## Key Features

### 1. Sample Data Injection

The most impactful feature. Before generating SQL, the agent queries the
database to extract sample values from every column. This solves a common
failure mode: LLMs often guess at data formats. For example, without sample
data, the model would generate `WHERE state = 'California'` when the actual
value stored is `'CA'`. By showing the model real values, it learns the correct
formats without being explicitly told.

### 2. Self-Correction with Error Feedback

When a generated query fails validation, the agent does not simply retry with
the same prompt. Instead, it appends the failed SQL and the exact error message
to the conversation history, then asks the model to fix it. This creates a
feedback loop where each attempt is informed by previous failures. The agent
allows up to 3 attempts before returning the best effort.

### 3. Few-Shot Examples

The prompt includes 5 hand-crafted question-to-SQL examples that cover:

- Simple counts and filters
- Multi-table JOINs with aggregations
- Revenue calculations using discount logic
- Name-based lookups across related tables
- GROUP BY with ORDER BY and LIMIT

These examples anchor the model's output format and demonstrate the expected
level of SQL complexity.

### 4. Thinking Model Support

Models like qwen3 produce `<think>...</think>` blocks before their actual
response. The agent detects and strips these blocks automatically, along with
markdown code fences and multi-statement outputs. This makes the agent
compatible with both standard and reasoning-style models.

### 5. Relevance Classification (Included, Not Active)

The codebase includes an LLM-based classifier (`_is_relevant_question`) that
determines whether a question can be answered by the bike store database. This
feature was built for production use cases where users might ask unrelated
questions. It is not called during query generation to avoid the risk of
false negatives on valid but unusually phrased questions during evaluation.

## Design Decisions

### Model Selection

qwen3:32b was chosen after testing multiple models including llama3.2, llama3.3,
and deepseek-r1:70b. qwen3:32b offered the best balance of SQL accuracy and
response time on the Carleton LLM server. The model name is configurable via
the OLLAMA_MODEL environment variable, so evaluators can substitute any
Ollama-compatible model.

### Validation Strategy

The agent uses DuckDB's `EXPLAIN` command rather than actually executing
queries for validation. This confirms the SQL is syntactically valid and all
referenced tables and columns exist, without risking side effects or
unnecessary computation on large result sets.

### Environment Flexibility

All configuration (Ollama host, API key, model name) is loaded from
environment variables with sensible defaults. This means the agent works
out of the box with a local Ollama installation but can be pointed at any
compatible server by setting the appropriate variables.

## Testing

The agent was tested against a wide range of query types:

- Simple aggregations: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
- Filtering: `WHERE` clauses with dates, strings, and numeric values
- Multi-table JOINs: up to 5 tables in a single query
- Subqueries: customers who spent more than the average
- Window functions and ranking
- NULL handling: customers without phone numbers
- Date ranges: orders between specific months
- Negative conditions: products never sold
- Edge cases: empty inputs, irrelevant questions
