"""All LLM prompts in one place, so they are easy to tune and to document in the thesis."""

SUMMARY_SYSTEM = (
    "You are a senior software architect documenting a microservice codebase. "
    "Write precise, factual summaries. No preamble, no markdown headings, no speculation."
)

FILE_SUMMARY = """Summarise what this source file does in 1-2 sentences (max 60 words).
Mention its main responsibility, key classes/functions, and any API endpoints, external calls,
databases or message topics it uses.

Service: {service}
File: {path} ({language})
Symbols: {symbols}
Endpoints: {endpoints}

Code (may be truncated):
```
{code}
```

Summary:"""

MODULE_SUMMARY = """Summarise this module (a folder of the '{service}' service) in 1-2 sentences (max 60 words),
based on the summaries of its files.

Module: {path}
Files:
{children}

Summary:"""

SERVICE_SUMMARY = """Describe the microservice '{service}' in 2-4 sentences (max 100 words): its business
responsibility, the main APIs it exposes, the services/infrastructure it depends on and who calls it.

Language: {language}
Exposed endpoints: {endpoints}
gRPC services implemented: {grpc}
Publishes topics: {published}  | Consumes topics: {consumed}
Calls / depends on: {outgoing}
Called by: {incoming}
Parts of the service:
{children}

Description:"""

SYSTEM_SUMMARY = """Write an architecture overview of the system '{project}' in 4-6 sentences (max 150 words):
what the application does, the main services and their roles, how they communicate
(HTTP / gRPC / messaging) and which infrastructure they use.

Services:
{services}

Dependencies:
{edges}

Overview:"""

# ----------------------------------------------------------------- navigation (vectorless)

NAV_SYSTEM = (
    "You navigate the structure of a microservice codebase, like an expert engineer browsing folders, "
    "to find where the answer to a question lives. You only answer with JSON."
)

NAV_SERVICES = """Question: {question}

System overview: {overview}

Services (ID | name | language | summary | endpoints | talks to):
{options}

Choose the services whose code is needed to answer the question (at most {max_select}).
If the question is about the overall architecture, choose the most relevant services and set scope to "architecture".
Scopes: "local" = one service, "cross_service" = interaction between services, "architecture" = whole system.

Respond with JSON only:
{{"selected": ["S1"], "scope": "local", "confidence": 0.8, "reason": "one short sentence"}}"""

NAV_CHILDREN = """Question: {question}

You are inside: {location}
{location_summary}

Entries (ID | type | name | summary):
{options}

Choose up to {max_select} entries that most likely contain the code needed to answer the question.

Respond with JSON only:
{{"selected": ["C1"], "confidence": 0.8, "reason": "one short sentence"}}"""

NAV_SYMBOLS = """Question: {question}

Candidate code symbols (ID | file | kind | name | summary):
{options}

Choose up to {max_select} symbols whose code is needed to answer the question, most relevant first.

Respond with JSON only:
{{"selected": ["F1"], "confidence": 0.8, "reason": "one short sentence"}}"""

# ----------------------------------------------------------------- answering

ANSWER_SYSTEM = """You are an expert software engineer helping a developer understand a microservice codebase.
Rules:
- Answer ONLY from the provided context. If the context is insufficient, say what is missing.
- Cite sources inline as [1], [2] ... matching the numbered code snippets.
- Name services, files, functions and endpoints exactly as they appear.
- For flows across services, describe the sequence step by step, e.g. checkout -> order-service (HTTP POST /orders) -> payment-service.
- Be concise and technical. Use short paragraphs or bullet points and include small code references where helpful."""

ANSWER_USER = """{context}

Question: {question}

Answer (with [n] citations):"""

SERVICE_DOC = """Write developer documentation in Markdown for the microservice '{service}'.

Use these sections: Overview, Responsibilities, API (table of endpoints), Dependencies (outgoing and incoming),
Messaging, Data stores, Key components (important files/classes and what they do), How a typical request flows.
Only use the facts below; write "Not detected" where information is missing.

Facts:
{facts}

Markdown documentation:"""
