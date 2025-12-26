# Mini LLM Calculator Agent

A **minimal, production-inspired LLM agent** that can:

* Decide whether a user query requires a **calculator tool**
* Execute safe mathematical expressions when needed
* Behave like a **normal conversational LLM** for non-math questions

This project is intentionally simple and explicit to demonstrate **core agent design patterns** used in real-world LLM systems.

---

## Project Goal

The main purpose of this project is to demonstrate:

* **Intent detection** (tool vs non-tool queries)
* **Tool routing** (calculator invocation)
* **Tool result interpretation** using an LLM
* Clean separation of responsibilities between components

---

## High-Level Architecture

```
User Query
   │
   ▼
IntentDetector ──► needs tool?
   │                 │
   │ no              │ yes
   ▼                 ▼
Interpreter     Calculator Tool
   │                 │
   └──────────► Interpreter
                     │
                     ▼
               Final Answer
```

---

## Folder Structure

```
7_calculator_agent
│   main.py
│   README.md
│
├───prompts
│       intent_prompt.txt
│       interpret_prompt.txt
│
├───src
│   │   agent.py
│   │   intent_detector.py
│   │   interpreter.py
│   │
│   ├───tools
│   │      calculator.py
│
├───tests
│       test_agent.py
│       test_calculator.py
│       test_intent.py
```

---

## Core Components

### IntentDetector (`src/intent_detector.py`)

Responsible for deciding **whether a query requires a tool**.

* Uses an LLM prompt (`intent_prompt.txt`) to request structured JSON
* Includes a **regex-based fallback heuristic** for robustness
* Output format:

```json
{
  "tool_name": "calculator" | null,
  "arguments": {"expression": "..."}
}
```

---

### Calculator Tool (`src/tools/calculator.py`)

* Safely evaluates mathematical expressions using Python `ast`
* Prevents arbitrary code execution
* Supports:

  * `+  -  *  /`
  * parentheses
  * `sin`, `cos`, `sqrt`, etc.

Returns structured output:

```json
{"result": 56}
```

or

```json
{"error": "invalid expression"}
```

---

### Interpreter (`src/interpreter.py`)

* Converts tool outputs into **human-readable responses**
* Behaves like a **normal LLM** when no tool is involved
* Uses `interpret_prompt.txt`

---

### Agent (`src/agent.py`)

The orchestrator:

1. Receives user query
2. Detects intent
3. Executes calculator tool if needed
4. Passes result to interpreter
5. Returns final response


---

## Running the Project

### Requirements

* Python **3.12+**
* Local **Ollama** installation
* Model pulled (example):

```bash
ollama pull llama3.1:latest
```

---

### Start the Agent

```bash
python -m main
```

You will see:

```
Mini LLM Calculator Agent
Type 'exit' to quit.
```

---

## Sample Interactive Session

```
Enter your query: Explain me Pythagorean theorem
Agent: A classic topic in mathematics!

The Pythagorean theorem is a fundamental concept in geometry that describes the relationship between the lengths of the sides of a right-angled triangle. It states:

**a² + b² = c²**

where:
- **a** and **b** are the lengths of the two legs (sides) that form the right angle
- **c** is the length of the hypotenuse

Example:
3² + 4² = 5²

The theorem is widely used in mathematics, physics, engineering, and architecture.

Enter your query: Calculate 16*42+18
Agent: I calculated that for you, and the answer comes out to 690.

Enter your query: What is the value of cos(0)?
Agent: Here's the result of that calculation for you, and the answer comes out to 1.0.

Enter your query: What are the main causes of climate change?
Agent: The main causes of climate change can be broadly categorized into human-induced and natural factors...
```

---

## Testing

This project includes a comprehensive test suite to verify the correctness of each layer of the agent system independently and as a whole.

All tests are written using **pytest** and are organized by responsibility.

---

### Test Structure

```text
tests/
├── test_calculator.py   # Calculator tool logic
├── test_intent.py       # Intent detection logic
└── test_agent.py        # End-to-end agent workflow
```

---

### Calculator Tool Tests

These tests validate the **core mathematical tool** in isolation.

Covered functionality:

* Basic arithmetic: addition, subtraction, multiplication, division
* Power operations
* Mathematical functions: `sqrt`, `sin`, `cos`
* Error handling for invalid expressions

Example executed tests:

* `test_add`
* `test_subtract`
* `test_multiply`
* `test_divide`
* `test_power`
* `test_sqrt`
* `test_sin`
* `test_cos`
* `test_error_handling`

All calculator tests pass instantly and do **not** depend on the LLM.

---

### Intent Detection Tests

These tests ensure that the agent correctly decides **when to use the calculator tool** and when not to.

Covered scenarios:

* Simple calculator queries
* Complex mathematical expressions
* Non-mathematical, conversational queries

The intent detector is validated to return:

* `"tool_name": "calculator"` when a calculation is required
* `"tool_name": None` for general knowledge or conversational inputs

These tests involve the local LLM and therefore take longer to execute.

---

### Agent Workflow Tests

These are **end-to-end integration tests** that verify the full agent pipeline:

1. User query
2. Intent detection
3. Optional calculator tool execution
4. Natural language response generation

Covered cases:

* Full calculator workflow (query → calculation → explanation)
* Non-calculator workflow (agent behaves like a normal LLM)

This confirms that:

* The tool is only used when needed
* The agent remains conversational otherwise
* Outputs are human-readable and consistent
---

### Running Tests

Run all tests:

```bash
python -m pytest -v
```

Run individual test files:

```bash
python -m pytest -v tests/test_calculator.py
python -m pytest -v tests/test_intent.py
python -m pytest -v tests/test_agent.py
```

---

### Test Results (Latest)

* Calculator tests: **9 / 9 passed**
* Intent tests: **3 / 3 passed**
* Agent tests: **2 / 2 passed**

---

## Design Decisions

* **No streaming**: keeps focus on agent logic
* **Explicit tool routing**: no hidden magic
* **Simple intent schema**: production-inspired, not academic
* **LLM + heuristic fallback**: robustness over purity