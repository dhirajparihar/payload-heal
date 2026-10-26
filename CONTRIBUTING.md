# Contributing to payload-heal

Thank you for your interest in contributing to `payload-heal`! We welcome bug fixes, documentation improvements, security enhancements, and benchmark datasets.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dhirajparihar/payload-heal.git
   cd payload-heal
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies in editable mode**:
   ```bash
   pip install -e ".[dev]"
   ```

---

## Running the Test Suite

Run all unit, integration, and security tests:

```bash
pytest -v
```

Before submitting a PR, ensure all tests pass and coverage is maintained or expanded.

---

## Pull Request Guidelines

1. **Focus on Safety & Correctness First**:
   Automatic healing must never silently corrupt data or coerce incompatible types. If a payload mapping is ambiguous, fail safe and allow standard Pydantic validation to trigger.
2. **Include Tests**:
   Add test coverage for any new heuristics, edge-case path syntax (e.g., bracketed keys, dots, array indices), or security controls.
3. **Preserve Privacy**:
   Any new telemetry or outbound AI prompt modifications must adhere to recursive PII and credential sanitization.
4. **Follow Formatting & Typing**:
   Ensure full type annotations and clean docstrings conform to repository standards.

---

## Reporting Security Issues

Please **do not** open public GitHub issues for security vulnerabilities. Instead, report security concerns directly to the project maintainers via GitHub Security Advisories or by emailing `contact@dhirajparihar.com`.
