# Contributing to `pyClarion`

Thank you for your interest in contributing to `pyClarion`!

## General Instructions

Please use Github issues for all communications, including bug reports, feature requests, and general questions.

Before working on a new feature, please discuss it with project maintainers to make sure it is appropriate.

When submitting a pull request, please make sure the following criteria are fulfilled.
- Target issues are referenced in the description
- Significant changes are reported in the changelog

## Coding Conventions

- Follow PEP 8 coding style
- Lint using mypy
- Include type annotations 
- Use reStructuredText for markup in docstrings (keep it light)
- Express preconditions, postconditions, invariants etc. using exceptions and assertions
- Limit dependencies to Python stdlib wherever possible
- Focus testing on functional requirements