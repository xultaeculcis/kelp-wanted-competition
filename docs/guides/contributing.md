This page explains the coding style and general contributing guidelines for the dev team.

## Instructions
There are a couple of rules that you should follow:

1. Use `logging` module instead of `print()` function
2. Try not to leave commended out code
3. If certain functionality is needed in multiple places - please crate a separate module or a class for it and import it in your code.
4. Use notebooks only for quick experimentation, they should not be the source of our production code.
5. Set a guide in your IDE to 120 characters. We use 120 instead of suggested 79 from the **PEP8** Guidelines.
6. We use `mypy` to guard the type annotations.
7. We use `isort`, `black` and `flake8` to maintain **PEP8** compliant code.
8. We use _google_ style for the docstrings.
9. Code review should be done for every PR.
10. We do not merge directly to **main** branch.
11. Always start from **main** branch when branching to new feature branch.
12. We require at least one approve on each PR and all threads to be resolved before the merge.
13. When merging we use **squash before merge** to have clean history
