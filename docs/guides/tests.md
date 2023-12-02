This page describes how to run tests locally using `pytest`.

## Instructions

To run tests marked as `unit` tests:

```shell
pytest -m "unit" -v
```

To run tests marked as `integration` tests:

```shell
pytest -m "integration" -v
```

To run tests marked as `e2e` tests:

```shell
pytest -m "e2e" -v
```

To run all tests:

```shell
pytest
```

> NOTE: Pre-commit hooks will only run those tests marked as `unit`.
