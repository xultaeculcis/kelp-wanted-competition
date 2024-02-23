## Requirements

In order to set up local development environment make sure you have installed:

* [conda](https://docs.conda.io/en/latest/miniconda.html)
* [conda-lock](https://github.com/conda/conda-lock)

You can install `conda-lock` on your `base` environment by running:

```shell
conda install -c conda-forge conda-lock -n base
```

## Using Makefile

Run:

```shell
make local-env
```

It will also install `pre-commit` hooks and the project in an editable mode.
Once done you can activate the environment by running:

```shell
conda activate kelp
```

## Manually

1. Run `conda-lock` command:
    ```shell
    conda-lock install --mamba -n kelp conda-lock.yml
    ```

2. Activate the env:
    ```shell
    conda activate kelp
    ```

3. Install `pre-commit` hooks:
    ```shell
    pre-commit install
    ```

4. Install the project in an editable mode:
    ```shell
    pip install -e .
    ```

## Pre-commit hooks
This project uses `pre-commit` package for managing and maintaining `pre-commit` hooks.

To ensure code quality - please make sure that you have it configured.

1. Install `pre-commit` and following packages: `isort`, `black`, `flake8`, `mypy`, `pytest`.

2. Install `pre-commit` hooks by running: `pre-commit install`

3. The command above will automatically run formatters, code checks and other steps defined in the `.pre-commit-config.yaml`

4. All of those checks will also be run whenever a new commit is being created i.e. when you run `git commit -m "blah"`

5. You can also run it manually with this command: `pre-commit run --all-files`

You can manually disable `pre-commit` hooks by running: `pre-commit uninstall` Use this only in exceptional cases.

## Setup environmental variables

> NOTE: The .env files are only needed if you plan to run Azure ML Pipelines.

Ask your colleagues for `.env` files which aren't included in this repository and put them inside the repo's root directory.

To see what variables you need see the `.env-sample` file.

## Torch-ORT support (optional)

Optionally, you can enable `torch-ort` support by configuring it via `Makefile` command:

```shell
make configure-torch-ort
```

Make sure `torch-ort` is in the `conda-lock` file before doing so!
