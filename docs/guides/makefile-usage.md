In order to use the Makefile commands you need to be on **Linux**.

## Basic usage

Makefile commands are easy to use. Just type `make` in your terminal, hit enter and see the list of available commands.

```shell
make
```

The command above is equivalent to running:

```shell
make help
```

## Development work commands

### Freezing project dependencies

* **conda-lock** - Creates conda-lock file

    ```shell
    make conda-lock
    ```

### Creating environments

* **conda-install** - Creates env from conda-lock file

    ```shell
    make conda-install
    ```

* **setup-pre-commit** - Installs pre-commit hooks

    ```shell
    make setup-pre-commit
    ```

* **setup-editable** - Installs the project in an editable mode

    ```shell
    make setup-editable
    ```

* **setup-local-env** - Creates local environment and installs pre-commit hooks

    ```shell
    make setup-local-env
    ```

### Helper commands

* **format** - Runs code formatting (`isort`, `black`, `flake8`)

    ```shell
    make format
    ```

* **type-check** - Runs type checking with `mypy`

    ```shell
    make type-check
    ```

* **test** - Runs pytest

    ```shell
    make test
    ```

* **testcov** - Runs tests and generates coverage reports

    ```shell
    make testcov
    ```

* **mpc** - Runs manual `pre-commit` stuff

    ```shell
    make mpc
    ```

* **docs** - Builds the documentation

    ```shell
    make docs
    ```

* **pc** - Runs `pre-commit` hooks

    ```shell
    make pc
    ```

* **clean** - Cleans artifacts

    ```shell
    make clean
    ```
