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

* **lock** - Creates conda-lock file

    ```shell
    make lock
    ```

### Creating environments

* **env** - Creates env from conda-lock file

    ```shell
    make env
    ```

* **setup-pre-commit** - Installs pre-commit hooks

    ```shell
    make setup-pre-commit
    ```

* **setup-editable** - Installs the project in an editable mode

    ```shell
    make setup-editable
    ```

* **configure-torch-ort** - Configures torch-ort

    ```shell
    make configure-torch-ort
    ```

* **local-env** - Creates local environment and installs pre-commit hooks

    ```shell
    make local-env
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

### Data prep

The commands in this section expect the data to be in certain directories.
Please see Makefile definition for more details.

* **sample-plotting** - Runs tile plotting

    ```shell
    make sample-plotting
    ```

* **aoi-grouping** - Runs AOI grouping

    ```shell
    make aoi-grouping
    ```

* **eda** - Runs EDA

    ```shell
    make eda
    ```

* **calculate-band-stats** - Runs band statistics calculation

    ```shell
    make calculate-band-stats
    ```

* **train-val-test-split-cv** - Runs train-val-test split using cross validation

    ```shell
    make train-val-test-split-cv
    ```

* **train-val-test-split-random** - Runs train-val-test split using random split

    ```shell
    make train-val-test-split-random
    ```


### Model training

The commands in this section accept arguments that can be modified from command line.
Please see the Makefile definition for more details.

* **train** - Trains single CV split

    ```shell
    make train
    ```

* **train-all-splits** - Trains on all splits

    ```shell
    make train-all-splits
    ```

### Model evaluation

The commands in this section accept arguments that can be modified from command line.
Please see the Makefile definition for more details.

* **eval** - Runs evaluation for selected run

    ```shell
    make eval
    ```

* **eval-many** - Runs evaluation for specified runs

    ```shell
    make eval-many
    ```

* **eval-from-folders** - Runs evaluation by comparing predictions to ground truth mask

    ```shell
    make eval-from-folders
    ```

* **eval-ensemble** - Runs ensemble evaluation

    ```shell
    make eval-ensemble
    ```

### Making submissions

The commands in this section accept arguments that can be modified from command line.
Please see the Makefile definition for more details.

* **predict** - Runs prediction

    ```shell
    make predict
    ```

* **submission** - Generates submission file

    ```shell
    make submission
    ```

* **predict-and-submit** - Runs inference and generates submission file

    ```shell
    make predict-and-submit
    ```

* **average-predictions** - Runs prediction averaging

    ```shell
    make average-predictions
    ```

* **cv-predict** - Runs inference on specified folds, averages the predictions and generates submission file

    ```shell
    make cv-predict
    ```
