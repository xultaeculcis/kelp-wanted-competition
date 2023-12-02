## Configs

The base classes for step entrypoint configs.

### Base classes

The base classes for step entrypoint configs.

::: kelp.core.configs.base

### Argument parsing

The argument parsing helper functions for the script entrypoint arguments.

::: kelp.core.configs.argument_parsing

## Settings

The application settings.

::: kelp.core.settings

### Importing settings

To use current settings without the need to parse them each time you can use:

```python
import logging

from kelp.core.settings import current_settings


# log current environment
logging.info(current_settings.env)  # INFO:dev
```
