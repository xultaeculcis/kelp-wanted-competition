Here you can find detailed documentation about the API reference including public functions and classes
that can be imported and used in your own scripts.

The diagram below shows the source code structure for the `kelp` package.

```
kelp
├── consts                               <- Constants
│   ├── data.py
│   ├── directories.py
│   ├── __init__.py
│   ├── logging.py
│   └── reproducibility.py
├── core                                 <- Common classes and functions shared across modules
│   ├── configs
│   │   ├── argument_parsing.py
│   │   ├── base.py
│   │   └── __init__.py
│   ├── device.py
│   ├── indices.py
│   ├── __init__.py
│   ├── settings.py
│   └── submission.py
├── data_prep                            <- Data preparation scripts
│   ├── aoi_grouping.py
│   ├── calculate_band_stats.py
│   ├── dataset_prep.py
│   ├── eda.py
│   ├── __init__.py
│   ├── move_split_files.py
│   ├── plot_samples.py
│   ├── sahi_dataset_prep.py
│   └── train_val_test_split.py
├── __init__.py
├── nn                                   <- Segmentation Neural Network stuff
│   ├── data
│   │   ├── band_stats.py
│   │   ├── datamodule.py
│   │   ├── dataset.py
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   └── utils.py
│   ├── inference
│   │   ├── average_predictions.py
│   │   ├── fold_weights.py
│   │   ├── __init__.py
│   │   ├── predict_and_submit.py
│   │   ├── predict.py
│   │   ├── preview_submission.py
│   │   └── sahi.py
│   ├── __init__.py
│   ├── models
│   │   ├── efficientunetplusplus
│   │   │   ├── decoder.py
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   ├── factories.py
│   │   ├── fcn
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── modules.py
│   │   ├── resunet
│   │   │   ├── decoder.py
│   │   │   ├── __init__.py
│   │   │   └──model.py
│   │   ├── resunetplusplus
│   │   │   ├── decoder.py
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   └── segmentation.py
│   └── training
│       ├── config.py
│       ├── eval_from_folders.py
│       ├── eval.py
│       ├── __init__.py
│       ├── options.py
│       └── train.py
├── py.typed
├── utils                                <- Utils
│   ├── gpu.py
│   ├── __init__.py
│   ├── logging.py
│   ├── mlflow.py
│   ├── plotting.py
│   └── serialization.py
└── xgb                                  <- XGBoost stuff
   ├── inference
   │   ├── __init__.py
   │   ├── predict_and_submit.py
   │   └── predict.py
   ├── __init__.py
   └── training
      ├── cfg.py
      ├── eval.py
      ├── __init__.py
      ├── options.py
      └── train.py
```
