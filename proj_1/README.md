my_ml_project/
├── .gitignore
├── README.md
├── setup.py
├── requirements.txt
|
├── config/
│   └── paths_config.yaml    # Stores all file and directory names
|
├── data/
│   ├── raw/
│   │   └── raw_data.csv     # Input data
│   └── processed/           # Split/cleaned data goes here (will be created by pipeline)
|
├── models/                  # Trained models and preprocessors saved here
|
├── notebooks/               # For EDA and rapid prototyping
|
├── src/                     # THE CORE APPLICATION CODE
│   ├── __init__.py          # Marks 'src' as a Python package
│   |
│   ├── components/          # Pipeline steps (the modules we discussed)
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   |
│   ├── utils/               # Helper functions (path and config management)
│   │   ├── __init__.py
│   │   ├── paths.py         # Path construction utility
│   │   └── config_loader.py # YAML loading utility
|
└── run_pipeline.py