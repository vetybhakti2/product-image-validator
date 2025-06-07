# Image Validator App

This project is a FastAPI application for product image validation.

## Project Structure
```
project/
├── route/
│ ├── init.py
│ └── main.py
├── service/
│ ├── init.py
│ └── image_validator_api.py
├── init.py
├── main.py
├── handler_fastapi.py
└── pyproject.toml
```

## Installation

Make sure you have [Poetry](https://python-poetry.org/) installed.

```bash
poetry install
```

## Running the Application
Run the ASGI server (e.g., using uvicorn):
```bash
poetry run uvicorn project.route.main:app --reload
```

## Brief Description
route/main.py contains the FastAPI endpoint definitions.

service/image_validator_api.py contains the business logic for image validation.

handler_fastapi.py includes handlers or middleware (optional).

main.py is the application entry point if applicable.

# Contributing
Feel free to fork this repository and submit pull requests for new features or bug fixes.

# License
This project is licensed under the MIT License (or specify your license).
