import os


def create_folder_structure(base_dir):
    # Define the folder structure
    structure = {
        "task-1": ['data', 'scripts', 'notebooks', 'reports', 'visualizations'],
        "task-2": ['data', 'scripts', 'notebooks', 'models', 'reports', 'visualizations'],
        "task-3": ['data', 'scripts', 'notebooks', 'reports', 'visualizations'],
        "logs": [],
        "models": [],
        "data": []
    }

    # Create the base directory
    os.makedirs(base_dir, exist_ok=True)

    # Create the subfolders for each task
    for parent, subfolders in structure.items():
        parent_dir = os.path.join(base_dir, parent)
        os.makedirs(parent_dir, exist_ok=True)
        
        for subfolder in subfolders:
            os.makedirs(os.path.join(parent_dir, subfolder), exist_ok=True)

    # Create essential files
    create_readme(base_dir)
    create_gitignore(base_dir)

    print(f"Project structure created successfully at: {base_dir}")


def create_readme(base_dir):
    readme_path = os.path.join(base_dir, 'README.md')
    content = """# Rossmann Pharmaceuticals - Sales Forecasting Project

## Project Overview
Rossmann Pharmaceuticals aims to forecast store sales across cities. This project involves:
- Exploratory Data Analysis (EDA)
- Machine Learning and Deep Learning model development
- API Deployment for real-time predictions

## Project Tasks
- **Task 1:** Exploratory Data Analysis
- **Task 2:** Model Development and Training
- **Task 3:** API Deployment for Predictions

## How to Use
- Place raw data in `task-1/data/`
- Run analysis scripts in `task-1/scripts/`
- Train models in `task-2/scripts/`
- Deploy and serve models through `task-3/scripts/`
"""
    with open(readme_path, 'w') as file:
        file.write(content)
    print("README.md created.")


def create_gitignore(base_dir):
    gitignore_path = os.path.join(base_dir, '.gitignore')
    content = """
# Ignore Python cache
*.pyc
__pycache__/

# Ignore Jupyter Notebook checkpoints
.ipynb_checkpoints

# Ignore model files and data
models/
data/
logs/

# Ignore virtual environments
venv/
.env
"""
    with open(gitignore_path, 'w') as file:
        file.write(content)
    print(".gitignore created.")


if __name__ == "__main__":
    base_directory = input("Enter the base directory for the project (e.g., rossmann_forecasting): ")
    create_folder_structure(base_directory)

