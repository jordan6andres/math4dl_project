#!/usr/bin/env bash
# =============================================================================
# setup.sh — DL Math Project (Linux / macOS)
# Creates a virtual environment, installs dependencies, and prints next steps.
# =============================================================================

set -euo pipefail

PROJECT_NAME="dl_math_project"
VENV_DIR="venv"

echo "=========================================="
echo "  Setting up ${PROJECT_NAME}"
echo "=========================================="
echo ""

# 1. Create virtual environment
if [ -d "${VENV_DIR}" ]; then
    echo "[INFO] Virtual environment already exists at ./${VENV_DIR}"
else
    echo "[STEP 1/3] Creating Python virtual environment ..."
    python3 -m venv "${VENV_DIR}"
    echo "[OK] Virtual environment created."
fi

# 2. Activate and install requirements
echo ""
echo "[STEP 2/3] Installing dependencies from requirements.txt ..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
echo "[OK] Dependencies installed."

# 3. Register ipykernel (optional but useful for Jupyter/VS Code)
echo ""
echo "[STEP 3/3] Registering Jupyter kernel ..."
python -m ipykernel install --user --name="${PROJECT_NAME}" --display-name "Python (${PROJECT_NAME})" || true
echo "[OK] Kernel registered."

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "       source ${VENV_DIR}/bin/activate"
echo ""
echo "  2. Download the AG News dataset:"
echo "       https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset"
echo "     Place train.csv and test.csv inside data/"
echo ""
echo "  3. Open the notebook and run all cells:"
echo "       jupyter notebook notebooks/dl_math_project.ipynb"
echo ""
echo "  4. (Optional) Launch the Streamlit demo:"
echo "       streamlit run app/app.py"
echo ""
