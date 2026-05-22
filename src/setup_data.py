"""
src/setup_data.py

A standalone helper script that downloads the AG News Classification dataset
directly from GitHub, loads it with pandas, maps numeric labels to category
names, and prints a quick summary.

Run this once before opening the notebook so that data/train.csv and
data/test.csv are ready on disk.
"""

# =============================================================================
# STEP 1 — Import the deep-learning framework
# =============================================================================

# TensorFlow is the library we use for building neural networks.
# We import it here so we can print the version and check whether a GPU
# is available (GPUs make training much faster than a CPU).
import tensorflow as tf

# =============================================================================
# STEP 2 — Import data-science and standard-library utilities
# =============================================================================

# pandas is the standard library for working with tabular data (DataFrames).
# We use it to load the downloaded CSV files and inspect them.
import pandas as pd

# numpy provides fast numerical operations on arrays.  It is often used
# behind the scenes by pandas and TensorFlow, so we import it even though
# this script only needs it indirectly.
import numpy as np

# os lets us interact with the operating system — creating folders, building
# file paths, and checking whether files already exist.
import os

# urllib.request is part of Python's standard library and gives us a simple
# way to download files from the internet without installing extra tools.
import urllib.request


# =============================================================================
# STEP 3 — Define a function to print TensorFlow version and GPU info
# =============================================================================

def print_tf_info():
    """
    Print the installed TensorFlow version and whether a GPU is detected.

    Knowing the TF version helps when debugging compatibility issues.
    GPU availability tells us whether model training will be fast (GPU)
    or slower (CPU).
    """
    # tf.__version__ is a string like "2.15.0" that tells us which
    # TensorFlow release is installed.
    print(f"TensorFlow version: {tf.__version__}")

    # tf.config.list_physical_devices('GPU') returns a list of GPU devices.
    # If the list is empty, no GPU was detected by TensorFlow.
    gpus = tf.config.list_physical_devices("GPU")

    # Convert the list to a boolean: True if at least one GPU exists.
    gpu_available = bool(gpus)
    print(f"GPU available: {gpu_available}")

    # If a GPU is present, print its name so the user knows which device
    # TensorFlow will use for accelerated training.
    if gpu_available:
        # gpus[0].name usually looks like "/physical_device:GPU:0"
        print(f"GPU device name: {gpus[0].name}")


# =============================================================================
# STEP 4 — Define a function to download a single file from a URL
# =============================================================================

def download_file(url, destination):
    """
    Download a file from the internet and save it to a local path.

    Parameters
    ----------
    url : str
        The web address of the file to download.
    destination : str
        The local file path where the downloaded file should be saved.

    Returns
    -------
    bool
        True if the download succeeded, False if it failed.
    """
    try:
        # urlretrieve opens the URL, reads the data, and writes it to disk.
        # It is the simplest built-in way to download a file in Python.
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded -> {destination}")
        return True
    except Exception:
        # If anything goes wrong (no internet, URL is down, permission error),
        # we catch the exception so the program does not crash.
        print("Download failed. Please check your internet connection.")
        return False


# =============================================================================
# STEP 5 — Define a function to download both AG News CSV files
# =============================================================================

def download_ag_news():
    """
    Download the AG News train.csv and test.csv from GitHub.

    The files are hosted in a public repository that mirrors the original
    AG News dataset.  We download them into the data/ folder.
    """
    # Create the data/ folder if it does not already exist.
    # exist_ok=True means Python will not raise an error if the folder
    # is already there.
    os.makedirs("data", exist_ok=True)

    # These are the direct download URLs for the two CSV splits.
    train_url = (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
        "master/data/ag_news_csv/train.csv"
    )
    test_url = (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
        "master/data/ag_news_csv/test.csv"
    )

    # Build the local file paths where the downloads will be saved.
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    print("Downloading AG News dataset ...\n")

    # Download the training file.
    success_train = download_file(train_url, train_path)

    # Download the test file.
    success_test = download_file(test_url, test_path)

    # If either download failed, stop the script early.
    if not (success_train and success_test):
        print("\nCould not download required files. Exiting.")
        return None, None

    return train_path, test_path


# =============================================================================
# STEP 6 — Define a function to load the CSVs into pandas DataFrames
# =============================================================================

def load_dataframes(train_path, test_path):
    """
    Load the downloaded CSV files into pandas DataFrames.

    The AG News CSVs from this source do not have a header row, so we
    manually supply the column names: class_index, title, description.

    Parameters
    ----------
    train_path : str
        Path to train.csv on disk.
    test_path : str
        Path to test.csv on disk.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df and test_df.
    """
    # names=... tells pandas what to call each column because the file
    # does not contain column names on the first line.
    train_df = pd.read_csv(train_path, names=["class_index", "title", "description"])
    test_df = pd.read_csv(test_path, names=["class_index", "title", "description"])

    return train_df, test_df


# =============================================================================
# STEP 7 — Define a function to map numeric labels to category names
# =============================================================================

def add_category_names(train_df, test_df):
    """
    Add a human-readable 'category' column to both DataFrames.

    In this version of the dataset the labels are 1-based:
      1 -> World
      2 -> Sports
      3 -> Business
      4 -> Sci/Tech

    Adding a text column makes the data easier to understand when we print
    samples or generate reports later.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split with a numeric 'class_index' column.
    test_df : pd.DataFrame
        Test split with a numeric 'class_index' column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The modified train_df and test_df.
    """
    # A dictionary that translates each integer label into its
    # corresponding category name.
    label_map = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech",
    }

    # .map() looks up every value in the 'class_index' column inside label_map
    # and returns the matching string.  The result is stored in a new
    # column called 'category'.
    train_df["category"] = train_df["class_index"].map(label_map)
    test_df["category"] = test_df["class_index"].map(label_map)

    return train_df, test_df


# =============================================================================
# STEP 8 — Define a function to print a summary report
# =============================================================================

def print_summary(train_df, test_df):
    """
    Print dataset shapes, class distribution, and the first few rows.

    This gives the user immediate feedback that everything worked correctly.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame (must contain a 'category' column).
    test_df : pd.DataFrame
        Test DataFrame (must contain a 'category' column).
    """
    # .shape returns a tuple (number_of_rows, number_of_columns).
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("\nClass distribution in training set:")

    # value_counts() counts how many times each category appears.
    # sort_index() orders the results alphabetically for consistent output.
    counts = train_df["category"].value_counts().sort_index()
    print(counts)

    print("\nFirst 3 rows of the training set:")
    # head(3) shows the first 3 rows so the user can see what the data
    # actually looks like.
    print(train_df.head(3))


# =============================================================================
# STEP 9 — Main entry point (ties everything together)
# =============================================================================

def main():
    """
    Run the full data-setup pipeline:
      1. Print TF version / GPU info.
      2. Download AG News CSV files.
      3. Load them into pandas DataFrames.
      4. Map labels to category names.
      5. Print summary.
    """
    # ---- 1. System info ----
    print_tf_info()

    # ---- 2. Download ----
    train_path, test_path = download_ag_news()

    # If download failed, train_path will be None, so we stop here.
    if train_path is None or test_path is None:
        return

    # ---- 3. Load into DataFrames ----
    train_df, test_df = load_dataframes(train_path, test_path)

    # ---- 4. Label mapping ----
    train_df, test_df = add_category_names(train_df, test_df)

    # ---- 5. Summary ----
    print()
    print_summary(train_df, test_df)

    print("\nAll done! You can now open notebooks/dl_math_project.ipynb.")


# =============================================================================
# STEP 10 — Guard that only runs main() when this file is executed directly
# =============================================================================

# When Python imports a file (e.g. `import setup_data`), it sets __name__ to
# the module name.  When you run the file directly (e.g.
# `python src/setup_data.py`), __name__ becomes "__main__".  This guard
# prevents main() from running accidentally if the file is ever imported
# as a module elsewhere.
if __name__ == "__main__":
    main()
