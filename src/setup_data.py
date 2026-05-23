"""
src/setup_data.py

A standalone helper script that downloads the AG News Classification dataset
directly from GitHub, loads it with pandas, maps numeric labels to category
names, saves the processed CSVs, and prints a quick summary.

Run this once before opening the notebook so that data/train.csv and
data/test.csv are ready on disk.
"""

import os
import urllib.request

import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------------------------
# Paths — always relative to THIS script so it works no matter where you run it from
# ---------------------------------------------------------------------------

# __file__ is the path to this script (src/setup_data.py).
# os.path.dirname(__file__) gives us the src/ folder.
# os.path.dirname(src/) gives us the project root folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def print_tf_info():
    """
    Print the installed TensorFlow version and whether a GPU is detected.

    Knowing the TF version helps when debugging compatibility issues.
    GPU availability tells us whether model training will be fast (GPU)
    or slower (CPU).

    Returns
    -------
    None
        This function only prints to stdout.
    """
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = bool(gpus)
    print(f"GPU available: {gpu_available}")
    if gpu_available:
        print(f"GPU device name: {gpus[0].name}")


def download_file(url, destination):
    """
    Download a file from the internet and save it locally.

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
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded -> {destination}")
        return True
    except Exception:
        print("Download failed. Please check your internet connection.")
        return False


def download_ag_news():
    """
    Download raw AG News CSVs from GitHub into the project's data/ folder.

    Returns
    -------
    tuple[str, str] | tuple[None, None]
        (train_path, test_path) on success, or (None, None) if either
        download failed.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    train_url = (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
        "master/data/ag_news_csv/train.csv"
    )
    test_url = (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/"
        "master/data/ag_news_csv/test.csv"
    )

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    print("Downloading AG News dataset ...\n")
    success_train = download_file(train_url, train_path)
    success_test = download_file(test_url, test_path)

    if not (success_train and success_test):
        print("\nCould not download required files. Exiting.")
        return None, None

    return train_path, test_path


def load_dataframes(train_path, test_path):
    """
    Load CSV files into pandas DataFrames.

    The raw AG News files do not contain a header row, so we manually
    supply the column names: class_index, title, description.

    Parameters
    ----------
    train_path : str
        Absolute path to train.csv on disk.
    test_path : str
        Absolute path to test.csv on disk.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        train_df and test_df with the three raw columns.
    """
    train_df = pd.read_csv(train_path, names=["class_index", "title", "description"])
    test_df = pd.read_csv(test_path, names=["class_index", "title", "description"])
    return train_df, test_df


def add_category_names(train_df, test_df):
    """
    Map numeric labels 1-4 to human-readable text category names.

    A new column called 'category' is added to both DataFrames.
    The original numeric labels are kept in 'class_index'.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split containing a 'class_index' column with values 1-4.
    test_df : pd.DataFrame
        Test split containing a 'class_index' column with values 1-4.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The modified train_df and test_df (modified in-place, but returned
        for convenience).
    """
    label_map = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech",
    }
    train_df["category"] = train_df["class_index"].map(label_map)
    test_df["category"] = test_df["class_index"].map(label_map)
    return train_df, test_df


def save_csv_files(train_df, test_df):
    """
    Save processed DataFrames back to disk WITH column headers.

    Files are written to data/train.csv and data/test.csv inside the
    project root.  The index is omitted so the CSVs contain only data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Fully processed training DataFrame (includes 'category' column).
    test_df : pd.DataFrame
        Fully processed test DataFrame (includes 'category' column).

    Returns
    -------
    None
        This function only writes files and prints paths.
    """
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved processed files:")
    print(f"  {train_path}")
    print(f"  {test_path}")


def print_summary(train_df, test_df):
    """
    Print dataset shapes, class distribution, and the first few rows.

    This gives immediate feedback that the download and transformation
    pipeline worked correctly.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame (must contain a 'category' column).
    test_df : pd.DataFrame
        Test DataFrame (must contain a 'category' column).

    Returns
    -------
    None
        This function only prints to stdout.
    """
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print("\nClass distribution in training set:")
    print(train_df["category"].value_counts().sort_index())
    print("\nFirst 3 rows:")
    print(train_df.head(3))


def main():
    """
    Run the full data-setup pipeline.

    Steps executed:
      1. Print TF version / GPU info.
      2. Download AG News CSV files from GitHub.
      3. Load them into pandas DataFrames.
      4. Map numeric labels to category names.
      5. Save processed CSVs back to disk.
      6. Print a summary.

    Returns
    -------
    None
    """
    print_tf_info()

    train_path, test_path = download_ag_news()
    if train_path is None or test_path is None:
        return

    train_df, test_df = load_dataframes(train_path, test_path)
    train_df, test_df = add_category_names(train_df, test_df)

    save_csv_files(train_df, test_df)
    print_summary(train_df, test_df)
    print("\nAll done! You can now open notebooks/dl_math_project.ipynb.")


if __name__ == "__main__":
    main()
