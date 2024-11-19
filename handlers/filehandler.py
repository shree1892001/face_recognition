import os

def ensure_directory_exists(directory):
    """
    Ensure the given directory exists. Create it if it does not.
    :param directory: Directory path to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Created directory: {directory}")
