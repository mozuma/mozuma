import os


def list_files_in_dir(dir_path, allowed_extensions=None):
    """Open all files in a directory and returns the opened objects

    :param dir_path:
    :param mode:
    :param allowed_extensions:
    :return:
    """
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if allowed_extensions is None or any(f.endswith(e) for e in allowed_extensions)
    ]
