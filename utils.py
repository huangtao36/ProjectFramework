import os


def mkdirs(paths):
    """
    :param paths: str or str-list
    :return: None
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)
