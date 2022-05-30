import os
import re


def get_project_root_dir() -> str:
    project_root_dir = __file__
    src_folder = "src"
    assert src_folder in project_root_dir, project_root_dir
    project_root_dir = re.split(src_folder, project_root_dir)[0]
    assert os.path.isdir(project_root_dir)
    return project_root_dir
