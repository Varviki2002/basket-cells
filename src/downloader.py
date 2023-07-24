import os

import requests

from src import PROJECT_PATH


class Downloader:
    def __init__(self, gdrive_id: str = None, file_name: str = None):
        """
        Constructor for downloading and loading the data file.
        :param str gdrive_id: Google Drive id
        :param str file_name: file name for saving
        """
        if gdrive_id is not None and file_name is not None:
            gdrive_link = "https://drive.google.com/uc?export=download&id="
            data_folder = os.path.join(PROJECT_PATH, "data")
            os.path.join(data_folder, "generated")
            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, file_name)
            if not os.path.isfile(file):
                r = requests.get(gdrive_link + gdrive_id, allow_redirects=True)
                open(file, "wb").write(r.content)