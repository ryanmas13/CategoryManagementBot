from llama_index.core import SimpleDirectoryReader
import os

class CustomDirectoryReader(SimpleDirectoryReader):
    def __init__(self, directory, exclude_files=None, recursive=True):
        super().__init__(directory, recursive=recursive)
        self.exclude_files = exclude_files if exclude_files else []

    def load_data(self):
        # Override load_data to exclude specific files
        # all_files = super().load_data()
        all_files = super().list_resources()
        filtered_files = [f for f in all_files if os.path.basename(f) not in self.exclude_files]
        documents = []
        for filePath in filtered_files:
            document = super().load_data(filePath)
            documents.append(document[0])
        return documents, filtered_files