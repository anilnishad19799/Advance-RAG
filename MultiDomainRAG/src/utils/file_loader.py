# src/utils/file_loader.py

import os
from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders import PyMuPDFLoader
import shutil


class FileLoader:
    """
    Unified file loader for PDF, TXT, or other supported files.
    Automatically saves files to `data/` folder (project root).
    """

    def __init__(self, file_path: str, save_dir: str = "../../data"):
        self.original_path = Path(file_path)
        if not self.original_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Absolute path for save_dir relative to file_loader.py
        self.save_dir = Path(__file__).parent.joinpath(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save a copy in data folder
        self.saved_path = self.save_dir / self.original_path.name
        shutil.copy(str(self.original_path), str(self.saved_path))

    def load(self) -> List[Document]:
        ext = self.saved_path.suffix.lower()

        if ext == ".pdf":
            loader = PyMuPDFLoader(str(self.saved_path))
            docs = loader.load()
        elif ext == ".txt":
            # Manual read to handle encoding issues
            try:
                with open(self.saved_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                # fallback if utf-8 fails
                with open(self.saved_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            docs = [Document(page_content=text)]
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: PDF, TXT")

        # Optional preprocessing
        for doc in docs:
            doc.page_content = doc.page_content.strip()

        return docs
