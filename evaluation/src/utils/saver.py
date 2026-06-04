"""
Result saver utilities - unified result saving with JSON, pickle support.
"""
import json
import pickle
from pathlib import Path
from typing import Any, Iterable


class ResultSaver:
    """Result saver."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize saver.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _tmp_path(self, filepath: Path) -> Path:
        """Build a temporary path in the same directory for atomic writes."""
        return filepath.with_name(f"{filepath.name}.tmp")
    
    def save_json(self, data: Any, filename: str):
        """
        Save JSON file.
        
        Args:
            data: Data to save
            filename: Filename
        """
        filepath = self.output_dir / filename
        tmp_path = self._tmp_path(filepath)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp_path.replace(filepath)

    def save_jsonl(self, rows: Iterable[Any], filename: str):
        """
        Save JSONL file.

        Args:
            rows: Iterable of row-like objects
            filename: Filename
        """
        filepath = self.output_dir / filename
        tmp_path = self._tmp_path(filepath)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        tmp_path.replace(filepath)

    def load_json(self, filename: str) -> Any:
        """
        Load JSON file.
        
        Args:
            filename: Filename
            
        Returns:
            Loaded data
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_jsonl(self, filename: str) -> list[Any]:
        """
        Load JSONL file.

        Args:
            filename: Filename

        Returns:
            Loaded rows
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    
    def save_pickle(self, data: Any, filename: str):
        """
        Save pickle file.
        
        Args:
            data: Data to save
            filename: Filename
        """
        filepath = self.output_dir / filename
        tmp_path = self._tmp_path(filepath)
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f)
        tmp_path.replace(filepath)
    
    def load_pickle(self, filename: str) -> Any:
        """
        Load pickle file.
        
        Args:
            filename: Filename
            
        Returns:
            Loaded data
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def file_exists(self, filename: str) -> bool:
        """Check if file exists."""
        return (self.output_dir / filename).exists()
