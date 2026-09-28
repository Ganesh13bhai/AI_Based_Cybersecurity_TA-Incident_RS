from pathlib import Path
import json


class DataAcquisitionService:
    """
    Discovers and inventories cybersecurity datasets
    registered in data/metadata/datasets.json.
    """

    SUPPORTED_EXTENSIONS = {
        ".csv": "CSV",
        ".parquet": "Parquet",
        ".json": "JSON",
        ".pcap": "PCAP",
        ".pcapng": "PCAPNG",
    }

    def __init__(self):
        # CyberSecAI project root
        self.project_root = Path(__file__).resolve().parents[3]

        # Dataset registry
        self.registry_path = (
            self.project_root
            / "data"
            / "metadata"
            / "datasets.json"
        )

    def load_registry(self):
        """Load the dataset registry."""

        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Dataset registry not found: {self.registry_path}"
            )

        with open(self.registry_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def discover_files(self, dataset_path):
        """
        Discover supported files inside a dataset directory.

        Supports both:
        - Relative paths inside CyberSecAI
        - Absolute paths outside CyberSecAI
        """

        dataset_path = Path(dataset_path)

        # Convert relative paths to project-relative paths
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path

        if not dataset_path.exists():
            return []

        if not dataset_path.is_dir():
            return []

        discovered = []

        for file_path in dataset_path.rglob("*"):

            # Ignore directories
            if not file_path.is_file():
                continue

            # Ignore macOS metadata files such as ._filename.csv
            if file_path.name.startswith("._"):
                continue

            extension = file_path.suffix.lower()

            # Ignore unsupported files such as .DS_Store
            if extension not in self.SUPPORTED_EXTENSIONS:
                continue

            size_bytes = file_path.stat().st_size

            # If file is inside CyberSecAI, use a relative path.
            # Otherwise keep the absolute path.
            if file_path.is_relative_to(self.project_root):
                path_value = str(
                    file_path.relative_to(self.project_root)
                )
            else:
                path_value = str(file_path)

            discovered.append(
                {
                    "name": file_path.name,
                    "path": path_value,
                    "format": self.SUPPORTED_EXTENSIONS[extension],
                    "extension": extension,
                    "size_bytes": size_bytes,
                    "size_mb": round(
                        size_bytes / (1024 * 1024),
                        2
                    ),
                }
            )

        return discovered

    def scan_datasets(self):
        """Scan every registered dataset."""

        registry = self.load_registry()

        results = []

        for dataset in registry.get("datasets", []):

            # Prefer the actual dataset location if available.
            source_path = dataset.get("source_path")

            scan_path = (
                source_path
                if source_path
                else dataset["raw_path"]
            )

            files = self.discover_files(scan_path)

            results.append(
                {
                    "id": dataset["id"],
                    "name": dataset["name"],
                    "source_type": dataset["source_type"],
                    "registry_status": dataset["status"],
                    "raw_path": dataset["raw_path"],
                    "source_path": source_path,
                    "file_count": len(files),
                    "files": files,
                    "acquisition_status": (
                        "available"
                        if files
                        else "not_available"
                    ),
                }
            )

        return results