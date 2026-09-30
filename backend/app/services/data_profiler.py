from pathlib import Path
import json

import pandas as pd


class DataProfilerService:
    """
    Profiles cybersecurity CSV datasets without loading
    the complete dataset into memory.

    Designed for large datasets and limited RAM environments.
    """

    CHUNK_SIZE = 50_000

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[3]

        self.registry_path = (
            self.project_root
            / "data"
            / "metadata"
            / "datasets.json"
        )

        self.schema_path = (
            self.project_root
            / "data"
            / "metadata"
            / "dataset_schema.json"
        )

        self.output_path = (
            self.project_root
            / "data"
            / "metadata"
            / "profile_results.json"
        )

    # ---------------------------------------------------------
    # Registry
    # ---------------------------------------------------------

    def load_registry(self):
        with open(
            self.registry_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    # ---------------------------------------------------------
    # Schema metadata
    # ---------------------------------------------------------

    def load_schema(self):
        if not self.schema_path.exists():
            return {}

        with open(
            self.schema_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    # ---------------------------------------------------------
    # Label candidate detection
    # ---------------------------------------------------------

    def detect_label_candidates(self, columns):
        """
        Detect possible target/label columns.

        This is heuristic only.
        Schema/Label Mapper will make the final decision later.
        """

        keywords = [
            "label",
            "target",
            "class",
            "attack",
            "category",
            "attack_cat"
        ]

        candidates = []

        for column in columns:
            normalized = column.strip().lower()

            for keyword in keywords:
                if keyword in normalized:
                    candidates.append(column)
                    break

        return list(dict.fromkeys(candidates))

    # ---------------------------------------------------------
    # CSV profiling
    # ---------------------------------------------------------

    def profile_csv(self, file_path, dataset_name):
        file_path = Path(file_path)

        file_size = file_path.stat().st_size

        row_count = 0
        chunk_count = 0

        missing_counts = {}
        label_distributions = {}

        column_names = []
        column_types = {}

        numeric_columns = set()
        categorical_columns = set()

        label_candidates = []

        sample_rows = []

        try:

            for chunk in pd.read_csv(
                file_path,
                chunksize=self.CHUNK_SIZE,
                low_memory=False
            ):

                chunk_count += 1

                if not column_names:
                    column_names = list(chunk.columns)

                    label_candidates = (
                        self.detect_label_candidates(
                            column_names
                        )
                    )

                row_count += len(chunk)

                # ---------------------------------------------
                # Missing values
                # ---------------------------------------------

                for column in chunk.columns:

                    missing = int(
                        chunk[column].isna().sum()
                    )

                    missing_counts[column] = (
                        missing_counts.get(column, 0)
                        + missing
                    )

                # ---------------------------------------------
                # Data types
                # ---------------------------------------------

                for column in chunk.columns:

                    dtype = str(
                        chunk[column].dtype
                    )

                    if column not in column_types:
                        column_types[column] = dtype

                    if pd.api.types.is_numeric_dtype(
                        chunk[column]
                    ):
                        numeric_columns.add(column)
                    else:
                        categorical_columns.add(column)

                # ---------------------------------------------
                # Label distributions
                # ---------------------------------------------

                for label_column in label_candidates:

                    if label_column not in chunk.columns:
                        continue

                    counts = (
                        chunk[label_column]
                        .fillna("<MISSING>")
                        .astype(str)
                        .value_counts()
                    )

                    if label_column not in label_distributions:
                        label_distributions[
                            label_column
                        ] = {}

                    for value, count in counts.items():

                        label_distributions[
                            label_column
                        ][value] = (
                            label_distributions[
                                label_column
                            ].get(value, 0)
                            + int(count)
                        )

                # ---------------------------------------------
                # Sample rows
                # ---------------------------------------------

                if len(sample_rows) < 5:

                    remaining = 5 - len(sample_rows)

                    sample = chunk.head(
                        remaining
                    )

                    sample_rows.extend(
                        sample.to_dict(
                            orient="records"
                        )
                    )

        except Exception as error:

            return {
                "status": "error",
                "dataset": dataset_name,
                "file": file_path.name,
                "path": str(file_path),
                "error": str(error)
            }

        # -----------------------------------------------------
        # Missing percentage
        # -----------------------------------------------------

        missing_percentages = {}

        for column, count in missing_counts.items():

            if row_count > 0:

                missing_percentages[column] = round(
                    (count / row_count) * 100,
                    2
                )

            else:

                missing_percentages[column] = 0.0

        # -----------------------------------------------------
        # Convert label distributions to JSON-safe structure
        # -----------------------------------------------------

        class_distributions = {}

        for label, distribution in (
            label_distributions.items()
        ):

            class_distributions[label] = {
                "classes": len(distribution),
                "distribution": distribution
            }

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        return {
            "status": "success",

            "dataset": dataset_name,

            "file": file_path.name,

            "path": str(file_path),

            "format": "CSV",

            "size_bytes": file_size,

            "size_mb": round(
                file_size / (1024 * 1024),
                2
            ),

            "rows": row_count,

            "columns": len(column_names),

            "chunk_count": chunk_count,

            "chunk_size": self.CHUNK_SIZE,

            "column_names": column_names,

            "column_types": column_types,

            "numeric_columns": sorted(
                numeric_columns
            ),

            "categorical_columns": sorted(
                categorical_columns
            ),

            "missing_values": missing_counts,

            "missing_percentages": (
                missing_percentages
            ),

            "label_candidates": label_candidates,

            "class_distributions": (
                class_distributions
            ),

            "sample_rows": sample_rows
        }

    # ---------------------------------------------------------
    # Dataset profiling
    # ---------------------------------------------------------

    def profile_datasets(self):

        registry = self.load_registry()

        results = []

        for dataset in registry.get(
            "datasets",
            []
        ):

            dataset_name = dataset["name"]

            source_path = dataset.get(
                "source_path"
            )

            if not source_path:
                continue

            source_path = Path(source_path)

            if not source_path.exists():
                continue

            # ---------------------------------------------
            # IMPORTANT:
            # Only CSV files are processed.
            # ---------------------------------------------

            csv_files = sorted(
                source_path.rglob("*.csv")
            )

            for file_path in csv_files:

                # Ignore macOS metadata files
                if file_path.name.startswith("._"):
                    continue

                print(
                    f"Profiling: "
                    f"{dataset_name} → "
                    f"{file_path.name}"
                )

                result = self.profile_csv(
                    file_path,
                    dataset_name
                )

                results.append(result)

        output = {
            "status": "success",
            "dataset_count": len(
                set(
                    result["dataset"]
                    for result in results
                    if "dataset" in result
                )
            ),
            "file_count": len(results),
            "files": results
        }

        # Save metadata
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False
            )

        return output
