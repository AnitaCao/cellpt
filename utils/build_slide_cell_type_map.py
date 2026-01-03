# build_slide_cell_type_map.py
import json
from pathlib import Path
import pandas as pd

CSV_IN = "/hpc/group/jilab/rz179/cellpt/combined/withBackground/combined_meta_human_only_20to1.csv"
OUT_JSON = "/hpc/group/jilab/rz179/cellpt/combined/withBackground/slide_cell_type_map.json"

SLIDE = "slide_id"
FINE = "cell_type"
COARSE = "cell_type_coarse"

def main():
    df = pd.read_csv(CSV_IN)
    for c in [SLIDE, FINE, COARSE]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV")

    df = df.dropna(subset=[SLIDE])

    per_slide = {}
    for slide, g in df.groupby(SLIDE, sort=True):
        uniques_fine   = sorted(map(str, g[FINE].dropna().unique().tolist()))
        uniques_coarse = sorted(map(str, g[COARSE].dropna().unique().tolist()))
        fine_counts    = {str(k): int(v) for k, v in g[FINE].value_counts(dropna=True).to_dict().items()}
        coarse_counts  = {str(k): int(v) for k, v in g[COARSE].value_counts(dropna=True).to_dict().items()}

        per_slide[str(slide)] = {
            "n_rows": int(len(g)),
            "cell_type": {
                "unique": uniques_fine,
                "counts": fine_counts,
            },
            "cell_type_coarse": {
                "unique": uniques_coarse,
                "counts": coarse_counts,
            },
        }

    # Also emit a compact whitelist that’s handy at inference
    whitelist = {
        slide: {
            "cell_type": d["cell_type"]["unique"],
            "cell_type_coarse": d["cell_type_coarse"]["unique"],
        }
        for slide, d in per_slide.items()
    }

    out_path = Path(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        json.dump(
            {
                "meta": {
                    "csv_path": CSV_IN,
                    "slide_col": SLIDE,
                    "fine_col": FINE,
                    "coarse_col": COARSE,
                    "n_rows_total": int(len(df)),
                    "n_slides": int(len(per_slide)),
                },
                "per_slide": per_slide,
                "slide_whitelist": whitelist,
            },
            f,
            indent=2,
        )

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
