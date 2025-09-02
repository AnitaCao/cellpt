#!/usr/bin/env python3
"""
Per-slide coarse label augmentation.

For each slide:
1) Read per-cell fine map:  <fine_dir>/<slide>.csv with columns [cellid, celltype]
2) Read per-cell coarse map: <coarse_dir>/<slide>.csv with columns [cellid, celltype]
3) Read per-slide meta: <meta_root>/<slide>/meta/nucleus_shapes.csv (expects columns cell_id, cell_type)
4) Add column cell_type_coarse to the meta CSV:
   - If per-cell coarse exists for the row’s cell_id, use it
   - Else use per-slide majority coarse for that fine label (computed from joined maps)
   - Else set cell_type_coarse = "not_found"
5) If within the same slide a fine label maps to multiple coarse labels, write fine_to_coarse_ambiguities.csv

Usage:
  python per_slide_coarse.py \
    --slides_file /hpc/group/jilab/rz179/XeniumData/slides.txt \
    --meta_root   /hpc/group/jilab/rz179/cellpt/nucleus_data \
    --fine_dir    /hpc/group/jilab/hz/xenium/celltype \
    --coarse_dir  /hpc/group/jilab/hz/xenium/celltype_lowres \
    --out_suffix  _with_coarse.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def load_slide_list(slides_file: Path) -> list[str]:
    lines = slides_file.read_text().splitlines()
    slides: list[str] = []
    for s in lines:
        s = s.strip()
        if not s or s.startswith("#"):
            continue
        slides.append(s.rstrip())
    if not slides:
        raise ValueError("No slides found in slides file")
    return slides


def read_map(p: Path, col_alias=("cellid", "celltype")) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    a, b = col_alias
    if a not in df or b not in df:
        raise ValueError(f"{p} missing {a},{b}")
    out = df[[a, b]].copy()
    out[a] = out[a].astype(str).str.strip()
    out[b] = out[b].astype(str).str.strip()
    return out.rename(columns={a: "cellid", b: "celltype"})


def add_coarse_column(meta_csv: Path, fine_map_csv: Path, coarse_map_csv: Path, out_path: Path) -> pd.Series:
    # load per-slide meta and maps
    df = pd.read_csv(meta_csv)
    if "cell_id" not in df.columns:
        raise RuntimeError(f"{meta_csv} missing cell_id")
    if "cell_type" not in df.columns:
        df["cell_type"] = ""

    fine = read_map(fine_map_csv)       # per-cell fine for this slide
    coarse = read_map(coarse_map_csv)   # per-cell coarse for this slide

    # join per-cell pairs for this slide only
    pairs = fine.merge(coarse, on="cellid", how="inner", suffixes=("_fine", "_coarse"))

    # count within-slide mappings fine -> coarse, ignore unknowns and blanks for the check
    counts = pd.DataFrame()
    if not pairs.empty:
        pairs_use = pairs.copy()
        pairs_use["coarse_norm"] = pairs_use["celltype_coarse"].str.strip().str.lower()
        pairs_use = pairs_use[~pairs_use["coarse_norm"].isin({"", "unknown"})]
        if not pairs_use.empty:
            counts = (
                pairs_use
                .value_counts(["celltype_fine", "celltype_coarse"])
                .rename("cnt").reset_index()
                .sort_values(["celltype_fine", "cnt", "celltype_coarse"], ascending=[True, False, True])
            )
            totals = counts.groupby("celltype_fine", as_index=False)["cnt"].sum().rename(columns={"cnt": "total"})
            counts = counts.merge(totals, on="celltype_fine", how="left")
            counts["share"] = counts["cnt"] / counts["total"]

    # find fines that map to more than one coarse within this slide
    if not counts.empty:
        tmp = counts.groupby("celltype_fine").size().reset_index(name="ncoarse")
        amb_fines = tmp[tmp["ncoarse"] > 1]["celltype_fine"]
        if not amb_fines.empty:
            amb = counts[counts["celltype_fine"].isin(amb_fines)].copy()
            amb = amb.rename(columns={"celltype_fine": "fine", "celltype_coarse": "coarse"})
            amb_file = out_path.with_name("fine_to_coarse_ambiguities.csv")
            amb[["fine", "coarse", "cnt", "total", "share"]].to_csv(amb_file, index=False)
            print(f"{meta_csv.name}: within slide ambiguous fine labels = {amb_fines.nunique()}  wrote {amb_file}")
            with pd.option_context("display.max_rows", 20, "display.width", 200):
                print(amb.sort_values(["fine", "cnt"], ascending=[True, False]).head(20).to_string(index=False))
        else:
            print(f"{meta_csv.name}: no within slide ambiguities")
    else:
        print(f"{meta_csv.name}: no fine–coarse pairs to analyze")
        
        
    # write per-slide fine→coarse mapping table
    # collect all fine labels seen on this slide (from per-cell fine map and meta)
    fine_set = (
        set(fine["celltype"].astype(str).str.strip().unique())
        | set(df["cell_type"].astype(str).str.strip().unique())
    )
    fine_set.discard("")  # drop blanks in mapping table

    # prepare resolved mapping with counts and share
    mapping_cols = ["fine", "coarse_resolved", "resolved_count", "total", "share", "status"]
    if not counts.empty:
        best = (
            counts.sort_values(["celltype_fine","cnt","celltype_coarse"], ascending=[True,False,True])
                  .drop_duplicates("celltype_fine", keep="first")
                  .rename(columns={
                      "celltype_fine": "fine",
                      "celltype_coarse": "coarse_resolved",
                      "cnt": "resolved_count"
                  })
        )
        best["status"] = best["share"].apply(lambda s: "ok" if s >= 0.7 else "ambiguous")
        best = best[["fine","coarse_resolved","resolved_count","total","share","status"]]
    else:
        best = pd.DataFrame(columns=mapping_cols)

    mapping = pd.DataFrame({"fine": sorted(fine_set)}).merge(best, on="fine", how="left")
    mapping["coarse_resolved"] = mapping["coarse_resolved"].fillna("unknown_coarse")
    mapping[["resolved_count","total"]] = mapping[["resolved_count","total"]].fillna(0).astype(int)
    mapping["share"] = mapping["share"].fillna(0.0)
    mapping["status"] = mapping["status"].fillna("unknown_coarse")

    mapping_file = out_path.with_name("fine_to_coarse.csv")
    mapping.to_csv(mapping_file, index=False)
    print(f"{meta_csv.name}: wrote per-slide map {mapping_file}")    
        
        

    # per-slide majority map fine -> coarse for fallback use
    f2c = {}
    if not counts.empty:
        best = counts.drop_duplicates("celltype_fine", keep="first")
        best = best.rename(columns={"celltype_fine": "fine", "celltype_coarse": "coarse"})
        f2c = dict(zip(best["fine"], best["coarse"]))

    # per-cell coarse dict, treat blank or 'unknown' as missing
    coarse_clean = coarse.copy()
    coarse_clean["celltype"] = coarse_clean["celltype"].astype(str).str.strip()
    coarse_clean.loc[coarse_clean["celltype"].str.lower().isin({"", "unknown"}), "celltype"] = ""
    percell_coarse = dict(zip(coarse_clean["cellid"], coarse_clean["celltype"]))

    # choose coarse for each row in meta
    def choose(row):
        cid = str(row["cell_id"]).strip()
        fine_label = str(row.get("cell_type", "")).strip()
        if cid in percell_coarse and percell_coarse[cid]:
            return percell_coarse[cid]
        if fine_label in f2c:
            return f2c[fine_label]
        return "unknown_coarse"

    df["cell_type_coarse"] = df.apply(choose, axis=1)
    # coverage summary
    n = len(df)
    n_percell = int((df["cell_id"].astype(str).isin(coarse_clean["cellid"])).sum())
    n_unknown = int((df["cell_type_coarse"] == "unknown_coarse").sum())
    n_fallback = n - n_percell - n_unknown
    print(f"{meta_csv.name}: coarse fill per-cell={n_percell} fallback={n_fallback} unknown={n_unknown} of {n}")
    
    
    df.to_csv(out_path, index=False)
    return df["cell_type_coarse"].value_counts()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_file", required=True, help="Text file with slide names, one per line")
    ap.add_argument("--meta_root", required=True, help="Root with per-slide folders containing meta/nucleus_shapes.csv")
    ap.add_argument("--meta_name",  default="nucleus_shapes.csv", help="Meta CSV filename under each slide's meta/")
    ap.add_argument("--fine_dir", required=True, help="Dir with fine per-cell maps, CSV per slide")
    ap.add_argument("--coarse_dir", required=True, help="Dir with coarse per-cell maps, CSV per slide")
    ap.add_argument("--out_suffix", default="_with_coarse.csv", help="Suffix for output CSV next to meta")
    args = ap.parse_args()

    slides = load_slide_list(Path(args.slides_file))
    meta_root = Path(args.meta_root)
    fine_dir = Path(args.fine_dir)
    coarse_dir = Path(args.coarse_dir)

    for slide in slides:
        meta = meta_root / slide / "meta" / args.meta_name
        fine = fine_dir / f"{slide}.csv"
        coarse = coarse_dir / f"{slide}.csv"

        if not meta.exists():
            print(f"{slide}: SKIP meta not found: {meta}")
            continue
        if not fine.exists():
            print(f"{slide}: SKIP fine map not found: {fine}")
            continue
        if not coarse.exists():
            print(f"{slide}: SKIP coarse map not found: {coarse}")
            continue

        out = meta.with_name(meta.stem + args.out_suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        vc = add_coarse_column(meta, fine, coarse, out)
        print(f"{slide}: wrote {out}")
        print(vc.to_string())
        print("-" * 60)


if __name__ == "__main__":
    main()
