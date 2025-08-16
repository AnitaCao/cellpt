#!/usr/bin/env python3
import argparse
import zarr
import numpy as np
from pathlib import Path
import pandas as pd

def inspect_zarr_cell_id(zarr_path, max_samples=10):
    """Thoroughly inspect the cell_id array in a zarr file"""
    
    print(f"\n{'='*80}")
    print(f"INSPECTING: {zarr_path}")
    print(f"{'='*80}")
    
    try:
        # Open zarr store
        p = Path(zarr_path)
        if p.suffix == ".zip":
            store = zarr.ZipStore(str(p), mode="r")
        else:
            store = zarr.DirectoryStore(str(p))
        
        g = zarr.open(store, mode="r")
        
        # List all available arrays
        print(f"Available arrays in zarr: {list(g.keys())}")
        
        if 'cell_id' not in g:
            print("❌ No 'cell_id' array found!")
            return None
        
        # Get cell_id array info
        cell_id_zarr = g['cell_id']
        print(f"\nCELL_ID ARRAY INFO:")
        print(f"  Shape: {cell_id_zarr.shape}")
        print(f"  Dtype: {cell_id_zarr.dtype}")
        print(f"  Chunks: {cell_id_zarr.chunks}")
        print(f"  Size: {cell_id_zarr.size}")
        
        # Load the actual data
        cell_id_array = np.asarray(cell_id_zarr)
        print(f"  Loaded shape: {cell_id_array.shape}")
        print(f"  Loaded dtype: {cell_id_array.dtype}")
        
        # Inspect based on dimensions
        if cell_id_array.ndim == 1:
            print(f"\n📋 1D ARRAY CONTENTS:")
            print(f"  Total elements: {len(cell_id_array)}")
            
            # Show first N elements
            n_show = min(max_samples, len(cell_id_array))
            print(f"  First {n_show} elements:")
            for i in range(n_show):
                element = cell_id_array[i]
                print(f"    [{i}]: {repr(element)} (type: {type(element).__name__})")
            
            # Show last N elements if array is large
            if len(cell_id_array) > max_samples * 2:
                print(f"  Last {n_show} elements:")
                for i in range(len(cell_id_array) - n_show, len(cell_id_array)):
                    element = cell_id_array[i]
                    print(f"    [{i}]: {repr(element)} (type: {type(element).__name__})")
            
            # Analyze patterns
            print(f"\n  PATTERN ANALYSIS:")
            
            # Check if they're strings with delimiters
            str_elements = [str(x) for x in cell_id_array[:100]]  # Sample first 100
            has_dash = any('-' in s for s in str_elements)
            has_underscore = any('_' in s for s in str_elements)
            has_dot = any('.' in s for s in str_elements)
            
            print(f"    Contains '-': {has_dash}")
            print(f"    Contains '_': {has_underscore}")
            print(f"    Contains '.': {has_dot}")
            
            # Check data types
            if cell_id_array.dtype.kind in ['U', 'S', 'O']:  # String types
                print(f"    String data detected")
                lengths = [len(str(x)) for x in cell_id_array[:100]]
                print(f"    String lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
            elif cell_id_array.dtype.kind in ['i', 'u']:  # Integer types
                print(f"    Integer data detected")
                print(f"    Range: {cell_id_array.min()} to {cell_id_array.max()}")
            
        elif cell_id_array.ndim == 2:
            print(f"\n📋 2D ARRAY CONTENTS:")
            print(f"  Shape: {cell_id_array.shape} (rows x cols)")
            
            # Show first N rows
            n_show = min(max_samples, cell_id_array.shape[0])
            print(f"  First {n_show} rows:")
            for i in range(n_show):
                row = cell_id_array[i]
                print(f"    Row [{i}]: {[repr(x) for x in row]}")
                print(f"             Types: {[type(x).__name__ for x in row]}")
            
            # Show last N rows if array is large
            if cell_id_array.shape[0] > max_samples * 2:
                print(f"  Last {n_show} rows:")
                start_idx = cell_id_array.shape[0] - n_show
                for i in range(start_idx, cell_id_array.shape[0]):
                    row = cell_id_array[i]
                    print(f"    Row [{i}]: {[repr(x) for x in row]}")
            
            # Analyze each column
            print(f"\n  COLUMN ANALYSIS:")
            for col in range(cell_id_array.shape[1]):
                col_data = cell_id_array[:, col]
                print(f"    Column {col}:")
                print(f"      Dtype: {col_data.dtype}")
                print(f"      Sample values: {[repr(x) for x in col_data[:5]]}")
                
                if col_data.dtype.kind in ['U', 'S', 'O']:  # String types
                    lengths = [len(str(x)) for x in col_data[:100]]
                    print(f"      String lengths: min={min(lengths)}, max={max(lengths)}")
                elif col_data.dtype.kind in ['i', 'u']:  # Integer types
                    print(f"      Range: {col_data.min()} to {col_data.max()}")
        
        else:
            print(f"\n❓ UNEXPECTED DIMENSIONS: {cell_id_array.ndim}")
            print(f"  Shape: {cell_id_array.shape}")
            print(f"  First element: {repr(cell_id_array.flat[0])}")
        
        return {
            'path': zarr_path,
            'shape': cell_id_array.shape,
            'dtype': str(cell_id_array.dtype),
            'sample_data': cell_id_array[:min(5, cell_id_array.shape[0])].tolist()
        }
        
    except Exception as e:
        print(f"❌ ERROR inspecting {zarr_path}: {e}")
        return None

def compare_multiple_zarr_files(zarr_paths, max_samples=5):
    """Compare cell_id arrays across multiple zarr files"""
    
    print(f"\n{'='*100}")
    print(f"COMPARING MULTIPLE ZARR FILES")
    print(f"{'='*100}")
    
    results = []
    
    for zarr_path in zarr_paths:
        result = inspect_zarr_cell_id(zarr_path, max_samples)
        if result:
            results.append(result)
    
    # Create summary table
    if results:
        print(f"\n{'='*100}")
        print(f"SUMMARY TABLE")
        print(f"{'='*100}")
        
        df = pd.DataFrame(results)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df[['path', 'shape', 'dtype']])
        
        # Group by format
        print(f"\nFORMAT GROUPS:")
        for name, group in df.groupby(['shape', 'dtype']):
            print(f"  {name}: {len(group)} files")
            for path in group['path']:
                print(f"    - {Path(path).name}")

def main():
    parser = argparse.ArgumentParser(description="Inspect cell_id arrays in Xenium zarr files")
    parser.add_argument("zarr_paths", nargs="+", help="Paths to zarr files or directories")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to show per array")
    parser.add_argument("--compare", action="store_true", help="Compare multiple files")
    
    args = parser.parse_args()
    
    if args.compare and len(args.zarr_paths) > 1:
        compare_multiple_zarr_files(args.zarr_paths, args.max_samples)
    else:
        for zarr_path in args.zarr_paths:
            inspect_zarr_cell_id(zarr_path, args.max_samples)

if __name__ == "__main__":
    main()