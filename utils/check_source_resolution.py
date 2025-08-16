#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

def analyze_source_vs_extracted_resolution(meta_csv, num_samples=20):
    """
    Compare source slide resolution vs extracted nucleus resolution
    to determine if re-extraction at higher resolution would help
    """
    
    df = pd.read_csv(meta_csv)
    df = df[df["skipped_reason"].fillna("") == ""].head(num_samples)
    
    print("Analyzing source vs extracted resolution...")
    print("="*60)
    
    results = []
    
    for idx, row in df.iterrows():
        try:
            # Load extracted nucleus image
            nucleus_img = Image.open(row['img_path']).convert('L')
            nucleus_array = np.array(nucleus_img)
            
            # Find nucleus bounding box in extracted image
            threshold = np.percentile(nucleus_array[nucleus_array > 0], 50) * 0.3 if np.any(nucleus_array > 0) else 10
            rows, cols = np.where(nucleus_array > threshold)
            
            if len(rows) == 0:
                continue
                
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            nucleus_width_px = max_col - min_col + 1
            nucleus_height_px = max_row - min_row + 1
            
            # Calculate nucleus size in microns (assuming 0.2125 µm/px)
            current_mpp = 0.2125
            nucleus_width_um = nucleus_width_px * current_mpp
            nucleus_height_um = nucleus_height_px * current_mpp
            
            # Estimate what size this nucleus would be at different resolutions
            at_half_mpp = nucleus_width_um / (current_mpp / 2)  # 2x zoom equivalent
            at_quarter_mpp = nucleus_width_um / (current_mpp / 4)  # 4x zoom equivalent
            
            result = {
                'img_path': row['img_path'],
                'nucleus_width_px_current': nucleus_width_px,
                'nucleus_height_px_current': nucleus_height_px,
                'nucleus_width_um': nucleus_width_um,
                'nucleus_height_um': nucleus_height_um,
                'width_at_2x_resolution': at_half_mpp,
                'width_at_4x_resolution': at_quarter_mpp,
                'current_mpp': current_mpp
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {row['img_path']}: {e}")
            continue
    
    if not results:
        print("No valid nucleus images found for analysis")
        return
    
    # Convert to DataFrame for analysis
    analysis_df = pd.DataFrame(results)
    
    # Summary statistics
    print(f"Analysis of {len(analysis_df)} nucleus images:")
    print(f"\nCurrent extraction (0.2125 µm/px):")
    print(f"  Average nucleus width: {analysis_df['nucleus_width_px_current'].mean():.1f} ± {analysis_df['nucleus_width_px_current'].std():.1f} pixels")
    print(f"  Average nucleus height: {analysis_df['nucleus_height_px_current'].mean():.1f} ± {analysis_df['nucleus_height_px_current'].std():.1f} pixels")
    
    print(f"\nPhysical size:")
    print(f"  Average nucleus width: {analysis_df['nucleus_width_um'].mean():.1f} ± {analysis_df['nucleus_width_um'].std():.1f} µm")
    print(f"  Average nucleus height: {analysis_df['nucleus_height_um'].mean():.1f} ± {analysis_df['nucleus_height_um'].std():.1f} µm")
    
    print(f"\nIf extracted at 2x resolution (0.10625 µm/px):")
    print(f"  Nucleus would be: {analysis_df['width_at_2x_resolution'].mean():.1f} ± {analysis_df['width_at_2x_resolution'].std():.1f} pixels wide")
    
    print(f"\nIf extracted at 4x resolution (0.053125 µm/px):")
    print(f"  Nucleus would be: {analysis_df['width_at_4x_resolution'].mean():.1f} ± {analysis_df['width_at_4x_resolution'].std():.1f} pixels wide")
    
    # Recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    avg_width_current = analysis_df['nucleus_width_px_current'].mean()
    avg_width_2x = analysis_df['width_at_2x_resolution'].mean()
    avg_width_4x = analysis_df['width_at_4x_resolution'].mean()
    
    print(f"\n1. Current approach (post-zoom 2x):")
    print(f"   - Quick and easy")
    print(f"   - Nucleus becomes ~{avg_width_current * 2:.0f} pixels wide")
    print(f"   - Some interpolation artifacts")
    
    print(f"\n2. Re-extract at 0.10625 µm/px:")
    print(f"   - Nucleus would be ~{avg_width_2x:.0f} pixels wide")
    print(f"   - Native resolution, no interpolation")
    print(f"   - Requires re-extraction time")
    
    if avg_width_2x > 100:
        print(f"   ✅ RECOMMENDED: {avg_width_2x:.0f}px nuclei would have good detail")
    elif avg_width_2x > 60:
        print(f"   ⚠️  MAYBE: {avg_width_2x:.0f}px nuclei would be adequate")
    else:
        print(f"   ❌ LIMITED BENEFIT: {avg_width_2x:.0f}px nuclei still quite small")
    
    print(f"\n3. Re-extract at 0.053125 µm/px (4x):")
    print(f"   - Nucleus would be ~{avg_width_4x:.0f} pixels wide")
    if avg_width_4x > 150:
        print(f"   ✅ EXCELLENT: {avg_width_4x:.0f}px nuclei would have excellent detail")
        print(f"   - But larger file sizes and processing time")
    elif avg_width_4x > 100:
        print(f"   ✅ GOOD: {avg_width_4x:.0f}px nuclei would have good detail")
    else:
        print(f"   ⚠️  Still relatively small at {avg_width_4x:.0f}px")
    
    # Decision matrix
    print(f"\n" + "="*60)
    print("DECISION MATRIX:")
    print("="*60)
    
    if avg_width_current < 30:
        print("Current nuclei are very small (<30px)")
        if avg_width_2x > 80:
            print("✅ Re-extraction at 2x would provide significant benefit")
        else:
            print("⚠️  Post-zoom is probably sufficient for initial testing")
    else:
        print("Current nuclei are reasonable size (>30px)")
        print("✅ Post-zoom is likely sufficient")
        print("Re-extraction would be marginal improvement")
    
    return analysis_df

def check_slide_metadata(meta_csv):
    """Check if slide metadata contains original resolution info"""
    df = pd.read_csv(meta_csv)
    
    print("Checking slide metadata for resolution information...")
    
    # Look for resolution-related columns
    res_columns = [col for col in df.columns if any(term in col.lower() 
                  for term in ['mpp', 'resolution', 'pixel', 'micron', 'um', 'scale'])]
    
    if res_columns:
        print(f"Found resolution-related columns: {res_columns}")
        for col in res_columns:
            if df[col].notna().any():
                print(f"  {col}: {df[col].dropna().iloc[0]} (first non-null value)")
    else:
        print("No obvious resolution columns found")
    
    # Check for any metadata that might indicate source resolution
    if 'slide_mpp' in df.columns:
        slide_mpp = df['slide_mpp'].iloc[0]
        current_mpp = 0.2125
        if slide_mpp < current_mpp:
            zoom_factor = current_mpp / slide_mpp
            print(f"Source slide resolution: {slide_mpp:.6f} µm/px")
            print(f"Current extraction: {current_mpp:.6f} µm/px")
            print(f"You're already downsampling by {zoom_factor:.1f}x")
            print("✅ Re-extraction at higher resolution would provide real benefit")
        else:
            print(f"Source slide resolution: {slide_mpp:.6f} µm/px")
            print("⚠️  Source may already be at limit")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_csv", required=True, help="Path to nucleus_shapes.csv")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to analyze")
    args = parser.parse_args()
    
    # Check metadata first
    check_slide_metadata(args.meta_csv)
    print("\n")
    
    # Analyze nucleus sizes
    analysis_df = analyze_source_vs_extracted_resolution(args.meta_csv, args.num_samples)
    
    if analysis_df is not None:
        analysis_df.to_csv('resolution_analysis.csv', index=False)
        print(f"\nDetailed analysis saved to 'resolution_analysis.csv'")