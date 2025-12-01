#!/usr/bin/env python3

"""
Generate graphs from a TEKF.csv file
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_beams_and_slopes(tekf_df, output_filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot 1: 4 beam values
    beams = ["TEKF.beam_fl", "TEKF.beam_fr", "TEKF.beam_rl", "TEKF.beam_rr"]
    for beam in beams:
        axes[0].plot(tekf_df["timestamp"], tekf_df[beam], label=beam, alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel("Range (m)")
    axes[0].set_title("DVL Beam Ranges")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Slopes
    axes[1].plot(tekf_df["timestamp"], tekf_df["TEKF.sn"], label="Unfiltered (sn)", alpha=0.5, linewidth=0.5)
    axes[1].plot(tekf_df["timestamp"], tekf_df["TEKF.ekf_sn"], label="Filtered (ekf_sn)", linewidth=1.0)
    axes[1].plot(tekf_df["timestamp"], tekf_df["TEKF.se"], label="Unfiltered (se)", alpha=0.5, linewidth=0.5)
    axes[1].plot(tekf_df["timestamp"], tekf_df["TEKF.ekf_se"], label="Filtered (ekf_se)", linewidth=1.0)
    axes[1].set_ylabel("Slope")
    axes[1].set_title("Slopes")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved graph to {output_filename}")
    plt.close(fig)


def plot_outliers(tekf_df, output_filename, zoom, window):
    if zoom is not None:
        df_zoom = tekf_df[(tekf_df["timestamp"] >= zoom - window) & (tekf_df["timestamp"] <= zoom + window)]
    else:
        df_zoom = tekf_df

    if df_zoom.empty:
        print(f"No data found around timestamp {zoom}, skipping outlier graph.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    # Plot 1: 4 beam values
    beams = ["TEKF.beam_fl", "TEKF.beam_fr", "TEKF.beam_rl", "TEKF.beam_rr"]
    for beam in beams:
        axes[0].plot(df_zoom["timestamp"], df_zoom[beam], label=beam, alpha=0.7, linewidth=0.7)
    axes[0].set_ylabel("Range (m)")
    axes[0].set_title("DVL Beam Ranges")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: slope north
    axes[1].plot(df_zoom["timestamp"], df_zoom["TEKF.sn"], label="sn (unfiltered)", alpha=0.7, linewidth=1.0)
    axes[1].plot(df_zoom["timestamp"], df_zoom["TEKF.ekf_sn"], label="ekf_sn (filtered)", linewidth=1.0)
    axes[1].set_ylabel("Slope")
    axes[1].set_title("Slope North")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: rov_up, terr_up, ekf_terr_up
    axes[2].plot(df_zoom["timestamp"], df_zoom["TEKF.rov_up"], label="rov_up", linestyle="--", alpha=0.7, linewidth=1.0)
    axes[2].plot(df_zoom["timestamp"], df_zoom["TEKF.terr_up"], label="terr_up (unfiltered)", alpha=0.7, linewidth=1.0)
    axes[2].plot(df_zoom["timestamp"], df_zoom["TEKF.ekf_terr_up"], label="ekf_terr_up (filtered)", linewidth=1.0)
    axes[2].set_ylabel("Elevation (m)")
    axes[2].set_title("Terrain Elevation")
    axes[2].set_xlabel("Timestamp")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved graph to {output_filename}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tekf_csv", help="Path to the TEKF CSV file")
    parser.add_argument("--zoom", type=float, default=None, help="Zoom in on a specific time")
    parser.add_argument("--window", type=float, default=10, help="Zoom window size")

    args = parser.parse_args()

    if not os.path.exists(args.tekf_csv):
        print(f"Error: File {args.tekf_csv} not found.")
        return

    try:
        tekf_df = pd.read_csv(args.tekf_csv)
    except Exception as e:
        print(f"Error reading {args.tekf_csv}: {e}")
        return

    base_name = os.path.splitext(args.tekf_csv)[0]
    if base_name.endswith("_TEKF"):
        base_name = base_name[:-5]

    # General results
    plot_beams_and_slopes(tekf_df, f"{base_name}_graph1.pdf")

    # Outlier rejection
    plot_outliers(tekf_df, f"{base_name}_graph2.pdf", args.zoom, args.window)


if __name__ == "__main__":
    main()
