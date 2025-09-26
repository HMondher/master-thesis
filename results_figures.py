#!/usr/bin/env python3
"""
Results FIGURES GENERATOR

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Professional color scheme for LaTeX/print compatibility
PROFESSIONAL_COLORS = {
    'QuestDB': '#2E4057',      # Dark blue-gray
    'InfluxDB': '#8B4A6B',     # Muted purple
    'TimescaleDB': '#C7522A',  # Burnt orange
    'GridDB': '#5F7A61'        # Forest green
}

# Configure matplotlib for professional output
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8
})

def create_figure_4_1_ingestion_throughput():
    """Figure 4.1: Ingestion Throughput Comparison (NASA LISOTD Dataset)"""
    print("Creating Figure 4.1: Ingestion Throughput Comparison...")
    
    # NASA LISOTD ingestion data (40M+ records)
    databases = ['QuestDB', 'InfluxDB', 'TimescaleDB', 'GridDB']
    throughput = [1255, 11245, 3585, 95011]  # records/second
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(databases, throughput, 
                  color=[PROFESSIONAL_COLORS[db] for db in databases],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Ingestion Throughput (Records/Second)', fontweight='bold')
    ax.set_xlabel('Time Series Database System', fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, throughput):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # Set y-axis to start from 0 and add some headroom
    ax.set_ylim(0, max(throughput) * 1.15)
    
    plt.tight_layout()
    plt.savefig('Figure_4_1_Ingestion_Throughput.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   Saved: Figure_4_1_Ingestion_Throughput.png")

def create_figure_4_2_query_performance():
    """Figure 4.2: Query Performance Comparison (Seconds, Log Scale)"""
    print("Creating Figure 4.2: Query Performance Comparison...")
    
    databases = ['QuestDB', 'InfluxDB', 'TimescaleDB', 'GridDB']
    range_queries = [16.77, 0.892, 0.049, 0.167]  # seconds
    aggregation_queries = [210, 15.23, 0.17, 0.203]  # seconds
    
    x = np.arange(len(databases))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width/2, range_queries, width, 
                   label='Range Queries', alpha=0.8, edgecolor='black', linewidth=1,
                   color='#4A6741')  # Dark green
    bars2 = ax.bar(x + width/2, aggregation_queries, width,
                   label='Aggregation Queries', alpha=0.8, edgecolor='black', linewidth=1,
                   color='#8B4A6B')  # Muted purple
    
    ax.set_ylabel('Query Response Time (Seconds, Log Scale)', fontweight='bold')
    ax.set_xlabel('Time Series Database System', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(databases)
    ax.legend()
    
    # Add value labels
    for bars, values in [(bars1, range_queries), (bars2, aggregation_queries)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{value:.3f}' if value < 1 else f'{value:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('Figure_4_2_Query_Performance.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   Saved: Figure_4_2_Query_Performance.png")

def create_figure_4_3_storage_compression():
    """Figure 4.3: Storage Compression Ratio Comparison"""
    print("Creating Figure 4.3: Storage Compression Ratio Comparison...")
    
    databases = ['QuestDB', 'InfluxDB', 'TimescaleDB', 'GridDB']
    compression_ratios = [8.7, 4.8, 3.1, 6.5]  # compression ratio (X:1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(databases, compression_ratios,
                  color=[PROFESSIONAL_COLORS[db] for db in databases],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Compression Ratio (X:1)', fontweight='bold')
    ax.set_xlabel('Time Series Database System', fontweight='bold')
    
    # Add value labels and economic impact
    economic_impact = ['53% cost reduction', 'Adequate compression', 
                      'Lowest compression', 'Balanced efficiency']
    
    for bar, value, impact in zip(bars, compression_ratios, economic_impact):
        height = bar.get_height()
        # Main value on top
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}:1', ha='center', va='bottom', fontweight='bold')
        # Economic impact inside bar
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                impact, ha='center', va='center', fontweight='bold',
                fontsize=9, color='white')
    
    ax.set_ylim(0, max(compression_ratios) * 1.2)
    
    plt.tight_layout()
    plt.savefig('Figure_4_3_Storage_Compression.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   Saved: Figure_4_3_Storage_Compression.png")

def create_figure_4_4_scalability_trends():
    """Figure 4.4: Scalability Performance Across Temporal Scales"""
    print("Creating Figure 4.4: Scalability Performance Across Temporal Scales...")
    
    # Synthetic data volumes and performance trends
    scales = ['Daily\n(8.6M)', 'Weekly\n(60.4M)', 'Monthly\n(259.2M)']
    data_volumes = [8.6, 60.4, 259.2]  # Million records
    
    # Performance degradation patterns (normalized to daily baseline = 100%)
    questdb_performance = [100, 98, 95]      # Linear scaling
    influxdb_performance = [100, 94, 87]     # Slight non-linear
    timescaledb_performance = [100, 97, 94]  # Linear scaling  
    griddb_performance = [100, 92, 78]       # Non-linear degradation
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Performance trends
    x = np.arange(len(scales))
    
    ax1.plot(x, questdb_performance, marker='o', linewidth=3, markersize=8,
             label='QuestDB', color=PROFESSIONAL_COLORS['QuestDB'])
    ax1.plot(x, influxdb_performance, marker='s', linewidth=3, markersize=8,
             label='InfluxDB', color=PROFESSIONAL_COLORS['InfluxDB'])
    ax1.plot(x, timescaledb_performance, marker='^', linewidth=3, markersize=8,
             label='TimescaleDB', color=PROFESSIONAL_COLORS['TimescaleDB'])
    ax1.plot(x, griddb_performance, marker='D', linewidth=3, markersize=8,
             label='GridDB', color=PROFESSIONAL_COLORS['GridDB'])
    
    ax1.set_ylabel('Performance Retention (%)', fontweight='bold')
    ax1.set_xlabel('Dataset Scale', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales)
    ax1.legend()
    ax1.set_ylim(70, 105)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Data volume scaling
    bars = ax2.bar(scales, data_volumes,
                   color=['#2E4057', '#8B4A6B', '#C7522A'],  # Professional gradient
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Dataset Size (Million Records)', fontweight='bold')
    ax2.set_xlabel('Temporal Scale', fontweight='bold')
    
    # Add value labels
    for bar, value in zip(bars, data_volumes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{value:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylim(0, max(data_volumes) * 1.15)
    
    plt.tight_layout()
    plt.savefig('Figure_4_4_Scalability_Trends.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("   Saved: Figure_4_4_Scalability_Trends.png")

def create_all_chapter4_figures():
    """Generate all Chapter 4 figures"""
    print("GENERATING CHAPTER 4 FOCUSED FIGURES")
    print("=" * 50)
    
    create_figure_4_1_ingestion_throughput()
    create_figure_4_2_query_performance()
    create_figure_4_3_storage_compression()
    create_figure_4_4_scalability_trends()
    
    print("\n" + "=" * 50)
    print("ALL CHAPTER 4 FIGURES COMPLETED")
    print("Files created:")
    print("- Figure_4_1_Ingestion_Throughput.png")
    print("- Figure_4_2_Query_Performance.png") 
    print("- Figure_4_3_Storage_Compression.png")
    print("- Figure_4_4_Scalability_Trends.png")
 
if __name__ == "__main__":
    create_all_chapter4_figures()

