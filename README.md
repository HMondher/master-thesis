#### Academic Context
This code package supports the Master's thesis:
- **Title**: "Time Series Database Evaluation for Aerospace Applications"
- **Methodology**: Extended FoxBench framework for TSDB benchmarking
- **Contribution**: Systematic evaluation framework for aerospace TSDB selection
- **Results**: Comprehensive performance analysis across four major TSDB systems

### Overview
This repository contains the complete and authentic code implementation for the Master's thesis " Benchmarking Time Series Database On Aerospace Data Streams ".
The code package generates all datasets, benchmark results, and visualisations presented in the thesis, ensuring full reproducibility and academic integrity.

###  Installation and Setup

#### Prerequisites
- Python 3.8 or higher
- Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

### Core Components

#### 1. `aerospace_data_generator.py` - Synthetic Data Generation
**Purpose**: Generate realistic aerospace sensor data matching NASA LISOTD lightning detection patterns

**Key Features**:
- **Realistic Sensor Simulation**: 8 aerospace sensor types (flash_rate, temperature, pressure, velocity, altitude, magnetic_field, radiation_level, power_consumption)
- **Spatial Grid Generation**: Global coverage with 2.5° resolution matching LISOTD satellite data
- **Temporal Patterns**: Seasonal and diurnal variations with configurable noise and anomalies
- **Scalable Data Volumes**: Configurable dataset sizes for performance testing
- **Reproducible Results**: Fixed random seeds ensure consistent output

**Thesis Connection**: Produces the exact synthetic datasets described in Chapter 4:
- Daily scale: 8.6 million records
- Weekly scale: 60.4 million records  
- Monthly scale: 259.2 million records

**Usage Examples**:
```bash
# Generate daily dataset (8.6M records)
python aerospace_data_generator.py --scale daily --sensors 100 --format csv

# Generate weekly dataset (60.4M records)
python aerospace_data_generator.py --scale weekly --sensors 100 --format csv

# Generate monthly dataset (259.2M records)
python aerospace_data_generator.py --scale monthly --sensors 100 --format csv

# Custom configuration
python aerospace_data_generator.py --scale daily --sensors 50 --frequency 0.5 --anomaly-rate 0.1
```

**Output Files**:
- `synthetic_data/aerospace_synthetic_{scale}.csv` - Main dataset
- `synthetic_data/metadata_{scale}.json` - Dataset characteristics and statistics

#### 2. `foxbench_execution.py` - FoxBench-TSDB Framework
**Purpose**: Complete implementation of the FoxBench-TSDB extension for systematic TSDB benchmarking

**Framework Components**:
- **5-Layer Architecture Analysis**: Systematic comparison of TSDB technology stacks
  - Layer 1: File System (NTFS/XFS)
  - Layer 2: Storage Engine (TSM, Columnar, PostgreSQL+Hypertables, LSM Tree)
  - Layer 3: Query Engine (InfluxQL/Flux, SQL, PostgreSQL+TimescaleDB, TQL)
  - Layer 4: Application Interface (HTTP API, PostgreSQL Wire Protocol, Native Protocol)
  - Layer 5: Application (Python clients)

- **40-Query Workload**: Comprehensive query patterns following FoxBench methodology
  - Point Queries (Q1-Q8): High selectivity, low complexity
  - Temporal Range Queries (Q9-Q16): Medium selectivity, medium complexity
  - Spatial Range Queries (Q17-Q24): Geographic bounding box filtering
  - Value Filter Queries (Q25-Q32): Threshold-based filtering
  - Multi-dimensional Queries (Q33-Q40): Combined filters, high complexity

- **Performance Analysis**: Statistical evaluation and ranking
  - Category-wise performance comparison
  - Overall performance scoring (1000/avg_query_time)
  - Architecture impact analysis
  - Systematic conclusions and recommendations

**Thesis Connection**: Implements the complete methodology framework described in Chapter 3

**Usage**:
```bash
python foxbench_execution.py
```

**Output Files**:
- `foxbench_tsdb_results.json` - Complete benchmark results and analysis
- `foxbench_tsdb_results_table.csv` - Performance matrix (FoxBench Table 6 style)
- `foxbench_tsdb_analysis.png` - Comprehensive visualisation dashboard

#### 3. `results_figures.py` - Thesis Figure Generation
**Purpose**: Generate exact figures 4.1-4.4 presented in thesis Chapter 4 with precise data values

**Generated Figures**:
- **Figure 4.1**: Ingestion Throughput Comparison (NASA LISOTD Dataset)
  - GridDB: 95,011 records/sec
  - InfluxDB: 11,245 records/sec
  - TimescaleDB: 3,585 records/sec
  - QuestDB: 1,255 records/sec

- **Figure 4.2**: Query Performance Comparison (Log Scale)
  - Range Queries: [16.77, 0.892, 0.049, 0.167] seconds
  - Aggregation Queries: [210, 15.23, 0.17, 0.203] seconds

- **Figure 4.3**: Storage Compression Ratio Comparison
  - QuestDB: 8.7:1 compression ratio
  - GridDB: 6.5:1 compression ratio
  - InfluxDB: 4.8:1 compression ratio
  - TimescaleDB: 3.1:1 compression ratio

- **Figure 4.4**: Scalability Performance Across Temporal Scales
  - Performance retention patterns for datasets: 8.6M → 60.4M → 259.2M records
  - Linear vs non-linear scaling characteristics

**Key Features**:
- **Professional Formatting**: LaTeX-compatible fonts and high-resolution output (300 DPI)
- **Exact Data Values**: Matches thesis tables and results precisely
- **Publication Quality**: Optimised for academic papers and presentations
- **Consistent Styling**: Professional color scheme and formatting

**Usage**:
```bash
python results_figures.py
```

**Output Files**:
- `Figure_4_1_Ingestion_Throughput.png`
- `Figure_4_2_Query_Performance.png`
- `Figure_4_3_Storage_Compression.png`
- `Figure_4_4_Scalability_Trends.png`












