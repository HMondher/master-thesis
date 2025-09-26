#!/usr/bin/env python3
"""
Aerospace Synthetic Data Generator for TSDB Benchmarking
Based on NASA LISOTD Lightning Detection Data Patterns

This module generates synthetic aerospace sensor data that mimics real-world
characteristics for benchmarking Time Series Databases (TSDBs).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass
import argparse

@dataclass
class AerospaceDataConfig:
    """Configuration for aerospace data generation"""
    # Temporal parameters
    start_date: str = "2024-01-01T00:00:00Z"
    duration_days: int = 30
    base_frequency_hz: float = 1.0  # Base sampling rate
    
    # Spatial parameters (global coverage like lightning data)
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    spatial_resolution: float = 2.5  # degrees (like LISOTD)
    
    # Sensor parameters
    num_sensors: int = 100
    sensor_types: List[str] = None
    
    # Data characteristics
    noise_level: float = 0.1
    anomaly_rate: float = 0.05
    seasonal_variation: bool = True
    diurnal_variation: bool = True
    
    # Volume parameters
    data_volume_scale: str = "daily"  # daily, weekly, monthly, yearly
    
    def __post_init__(self):
        if self.sensor_types is None:
            self.sensor_types = [
                "flash_rate",
                "temperature", 
                "pressure",
                "velocity",
                "altitude",
                "magnetic_field",
                "radiation_level",
                "power_consumption"
            ]

class AerospaceDataGenerator:
    """Generate synthetic aerospace sensor data for TSDB benchmarking"""
    
    def __init__(self, config: AerospaceDataConfig):
        self.config = config
        self.np_random = np.random.RandomState(42)  # Reproducible results
        
        # Generate spatial grid (like LISOTD)
        self.spatial_grid = self._create_spatial_grid()
        
        # Define sensor characteristics based on real aerospace data
        self.sensor_characteristics = self._define_sensor_characteristics()
        
    def _create_spatial_grid(self) -> List[Tuple[float, float]]:
        """Create spatial grid points for sensor locations"""
        lats = np.arange(
            self.config.lat_min, 
            self.config.lat_max + self.config.spatial_resolution, 
            self.config.spatial_resolution
        )
        lons = np.arange(
            self.config.lon_min, 
            self.config.lon_max + self.config.spatial_resolution, 
            self.config.spatial_resolution
        )
        
        # Create grid and sample locations
        grid_points = [(lat, lon) for lat in lats for lon in lons]
        
        # Sample subset for manageable data size
        if len(grid_points) > self.config.num_sensors:
            indices = self.np_random.choice(
                len(grid_points), 
                self.config.num_sensors, 
                replace=False
            )
            return [grid_points[i] for i in indices]
        
        return grid_points
    
    def _define_sensor_characteristics(self) -> Dict[str, Dict]:
        """Define realistic sensor characteristics based on aerospace applications"""
        return {
            "flash_rate": {
                "base_value": 0.5,
                "range": (0.0, 50.0),
                "unit": "flashes/km²/year",
                "seasonal_amplitude": 0.3,
                "diurnal_amplitude": 0.2,
                "noise_std": 0.1
            },
            "temperature": {
                "base_value": -40.0,
                "range": (-80.0, 60.0),
                "unit": "°C",
                "seasonal_amplitude": 20.0,
                "diurnal_amplitude": 10.0,
                "noise_std": 2.0
            },
            "pressure": {
                "base_value": 1013.25,
                "range": (300.0, 1100.0),
                "unit": "hPa",
                "seasonal_amplitude": 50.0,
                "diurnal_amplitude": 5.0,
                "noise_std": 1.0
            },
            "velocity": {
                "base_value": 7800.0,
                "range": (7700.0, 7900.0),
                "unit": "m/s",
                "seasonal_amplitude": 10.0,
                "diurnal_amplitude": 5.0,
                "noise_std": 1.0
            },
            "altitude": {
                "base_value": 400000.0,
                "range": (350000.0, 450000.0),
                "unit": "m",
                "seasonal_amplitude": 5000.0,
                "diurnal_amplitude": 1000.0,
                "noise_std": 100.0
            },
            "magnetic_field": {
                "base_value": 25000.0,
                "range": (20000.0, 65000.0),
                "unit": "nT",
                "seasonal_amplitude": 1000.0,
                "diurnal_amplitude": 500.0,
                "noise_std": 50.0
            },
            "radiation_level": {
                "base_value": 0.1,
                "range": (0.01, 10.0),
                "unit": "mSv/h",
                "seasonal_amplitude": 0.02,
                "diurnal_amplitude": 0.01,
                "noise_std": 0.005
            },
            "power_consumption": {
                "base_value": 150.0,
                "range": (100.0, 300.0),
                "unit": "W",
                "seasonal_amplitude": 20.0,
                "diurnal_amplitude": 10.0,
                "noise_std": 5.0
            }
        }
    
    def generate_time_series(self, 
                           sensor_type: str, 
                           location: Tuple[float, float],
                           start_time: datetime,
                           duration: timedelta,
                           frequency_hz: float) -> pd.DataFrame:
        """Generate time series data for a specific sensor and location"""
        
        # Create time index
        total_seconds = int(duration.total_seconds())
        num_points = int(total_seconds * frequency_hz)
        
        time_index = pd.date_range(
            start=start_time,
            periods=num_points,
            freq=f"{1/frequency_hz}s"
        )
        
        # Get sensor characteristics
        sensor_char = self.sensor_characteristics[sensor_type]
        lat, lon = location
        
        # Generate base signal
        base_value = sensor_char["base_value"]
        
        # Add temporal variations
        values = np.full(num_points, base_value)
        
        if self.config.seasonal_variation:
            # Seasonal variation (yearly cycle)
            day_of_year = np.array([t.timetuple().tm_yday for t in time_index])
            seasonal_component = sensor_char["seasonal_amplitude"] * np.sin(
                2 * np.pi * day_of_year / 365.25
            )
            values += seasonal_component
        
        if self.config.diurnal_variation:
            # Diurnal variation (daily cycle)
            hour_of_day = np.array([t.hour + t.minute/60.0 for t in time_index])
            diurnal_component = sensor_char["diurnal_amplitude"] * np.sin(
                2 * np.pi * hour_of_day / 24.0
            )
            values += diurnal_component
        
        # Add spatial variation based on location
        spatial_factor = self._calculate_spatial_factor(sensor_type, lat, lon)
        values *= spatial_factor
        
        # Add noise
        noise = self.np_random.normal(
            0, 
            sensor_char["noise_std"], 
            num_points
        )
        values += noise
        
        # Add anomalies
        if self.config.anomaly_rate > 0:
            num_anomalies = int(num_points * self.config.anomaly_rate)
            anomaly_indices = self.np_random.choice(
                num_points, 
                num_anomalies, 
                replace=False
            )
            
            for idx in anomaly_indices:
                anomaly_magnitude = self.np_random.uniform(2.0, 5.0)
                anomaly_sign = self.np_random.choice([-1, 1])
                values[idx] += anomaly_sign * anomaly_magnitude * sensor_char["noise_std"]
        
        # Clip to realistic range
        values = np.clip(values, sensor_char["range"][0], sensor_char["range"][1])
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': time_index,
            'sensor_type': sensor_type,
            'latitude': lat,
            'longitude': lon,
            'value': values,
            'unit': sensor_char["unit"]
        })
        
        return df
    
    def _calculate_spatial_factor(self, sensor_type: str, lat: float, lon: float) -> float:
        """Calculate spatial variation factor based on location"""
        
        if sensor_type == "flash_rate":
            # Lightning is more common in tropical regions
            tropical_factor = 1.0 + 0.5 * np.exp(-((lat / 30.0) ** 2))
            return tropical_factor
        
        elif sensor_type == "temperature":
            # Temperature varies with latitude
            temp_factor = 1.0 - 0.3 * np.abs(lat) / 90.0
            return temp_factor
        
        elif sensor_type == "magnetic_field":
            # Magnetic field varies with latitude (stronger at poles)
            mag_factor = 1.0 + 0.3 * np.abs(lat) / 90.0
            return mag_factor
        
        else:
            # Default: slight random spatial variation
            spatial_hash = hash(f"{lat:.1f}_{lon:.1f}_{sensor_type}") % 1000
            return 1.0 + 0.1 * (spatial_hash / 1000.0 - 0.5)
    
    def generate_dataset(self, 
                        output_format: str = "csv",
                        output_dir: str = "synthetic_data") -> Dict[str, str]:
        """Generate complete synthetic aerospace dataset"""
        
        print(f" Generating synthetic aerospace data...")
        print(f" Configuration: {self.config.data_volume_scale} scale")
        print(f" Locations: {len(self.spatial_grid)}")
        print(f" Sensors: {len(self.config.sensor_types)}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate time parameters based on scale
        duration_map = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "yearly": 365
        }
        
        duration_days = duration_map.get(self.config.data_volume_scale, 30)
        start_time = datetime.fromisoformat(self.config.start_date.replace('Z', '+00:00'))
        duration = timedelta(days=duration_days)
        
        # Adjust frequency based on scale to manage data size
        frequency_map = {
            "daily": 1.0,    # 1 Hz
            "weekly": 0.5,   # 0.5 Hz
            "monthly": 0.1,  # 0.1 Hz (every 10 seconds)
            "yearly": 0.01   # 0.01 Hz (every 100 seconds)
        }
        
        frequency = frequency_map.get(self.config.data_volume_scale, 1.0)
        
        all_data = []
        total_combinations = len(self.spatial_grid) * len(self.config.sensor_types)
        
        print(f" Generating {total_combinations} time series...")
        
        for i, location in enumerate(self.spatial_grid):
            for j, sensor_type in enumerate(self.config.sensor_types):
                
                # Progress indicator
                current = i * len(self.config.sensor_types) + j + 1
                if current % 50 == 0 or current == total_combinations:
                    print(f"Progress: {current}/{total_combinations} ({100*current/total_combinations:.1f}%)")
                
                # Generate time series for this sensor-location combination
                ts_data = self.generate_time_series(
                    sensor_type=sensor_type,
                    location=location,
                    start_time=start_time,
                    duration=duration,
                    frequency_hz=frequency
                )
                
                all_data.append(ts_data)
        
        # Combine all data
        print("Combining datasets...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp for realistic time-series ordering
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Generate metadata
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "config": {
                "data_volume_scale": self.config.data_volume_scale,
                "duration_days": duration_days,
                "frequency_hz": frequency,
                "num_locations": len(self.spatial_grid),
                "sensor_types": self.config.sensor_types,
                "total_records": len(combined_df)
            },
            "statistics": {
                "start_time": combined_df['timestamp'].min().isoformat(),
                "end_time": combined_df['timestamp'].max().isoformat(),
                "total_records": len(combined_df),
                "records_per_sensor": len(combined_df) // len(self.config.sensor_types),
                "estimated_size_mb": len(combined_df) * 100 / 1024 / 1024  # Rough estimate
            }
        }
        
        # Save data
        output_files = {}
        
        if output_format.lower() in ["csv", "all"]:
            csv_file = os.path.join(output_dir, f"aerospace_synthetic_{self.config.data_volume_scale}.csv")
            combined_df.to_csv(csv_file, index=False)
            output_files["csv"] = csv_file
            print(f" Saved CSV: {csv_file}")
        
        if output_format.lower() in ["parquet", "all"]:
            parquet_file = os.path.join(output_dir, f"aerospace_synthetic_{self.config.data_volume_scale}.parquet")
            combined_df.to_parquet(parquet_file, index=False)
            output_files["parquet"] = parquet_file
            print(f" Saved Parquet: {parquet_file}")
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f"metadata_{self.config.data_volume_scale}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        output_files["metadata"] = metadata_file
        
        print(f" Dataset generation complete!")
        print(f" Total records: {len(combined_df):,}")
        print(f"️  Time range: {metadata['statistics']['start_time']} to {metadata['statistics']['end_time']}")
        print(f" Files saved in: {output_dir}")
        
        return output_files

def main():
    """Command-line interface for data generation"""
    parser = argparse.ArgumentParser(description="Generate synthetic aerospace sensor data for TSDB benchmarking")
    
    parser.add_argument("--scale", choices=["daily", "weekly", "monthly", "yearly"], 
                       default="daily", help="Data volume scale")
    parser.add_argument("--sensors", type=int, default=50, 
                       help="Number of sensor locations")
    parser.add_argument("--frequency", type=float, default=1.0, 
                       help="Base sampling frequency (Hz)")
    parser.add_argument("--output-dir", default="synthetic_data", 
                       help="Output directory")
    parser.add_argument("--format", choices=["csv", "parquet", "all"], 
                       default="csv", help="Output format")
    parser.add_argument("--anomaly-rate", type=float, default=0.05, 
                       help="Rate of anomalies in data")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AerospaceDataConfig(
        data_volume_scale=args.scale,
        num_sensors=args.sensors,
        base_frequency_hz=args.frequency,
        anomaly_rate=args.anomaly_rate
    )
    
    # Generate data
    generator = AerospaceDataGenerator(config)
    output_files = generator.generate_dataset(
        output_format=args.format,
        output_dir=args.output_dir
    )
    
    print("\n Generation Summary:")
    for file_type, file_path in output_files.items():
        print(f"  {file_type.upper()}: {file_path}")

if __name__ == "__main__":
    main()
