# Using Surftrak with a DVL

## Motivation

We run transects 100 cm above the seafloor to capture images for scientific purposes.
The range, or HAGL (height above ground level), should be as consistent as possible.
This is partially automated using the [SURFTRAK](https://github.com/clydemcqueen/ardusub_surftrak) flight mode, which adjusts target depth based on rangefinder readings.

## The Problem: Sensor Delay

As noted in the SURFTRAK [documentation](https://github.com/clydemcqueen/ardusub_surftrak#sensor-notes), sensor delays (e.g., ~300ms for the Water Linked A50) cause oscillations.
HAGL is calculated as `terrain_z - rov_z`; while `rov_z` is known at `T=now`, the range measurement reflects the state at `T-delay`.
This latency, combined with varying delays and sensor noise, limits the effectiveness of simple feedback loops or PID tuning.

## Proposed Solution

Instead of treating the DVL as a simple rangefinder, we use the 4 individual beam readings to estimate the terrain geometry below the ROV.
We can use a Terrain Extended Kalman Filter (TerrainEKF) to:
1.  **Estimate terrain slope**: Use the beam readings to calculate terrain slope (north/east).
2.  **Predict HAGL**: Use ROV velocity and the estimated slope to predict HAGL change, compensating for latency.
3.  **Reject outliers**: Use the slope estimate to predict expected beam readings and reject anomalies (e.g., obstacles, fish) using Normalized Innovation Squared (NIS).

### Beam Splitter

The [beam splitter project](https://github.com/Seattle-Aquarium/CCR_development/issues/16) modified the WaterLinked DVL BlueOS extension to publish the 4 individual beam distances to ArduSub as additional `DISTANCE_SENSOR` messages, enabling per-beam processing.

### TerrainEKF

We use a 3-state EKF: `[terrain_z, slope_n, slope_e]`.

**Prediction Step:** Projects the state forward using ROV velocity:
*   `terrain_z += (vel_n * slope_n + vel_e * slope_e - vel_d) * dt`
*   `slope_n` and `slope_e` are assumed constant (random walk).

**Update Step:** Uses beam geometry to predict measurements. NIS is used to reject individual beam readings that deviate significantly from the planar terrain model.

**Projection:** The TerrainEKF runs at `T-delay`. We can project `terrain_z` forward to `T=now`, and then compute HAGL at `T=now`.

## Log Analysis

Data collected from Elliott Bay on 2025-10-08 with beam splitter running is available in the `data` directory.

The `replay_terrain.py` script reconstructs DVL readings and runs the TerrainEKF from tlog data (`DISTANCE_SENSOR`, `ATTITUDE`, `GLOBAL_POSITION_INT`). The results are saved in two csv files:
1. `*_TEKF.csv` - EKF inputs and outputs at `T-delay`
2. `*_TPRJ.csv` - EKF outputs projected forward to `T-now`

### Usage

```bash
./replay_terrain.py --csv data/transect1.tlog
./graph_results.py data/transect1_TEKF.csv
```

### Results (TODO)

This is from `transect2_graph2.pdf`:

![transect2_graph2](images/transect2_graph2.png)

In both transects the ROV is running up-slope from south to north at a constant rate of 10 cm/s.
The slope is initially significant, but drops substantially at T=8100 in the graph.
Even with the dampened response, you can see that the ROV depth (and therefore HAGL) oscillates quite a bit in the steeper part of the transect.
Note that the TerrainEKF quickly learns the slope.

This is from `short_graph2.pdf`:

![short_graph2](images/short_graphs2.png)

In this segment, the ROV is returning to the vessel west-to-east. It is fairly close to the surface, and at T=595 one of the thrusters
forces some air into the water column, interrupting 2 of the 4 sonar beams. The readings are rejected by the EKF.