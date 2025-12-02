#!/usr/bin/env python3

"""
Read a tlog file, re-create the DVL reading (4 beams plus the estimated range directly below the ROV), and estimate the
slope of the terrain.

The WL DVL generates a rangefinder reading by averaging the 4 beams and adjusting for the beam angle:
        range_down = (beam1 + beam2 + beam3 + beam4) / 4.0 * math.cos(22.5)

The range_down value is sent as a DISTANCE_SENSOR message with id=0, followed by the 4 beam values with id=1,2,3,4.

Everything is in meters and NED convention unless noted.
"""

import argparse
import bisect
import csv
import math
import os
import sys
import typing
import time

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from dvl_model import DVLModel, BeamSet
import sub_state
import terrain_ekf


MIN_VERTICAL_PROJECTION = math.cos(math.radians(45))


class Reading(typing.NamedTuple):
    """A DVL reading and the corresponding ROV information at the same timestamp."""

    beams: BeamSet
    gpi: sub_state.GlobalPosInt
    att: sub_state.Attitude
    terrain_from_range: float


class FilterState(typing.NamedTuple):
    """EKF prior and posterior states."""

    t_capture: float
    x_prior: np.ndarray
    P_prior: np.ndarray
    F: np.ndarray
    x_post: np.ndarray
    P_post: np.ndarray


class SmoothedState(typing.NamedTuple):
    """Smoothed EKF state, this is our best guess at ground truth."""

    timestamp: float  # t_capture if part of self.smoothed_states, t_now if part of self.ground_truths
    x: np.ndarray
    P: np.ndarray


class Projection(typing.NamedTuple):
    """Projected EKF state."""

    t_now: float
    x: np.ndarray
    nees_metric: float


class FilterResults(typing.NamedTuple):
    """EKF inputs and outputs we want to graph."""

    timestamp: float

    # ROV state (interpolated)
    rov_up: float

    # Range calculated by the DVL (black box)
    range: float

    # The raw beam values
    beam_rr: float
    beam_rl: float
    beam_fl: float
    beam_fr: float

    # Calculated from the ROV state and the beam values (unfiltered)
    terr_up: float  # Terrain position
    sx: float  # Slope forward
    sy: float  # Slope right
    sn: float  # Slope north
    se: float  # Slope east

    # EKF outputs
    ekf_terr_up: float  # Estimated terrain position
    ekf_sn: float  # Estimated slope north
    ekf_se: float  # Estimated slope east


class ProjectionResults(typing.NamedTuple):
    """Projection results we want to graph."""

    timestamp: float

    # So-called ground truth
    gt_terr_up: float  # Projected terrain position
    gt_sn: float  # Projected slope north
    gt_se: float  # Projected slope east

    # Projection outputs
    proj_terr_up: float  # Projected terrain position
    proj_sn: float  # Projected slope north
    proj_se: float  # Projected slope east

    # Projection error
    nees_metric: float


class LogReader:

    def __init__(
        self,
        tlog_file: str,
        verbose: bool,
        terrain_process_noise: float = 0.01,
        slope_process_noise: float = 0.01,
        gate_threshold: float = 9.0,
        start_time: float | None = None,
        stop_time: float | None = None,
    ):
        self.tlog_file = tlog_file
        self.verbose = verbose
        self.gate_threshold = gate_threshold
        self.start_time = start_time
        self.stop_time = stop_time
        self.dvl = DVLModel()

        # MAVLink messages that contain ROV state
        self.gpi_msgs: list[sub_state.GlobalPosInt] = []
        self.att_msgs: list[sub_state.Attitude] = []

        # Sonar measurements
        self.beam_sets: list[BeamSet] = []

        # Readings: sonar measurements with the time-aligned ROV state
        # readings[i].timestamp == beam_sets[i].beams.timestamp
        # We drop some beam_sets, so len(readings) < len(beam_sets)
        self.readings: list[Reading] = []

        initial_terrain_z = 100.0  # Avoid rejection: est_terrain_z - rov_depth < 0.1
        initial_slope_n = 0.0
        initial_slope_e = 0.0
        initial_terrain_variance = 100.0  # Avoid NIS rejection
        initial_slope_variance = 1.0

        if verbose:
            print(f"TerrainEKF initial_terrain_z:        {initial_terrain_z}")
            print(f"TerrainEKF initial_slope_n:          {initial_slope_n}")
            print(f"TerrainEKF initial_slope_e:          {initial_slope_e}")
            print(f"TerrainEKF initial_terrain_variance: {initial_terrain_variance}")
            print(f"TerrainEKF initial_slope_variance:   {initial_slope_variance}")
            print(f"TerrainEKF terrain_process_noise:    {terrain_process_noise}")
            print(f"TerrainEKF slope_process_noise:      {slope_process_noise}")
            print(f"TerrainEKF gate_threshold:           {gate_threshold}")

        # 3-state terrain EKF
        self.ekf = terrain_ekf.TerrainEKF(
            self.dvl,
            initial_terrain_z,
            initial_slope_n,
            initial_slope_e,
            initial_terrain_variance,
            initial_slope_variance,
            terrain_process_noise,
            slope_process_noise,
            gate_threshold,
        )

        # Gather EKF states for smoothing
        # filter_states[i].timestamp == readings[i].timestamp
        self.filter_states: list[FilterState] = []

        # Smoothed EKF states, this is our best guess at ground truth
        # smoothed_states[i].timestamp == readings[i].timestamp
        self.smoothed_states: list[SmoothedState] = []
        self.smoothed_idx: int = 0  # Lookup optimizer

        # EKF states projected forward to T=now
        # projections[i].timestamp == readings[i].timestamp + sensor_delay
        # We drop some readings, so len(projections) < len(readings)
        self.projections: list[Projection] = []

        # Smoothed EKF states, interpolated at T=now, "ground truth" for NEES metric
        # smoothed_proj_aligned[i].timestamp == projections[i].timestamp
        self.ground_truths: list[SmoothedState] = []

        # NEES metric average for this run
        self.nees_average = None

        # MSE results
        self.mse_surftrak1 = None
        self.mse_surftrak2 = None

        # Stuff to graph 1
        # filter_results[i].timestamp == readings[i].timestamp
        self.filter_results: list[FilterResults] = []

        # Stuff to graph 2
        # projection_results[i].timestamp == projections[i].timestamp
        self.projection_results: list[ProjectionResults] = []

    def parse_tlog(self):
        """
        Parse a tlog file for DISTANCE_SENSOR messages and reconstruct the DVL reading.
        We take advantage of the fact that the messages are published in id order: [0,1,2,3,4]
        Get ROV state at T=now from ATTITUDE and GLOBAL_POSITION_INT messages.
        """
        print(f"Parse {self.tlog_file}")
        mlog = mavutil.mavlink_connection(self.tlog_file)

        ds_msg_count = 0
        ds_dropped = 0
        beams: list[float] = []

        # The first DISTANCE_SENSOR in the sequence sets t_capture (t_now - sensor_delay)
        t_capture = 0.0

        while True:
            msg = mlog.recv_match(type=["DISTANCE_SENSOR", "ATTITUDE", "GLOBAL_POSITION_INT"], blocking=False)
            if msg is None:
                break

            msg_timestamp = getattr(msg, "_timestamp", 0.0)
            msg_type = msg.get_type()

            if self.start_time is not None and msg_timestamp < self.start_time:
                continue

            if self.stop_time is not None and msg_timestamp > self.stop_time:
                break

            if msg_type == "ATTITUDE":
                # The ROV state is the EK3 output projected to T=now. Assume no message delay.
                self.att_msgs.append(sub_state.Attitude(msg_timestamp, msg.roll, msg.pitch, msg.yaw))

            elif msg_type == "GLOBAL_POSITION_INT":
                # alt: mm to meters, up -> down
                # vn, ve: cm/s to m/s
                self.gpi_msgs.append(
                    sub_state.GlobalPosInt(msg_timestamp, -msg.relative_alt / 1000.0, msg.vx / 100.0, msg.vy / 100.0)
                )

            elif msg_type == "DISTANCE_SENSOR":
                # Ignore messages from ArduSub, these are duplicates
                if msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                    continue

                ds_msg_count += 1
                sensor_id = msg.id
                distance_m = msg.current_distance / 100.0

                # Enforce strict ordering 0 -> 1 -> 2 -> 3 -> 4
                if sensor_id == 0:
                    # Start of a new sequence
                    beams = [distance_m]
                    t_capture = msg_timestamp - self.dvl.sensor_delay

                elif len(beams) == sensor_id:
                    # This is the next expected id in the sequence
                    beams.append(distance_m)

                    # Is the sequence complete?
                    if len(beams) == 5:
                        self.beam_sets.append(BeamSet.from_array(t_capture, beams))
                        beams = []  # Reset for the next sequence

                else:
                    # Out of order, corrupted or missing message. Drop the current sequence and wait for id=0
                    drop = len(beams) + 1
                    if self.verbose:
                        print(f"Expected beam id {len(beams)} but found {sensor_id} at {msg_timestamp:.3f}")
                    ds_dropped += drop
                    beams = []

        print(f"DVL DISTANCE_SENSOR messages found:       {ds_msg_count}")
        print(f"DVL DISTANCE_SENSOR messages dropped:     {ds_dropped} ({ds_dropped / ds_msg_count * 100:.2f}%)")
        print(f"DVL sets reconstructed:                   {len(self.beam_sets)}")

    def create_readings(self):
        """
        Create a list of readings with beam data and ROV state at the same timestamp.
        """
        if self.verbose:
            print("Interpolate ROV state")

        for beam_set in self.beam_sets:
            t_capture = beam_set.t_capture

            # Interpolate the ROV state at t_capture
            gpi = sub_state.lookup(self.gpi_msgs, t_capture)
            att = sub_state.lookup(self.att_msgs, t_capture)

            if gpi is None or att is None:
                if self.verbose:
                    print(f"ROV state not available at {t_capture:.3f}")
                continue

            # Limit roll and pitch to 45 degrees
            vertical_projection = math.cos(att.roll) * math.cos(att.pitch)
            if vertical_projection < MIN_VERTICAL_PROJECTION:
                if self.verbose:
                    print(
                        f"High tilt (roll={math.degrees(att.roll):.3f}, "
                        f"pitch={math.degrees(att.pitch):.3f}) at {t_capture:.3f}"
                    )
                continue

            # The terrain position directly below the ROV at t_capture
            terrain_from_range = gpi.alt + beam_set.range / vertical_projection

            self.readings.append(Reading(beam_set, gpi, att, terrain_from_range))

    def run_ekf_forward(self):
        if self.verbose:
            print("Run EKF forward")

        last_t = None
        for reading in self.readings:
            # Convert Euler angles to rotation matrix using ZYX convention (yaw, pitch, roll)
            r = Rotation.from_euler("zyx", [reading.att.yaw, reading.att.pitch, reading.att.roll])
            R_body_to_earth = r.as_matrix()

            # Predict the next state
            dt = self.dvl.sensor_delay if last_t is None else reading.beams.t_capture - last_t
            if dt > 1.0 and self.verbose:
                print(f"Warning: dt {dt:.3f} > 1.0")
            last_t = reading.beams.t_capture
            F = self.ekf.predict(reading.gpi.vn, reading.gpi.ve, dt)

            # Save the prior
            x_prior = self.ekf.x.copy()
            P_prior = self.ekf.P.copy()

            # Update the state using the 4 DVL beams, saving F
            # TODO keep track of outliers and report at the end
            self.ekf.update(
                reading.beams.beams4(), reading.gpi.alt, R_body_to_earth, self.dvl.beam_variance, self.verbose
            )

            # Save the posterior
            x_post = self.ekf.x.copy()
            P_post = self.ekf.P.copy()

            self.filter_states.append(FilterState(reading.beams.t_capture, x_prior, P_prior, F, x_post, P_post))

    def run_smoother_backward(self):
        if self.verbose:
            print("Run smoother backward")

        n = len(self.filter_states)
        if n == 0:
            return

        # Initialize with the last state
        last_state = self.filter_states[-1]
        x_smoothed = last_state.x_post
        P_smoothed = last_state.P_post

        self.smoothed_states = [SmoothedState(last_state.t_capture, x_smoothed, P_smoothed)]

        # Iterate backwards
        for k in range(n - 2, -1, -1):
            curr_state = self.filter_states[k]
            next_state = self.filter_states[k + 1]

            # Calculate smoother gain J (sometimes denoted as C or K_s)
            # J = P_curr_post * F^T * P_next_prior^-1
            # F in next_state corresponds to transition from curr to next
            F = next_state.F
            P_prior_inv = np.linalg.inv(next_state.P_prior)
            J = curr_state.P_post @ F.T @ P_prior_inv

            # Calculate smoothed state and covariance
            x_smoothed = curr_state.x_post + J @ (x_smoothed - next_state.x_prior)
            P_smoothed = curr_state.P_post + J @ (P_smoothed - next_state.P_prior) @ J.T

            self.smoothed_states.append(SmoothedState(curr_state.t_capture, x_smoothed, P_smoothed))

        # Reverse to get chronological order
        self.smoothed_states.reverse()

    def lookup_smoothed_state(self, timestamp):
        """
        Find the smoothed state at target_time by interpolating between the two nearest smoothed states.
        Return (x, P) or (None, None) if out of bounds.
        """
        # 1. Check bounds
        if not self.smoothed_states:
            return None, None
        if timestamp < self.smoothed_states[0].timestamp:
            return None, None
        if timestamp > self.smoothed_states[-1].timestamp:
            # Target is beyond our last known truth (end of log)
            return None, None

        # 2. Find the index where smoothed_states[i].ts <= target_time < smoothed_states[i+1].ts
        # Since timestamps are sorted, we can iterate or use bisect.
        # Given the sequential nature of the loop, simple search is fine or just binary search.
        self.smoothed_idx = (
            bisect.bisect_right(self.smoothed_states, timestamp, key=lambda x: x.timestamp, lo=self.smoothed_idx) - 1
        )

        if self.smoothed_idx < 0 or self.smoothed_idx >= len(self.smoothed_states) - 1:
            return None, None

        # 3. Interpolate
        s0 = self.smoothed_states[self.smoothed_idx]
        s1 = self.smoothed_states[self.smoothed_idx + 1]

        dt_total = s1.timestamp - s0.timestamp
        if dt_total <= 1e-6:
            return s0.x, s0.P

        alpha = (timestamp - s0.timestamp) / dt_total

        # Linear Interpolation of State
        x_interp = s0.x + alpha * (s1.x - s0.x)

        # Linear Interpolation of Covariance
        P_interp = s0.P + alpha * (s1.P - s0.P)

        return x_interp, P_interp

    def calculate_metrics(self):
        if self.verbose:
            print("Project EKF state forward to T=now and calculate metrics")

        sq_error_sum_st1 = 0.0
        sq_error_sum_st2 = 0.0
        count = 0

        for reading, filter_state in zip(self.readings, self.filter_states):
            # t_capture = t_now - sensor_delay, recover t_now
            t_now = reading.beams.t_capture + self.dvl.sensor_delay

            # Get the ROV state at t_now
            gpi_now = sub_state.lookup(self.gpi_msgs, t_now)
            if gpi_now is None:
                if self.verbose:
                    print(f"No global position message at {t_now:.3f}")
                continue

            # Calculate the average ROV velocity during the period (t_capture, t_now)
            vn_avg = (reading.gpi.vn + gpi_now.vn) / 2.0
            ve_avg = (reading.gpi.ve + gpi_now.ve) / 2.0

            # Look up the smoothed state at t_now
            x_smooth_now, P_smooth_now = self.lookup_smoothed_state(t_now)
            if x_smooth_now is None:
                if self.verbose:
                    print(f"No smoothed state at {t_now:.3f}")
                continue

            # Project TerrainEKF forward from t_capture to t_now
            x_proj, P_proj = terrain_ekf.project(
                filter_state.x_post, filter_state.P_post, self.ekf.Q_per_sec, vn_avg, ve_avg, self.dvl.sensor_delay
            )

            # Measure the squared error between each method and "ground truth"
            error_st1 = x_smooth_now[0, 0] - reading.terrain_from_range
            error_st2 = x_smooth_now[0, 0] - x_proj[0]
            sq_error_sum_st1 += error_st1**2
            sq_error_sum_st2 += error_st2**2
            count += 1

            # Calculate NEES metric
            error = x_proj - x_smooth_now.flatten()
            P_diff = P_proj - P_smooth_now
            try:
                nees_metric = error.T @ np.linalg.solve(P_diff, error)
            except np.linalg.LinAlgError:
                # This happens if P_diff is not positive-definite (smoother > filter confidence)
                # which theoretically shouldn't happen but can with numerical noise
                if self.verbose:
                    print(f"NEES calculation failed at {t_now:.3f}")
                continue

            # Save results
            self.ground_truths.append(SmoothedState(t_now, x_smooth_now, P_smooth_now))
            self.projections.append(Projection(t_now, x_proj, float(nees_metric)))

        if count > 0:
            self.mse_surftrak1 = sq_error_sum_st1 / count
            self.mse_surftrak2 = sq_error_sum_st2 / count
            print(f"MSE current:  {self.mse_surftrak1:.6f}")
            print(f"MSE proposed: {self.mse_surftrak2:.6f}")
            if self.mse_surftrak1 > 0:
                improvement = (self.mse_surftrak1 - self.mse_surftrak2) / self.mse_surftrak1 * 100
                print(f"Improvement:    {improvement:.2f}%")

        # Calculate the average NEES (remove NaNs)
        # Average ~= 3: P_proj accurately predicts the error
        # Average >> 3: filter is overconfident, increase process noise
        # Average << 3: filter is conservative, decrease process noise
        self.nees_average = np.average([p.nees_metric for p in self.projections])
        print(f"NEES average: {self.nees_average:.2f}")
        if self.verbose:
            if self.nees_average > 5.0:
                print("Filter is overconfident, increase Q_per_sec?")
            elif self.nees_average < 1.0:
                print("Filter is conservative, decrease Q_per_sec?")

    def create_result_records(self):
        """
        Copy data into named tuples for easy CSV export. We write 2 tables so that the timestamps line up in graphing
        tools. This makes it easy to compare raw, filtered, smoothed ("ground truth") and projected values.
        """
        if self.verbose:
            print("Create result records")

        # Raw sensor readings and TerrainEKF outputs at t_capture = t_now - sensor_delay
        for reading, filter_state in zip(self.readings, self.filter_states):
            filtered = filter_state.x_post.flatten()

            # Unfiltered slopes calculated from the raw DVL beams, in the body frame
            slope_x, slope_y = self.dvl.calc_terrain_slope(reading.beams)

            # Rotate the unfiltered slopes from body frame to world frame
            slope_n = slope_x * math.cos(reading.att.yaw) - slope_y * math.sin(reading.att.yaw)
            slope_e = slope_x * math.sin(reading.att.yaw) + slope_y * math.cos(reading.att.yaw)

            self.filter_results.append(
                FilterResults(
                    timestamp=reading.beams.t_capture,
                    rov_up=-reading.gpi.alt,  # Down -> up
                    range=reading.beams.range,
                    beam_rr=reading.beams.beam_rr,
                    beam_rl=reading.beams.beam_rl,
                    beam_fl=reading.beams.beam_fl,
                    beam_fr=reading.beams.beam_fr,
                    terr_up=-reading.terrain_from_range,  # Down -> up
                    sx=slope_x,
                    sy=slope_y,
                    sn=slope_n,
                    se=slope_e,
                    ekf_terr_up=float(-filtered[0]),  # Down -> up
                    ekf_sn=float(filtered[1]),
                    ekf_se=float(filtered[2]),
                )
            )

        # Ground truth and projections at t_now
        for ground_truth, projection in zip(self.ground_truths, self.projections):
            gt = ground_truth.x.flatten()
            projected = projection.x.flatten()

            self.projection_results.append(
                ProjectionResults(
                    timestamp=projection.t_now,
                    gt_terr_up=float(-gt[0]),  # Down -> up
                    gt_sn=float(gt[1]),
                    gt_se=float(gt[2]),
                    proj_terr_up=float(-projected[0]),  # Down -> up
                    proj_sn=float(projected[1]),
                    proj_se=float(projected[2]),
                    nees_metric=projection.nees_metric,
                )
            )

    def write_results(self):
        base_name = os.path.splitext(self.tlog_file)[0]
        for prefix, fields, results in zip(
            ["TEKF", "TPRJ"],
            [FilterResults._fields, ProjectionResults._fields],
            [self.filter_results, self.projection_results],
        ):
            output_file = f"{base_name}_{prefix}.csv"

            # Add a prefix to all fields (except timestamp), this makes plotjuggler a bit easier to use
            header = [f"{prefix}.{field}" if field != "timestamp" else field for field in fields]

            try:
                with open(output_file, mode="w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)

                    # NamedTuples are iterable
                    writer.writerows(results)

                print(f"Successfully wrote {len(results)} rows to '{output_file}'")

            except IOError as e:
                print(f"Error writing to file: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tlog_file", help="Path to the .tlog file")
    parser.add_argument("--csv", action="store_true", help="Write CSV output")
    parser.add_argument("--verbose", action="store_true", help="Explain message drop reasons")
    parser.add_argument("--terrain-noise", type=float, default=0.01, help="Terrain process noise (default: 0.01)")
    parser.add_argument("--slope-noise", type=float, default=0.01, help="Slope process noise (default: 0.01)")
    parser.add_argument("--gate", type=float, default=9.0, help="Innovation gate threshold (default: 9.0)")
    parser.add_argument("--start", type=float, default=None, help="Ignore data before this time (default: None)")
    parser.add_argument("--stop", type=float, default=None, help="Ignore data after this time (default: None)")
    parser.add_argument("--tune", type=int, default=0, help="Auto-tune process noise, see code for details")
    parser.add_argument("--factor", type=float, default=0.5, help="Process noise factor (default: 0.5)")
    args = parser.parse_args()

    if not os.path.exists(args.tlog_file):
        print(f"Error: {args.tlog_file} not found")
        sys.exit(1)

    if args.tune > 0:
        if args.csv:
            print("Tuning is not supported with CSV output")
            sys.exit(1)

        terrain_noise_values = [args.terrain_noise]
        slope_noise_values = [args.slope_noise]

        for i in range(args.tune):
            terrain_noise_values.append(terrain_noise_values[-1] * args.factor)
            slope_noise_values.append(slope_noise_values[-1] * args.factor)

        steps = args.tune + 1
        nees_averages = np.zeros((steps, steps))

        for i, terrain_noise in enumerate(terrain_noise_values):
            for j, slope_noise in enumerate(slope_noise_values):
                print(f"++++++++++ TERRAIN PROCESS NOISE {terrain_noise}, SLOPE PROCESS NOISE {slope_noise}")
                log_reader = LogReader(
                    args.tlog_file, args.verbose, terrain_noise, slope_noise, args.gate, args.start, args.stop
                )
                log_reader.parse_tlog()
                log_reader.create_readings()
                log_reader.run_ekf_forward()
                log_reader.run_smoother_backward()
                log_reader.calculate_metrics()
                nees_averages[i, j] = log_reader.nees_average

        print("++++++++++")
        print("++++++++++\nResults")
        print("++++++++++")
        np.set_printoptions(precision=4, suppress=True)
        print(f"Terrain noise values: {terrain_noise_values}")
        print(f"Slope noise values: {slope_noise_values}")
        print(f"Average NEES over {steps}x{steps} runs:")
        print(nees_averages)

    else:
        log_reader = LogReader(
            args.tlog_file, args.verbose, args.terrain_noise, args.slope_noise, args.gate, args.start, args.stop
        )

        if args.verbose:
            start_time = time.time()
            log_reader.parse_tlog()
            end_time = time.time()
            print(f"Time taken by parse_tlog: {end_time - start_time:.4f} seconds")

            start_time = time.time()
            log_reader.create_readings()
            end_time = time.time()
            print(f"Time taken by create_readings: {end_time - start_time:.4f} seconds")

            start_time = time.time()
            log_reader.run_ekf_forward()
            end_time = time.time()
            print(f"Time taken by run_ekf_forward: {end_time - start_time:.4f} seconds")

            start_time = time.time()
            log_reader.run_smoother_backward()
            end_time = time.time()
            print(f"Time taken by run_smoother_backward: {end_time - start_time:.4f} seconds")

            start_time = time.time()
            log_reader.calculate_metrics()
            end_time = time.time()
            print(f"Time taken by calculate_metrics: {end_time - start_time:.4f} seconds")

        else:
            log_reader.parse_tlog()
            log_reader.create_readings()
            log_reader.run_ekf_forward()
            log_reader.run_smoother_backward()
            log_reader.calculate_metrics()

        if args.csv:
            log_reader.create_result_records()
            log_reader.write_results()


if __name__ == "__main__":
    main()
