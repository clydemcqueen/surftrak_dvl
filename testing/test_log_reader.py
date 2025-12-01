#!/usr/bin/env python3

"""
Unit tests for LogReader class from parse_dvl_tlog.py

python -m unittest testing/test_log_reader.py
"""

import unittest
import os
import sys
from io import StringIO

from replay_terrain import LogReader


class TestLogReader(unittest.TestCase):
    """Test cases for LogReader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_tlog_path = os.path.join("testing", "small.tlog")

    def test_small_tlog_exists(self):
        """Test that the small.tlog file exists"""
        self.assertTrue(os.path.exists(self.test_tlog_path), f"Test file {self.test_tlog_path} not found")

    def test_small_tlog_statistics(self):
        """
        Test that parsing testing/small.tlog produces expected statistics:
        - DVL DISTANCE_SENSOR messages found: 2655
        - DVL DISTANCE_SENSOR messages dropped: 0 (0.00%)
        - DVL sets reconstructed: 531
        """
        # Capture stdout to check the printed statistics
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Create a LogReader instance and parse the log
            log_reader = LogReader(self.test_tlog_path, verbose=False)
            log_reader.parse_tlog()

            # Get the output
            output = captured_output.getvalue()

            # Check for expected statistics in the output
            self.assertIn(
                "DVL DISTANCE_SENSOR messages found:       2655", output, "Expected 2655 DISTANCE_SENSOR messages"
            )
            self.assertIn("DVL DISTANCE_SENSOR messages dropped:     0 (0.00%)", output, "Expected 0 dropped messages")
            self.assertIn(
                "DVL sets reconstructed:                   531", output, "Expected 531 reconstructed DVL sets"
            )

        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

    def test_small_tlog_data_collection(self):
        """Test that LogReader collects data from small.tlog"""
        log_reader = LogReader(self.test_tlog_path, verbose=False)
        log_reader.parse_tlog()
        log_reader.create_readings()
        log_reader.run_ekf_forward()
        log_reader.run_smoother_backward()
        log_reader.calculate_nees()
        log_reader.create_result_records()

        # Expect to drop 2 sets because we can't extrapolate
        self.assertEqual(len(log_reader.filter_results), 529, "Should have 529 filter_states")

    def test_readings_timestamps_alignment(self):
        """
        Test that the timestamps for beams, gpi, and att in each reading are the same.
        """
        log_reader = LogReader(self.test_tlog_path, verbose=False)
        log_reader.parse_tlog()
        log_reader.create_readings()

        self.assertGreater(len(log_reader.readings), 0, "No readings were created.")

        for i, reading in enumerate(log_reader.readings):
            beam_ts = reading.beams.t_capture
            gpi_ts = reading.gpi.t_now
            att_ts = reading.att.t_now

            # Check if all three timestamps are almost equal
            self.assertAlmostEqual(
                beam_ts, gpi_ts, delta=0.001, msg=f"Beam and GPI timestamps differ significantly in reading {i}"
            )
            self.assertAlmostEqual(
                beam_ts, att_ts, delta=0.001, msg=f"Beam and ATT timestamps differ significantly in reading {i}"
            )


if __name__ == "__main__":
    unittest.main()
