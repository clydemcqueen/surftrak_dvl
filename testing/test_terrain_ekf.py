import unittest
import numpy as np

from dvl_model import DVLModel
from terrain_ekf import TerrainEKF


class TestTerrainEKF(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh EKF before each test.
        Initial State: Depth 10m, Flat bottom (Slopes 0).
        """
        self.dvl = DVLModel()
        self.ekf = TerrainEKF(
            self.dvl,
            initial_terrain_z=10.0,
            initial_slope_n=0.0,
            initial_slope_e=0.0,
            initial_terrain_variance=100.0,  # High variance to allow easy updates
            initial_slope_variance=100.0,
            gate_threshold=9.0,
        )
        # Identity rotation matrix (ROV is flat, heading North)
        self.R_identity = np.eye(3)

    def test_initialization(self):
        """Verify initial state vector and covariance matrix structure."""
        state = self.ekf.get_state()
        self.assertEqual(state[0], 10.0)
        self.assertEqual(state[1], 0.0)
        self.assertEqual(state[2], 0.0)

        # Check Covariance Diagonal
        self.assertEqual(self.ekf.P[0, 0], 100.0)

    def test_predict_moving_uphill(self):
        """
        Kinematic Check: If we are on an UPHILL slope (Positive Slope N)
        and we move FORWARD (North), our depth should DECREASE.
        """
        # Force state to be Uphill
        # Depth=10m, SlopeN=+0.1 (10% grade uphill), SlopeE=0
        self.ekf.x = np.array([[10.0], [0.1], [0.0]])

        # Move North at 1 m/s for 10 seconds
        # Change in depth should be: - (SlopeN * vn * dt)
        # Delta = - (0.1 * 1.0 * 10.0) = -1.0m
        self.ekf.predict(vn=1.0, ve=0.0, dt=10.0)

        new_depth = float(self.ekf.x[0, 0])
        self.assertAlmostEqual(new_depth, 9.0, places=4, msg="Moving North on an Uphill slope should decrease depth.")

    def test_predict_moving_downhill(self):
        """
        Kinematic Check: If we are on a DOWNHILL slope (Negative Slope N)
        and we move FORWARD (North), our depth should INCREASE.
        """
        # Force state to be Downhill
        # Depth=10m, SlopeN=-0.1 (10% grade downhill)
        self.ekf.x = np.array([[10.0], [-0.1], [0.0]])

        # Move North at 1 m/s for 10 seconds
        # Delta = - (-0.1 * 1.0 * 10.0) = +1.0m
        self.ekf.predict(vn=1.0, ve=0.0, dt=10.0)

        new_depth = float(self.ekf.x[0, 0])
        self.assertAlmostEqual(new_depth, 11.0, places=4, msg="Moving North on a Downhill slope should increase depth.")

    def test_update_static_uphill(self):
        """
        Measurement Check: If the FRONT beams are shorter than REAR beams,
        the filter should estimate a POSITIVE Slope N (Uphill).
        """
        # Beam ranges: Rear (1,2) = 15m, Front (3,4) = 10m
        beams = [15.0, 15.0, 10.0, 10.0]
        rov_depth = 0.0

        # Run a few updates to let the filter converge
        for _ in range(5):
            self.ekf.update(beams, rov_depth, self.R_identity, self.dvl.beam_variance, False)

        state = self.ekf.get_state()
        slope_n = state[1]
        slope_e = state[2]

        # Verify Slope N is Positive (Uphill)
        self.assertGreater(slope_n, 0.1, "Front beams closer should result in Positive Slope N")

        # Verify Slope E is near 0 (Symmetric L/R)
        self.assertAlmostEqual(slope_e, 0.0, delta=0.05)

    def test_update_static_bank_right(self):
        """
        Measurement Check: If RIGHT beams are shorter than LEFT beams,
        the filter should estimate a POSITIVE Slope E (Bank Right).
        """
        # Beam ranges: Left (2,3) = 15m, Right (1,4) = 10m
        # Beams ordered: [RR, RL, FL, FR]
        # RR(1)=10, RL(2)=15, FL(3)=15, FR(4)=10
        beams = [10.0, 15.0, 15.0, 10.0]
        rov_depth = 0.0

        for _ in range(5):
            self.ekf.update(beams, rov_depth, self.R_identity, self.dvl.beam_variance, False)

        state = self.ekf.get_state()
        slope_n = state[1]
        slope_e = state[2]

        # Verify Slope E is Positive (Rising to the Right)
        self.assertGreater(slope_e, 0.1, "Right beams closer should result in Positive Slope E")

        # Verify Slope N is near 0 (Symmetric F/R)
        self.assertAlmostEqual(slope_n, 0.0, delta=0.05)

    def test_update_invalid_beams(self):
        """Verify that invalid beams (<=0) are ignored."""
        initial_state = self.ekf.get_state()

        # All beams 0
        beams = [0.0, 0.0, 0.0, 0.0]
        self.ekf.update(beams, 0.0, self.R_identity, self.dvl.beam_variance, False)

        final_state = self.ekf.get_state()

        # State should be exactly the same (no update occurred)
        np.testing.assert_array_equal(initial_state, final_state)

    def test_project_method(self):
        """
        Verify the 'project' method returns a future state
        WITHOUT modifying the actual internal state.
        """
        self.ekf.x = np.array([[10.0], [0.1], [0.0]])  # Uphill

        # Project 10 seconds into the future moving North
        # Should predict depth = 9.0
        projected_state, _ = self.ekf.project(vn=1.0, ve=0.0, sensor_delay=10.0)

        # Check Projection
        self.assertAlmostEqual(projected_state[0], 9.0)

        # Check Internal State (Should NOT have changed)
        self.assertEqual(self.ekf.x[0, 0], 10.0, "Project method should not modify internal state")


if __name__ == "__main__":
    unittest.main()
