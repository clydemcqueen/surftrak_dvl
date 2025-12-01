import unittest
import numpy as np
from dvl_model import DVLModel, BeamSet


class TestDVLModel(unittest.TestCase):

    def setUp(self):
        self.dvl = DVLModel()  # Default A50

    def test_flat_terrain(self):
        """
        Scenario: The ROV is parallel to flat ground.
        Input: All beams measure exactly 10.0 meters.
        Expected: Slope X and Y should be 0.0.
        """
        # Beam 1 (RR), Beam 2 (RL), Beam 3 (FL), Beam 4 (FR)
        beam_set = BeamSet(t_capture=0.0, range=0.0, beam_rr=10.0, beam_rl=10.0, beam_fl=10.0, beam_fr=10.0)
        slope_x, slope_y = self.dvl.calc_terrain_slope(beam_set)

        self.assertAlmostEqual(slope_x, 0.0, places=5)
        self.assertAlmostEqual(slope_y, 0.0, places=5)

    def test_uphill_slope_positive_pitch(self):
        """
        Scenario: The ROV is flying flat towards a ramp (Uphill).
        Physical Reality: The ground is closer to the FRONT sensors.

        Beam 1 (Rear Right): 15m (Far)
        Beam 2 (Rear Left):  15m (Far)
        Beam 3 (Front Left): 10m (Close)
        Beam 4 (Front Right): 10m (Close)

        Intuition: We are looking at an 'Uphill' slope.
        Convention: This should be POSITIVE Pitch (Nose Up).
        """
        beam_set = BeamSet(t_capture=0.0, range=0.0, beam_rr=15.0, beam_rl=15.0, beam_fl=10.0, beam_fr=10.0)
        slope_x, slope_y = self.dvl.calc_terrain_slope(beam_set)

        # Expect Positive Slope X (Uphill)
        self.assertGreater(slope_x, 0, "Uphill terrain should yield Positive X slope")
        # Expect Zero Slope Y (No roll)
        self.assertAlmostEqual(slope_y, 0.0, places=5)

    def test_downhill_slope_negative_pitch(self):
        """
        Scenario: The ROV is flying flat over a cliff drop-off (Downhill).
        Physical Reality: The ground is farther from the FRONT sensors.

        Rear:  10m (Close)
        Front: 15m (Far)

        Intuition: We are looking at a 'Downhill' slope.
        Convention: This should be NEGATIVE Pitch (Nose Down).
        """
        beam_set = BeamSet(t_capture=0.0, range=0.0, beam_rr=10.0, beam_rl=10.0, beam_fl=15.0, beam_fr=15.0)
        slope_x, slope_y = self.dvl.calc_terrain_slope(beam_set)

        self.assertLess(slope_x, 0, "Downhill terrain should yield Negative X slope")

    def test_banked_right_positive_roll(self):
        """
        Scenario: The terrain rises to the RIGHT of the vehicle.
        Physical Reality: The ground is closer to the RIGHT sensors.

        Left Beams (2, 3):  15m (Far)
        Right Beams (1, 4): 10m (Close)

        Intuition: Terrain is 'uphill' to our right.
        Convention: Positive Slope Y.
        """
        # Beam 1 (RR-Close), Beam 2 (RL-Far), Beam 3 (FL-Far), Beam 4 (FR-Close)
        beam_set = BeamSet(t_capture=0.0, range=0.0, beam_rr=10.0, beam_rl=15.0, beam_fl=15.0, beam_fr=10.0)
        slope_x, slope_y = self.dvl.calc_terrain_slope(beam_set)

        self.assertGreater(slope_y, 0, "Terrain rising to right should yield Positive Y slope")
        self.assertAlmostEqual(slope_x, 0.0, places=5)

    def test_invalid_input_handling(self):
        """
        Scenario: Sensor returns garbage data (0 or negative).
        Expected: Function returns (None, None).
        """
        beam_set = BeamSet(t_capture=0.0, range=0.0, beam_rr=10.0, beam_rl=-5.0, beam_fl=10.0, beam_fr=10.0)
        slope_x, slope_y = self.dvl.calc_terrain_slope(beam_set)
        self.assertIsNone(slope_x)
        self.assertIsNone(slope_y)

    def test_beam_vectors(self):
        vectors = self.dvl.get_beam_vectors_body()
        self.assertEqual(len(vectors), 4)
        # Check Z component is cos(22.5) approx 0.9239
        expected_z = np.cos(np.radians(22.5))
        for v in vectors:
            self.assertAlmostEqual(v[2], expected_z, places=4)


if __name__ == "__main__":
    unittest.main()
