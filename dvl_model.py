import math
import typing
import numpy as np


class BeamSet(typing.NamedTuple):
    t_capture: float  # Capture time, this is the DISTANCE_SENSOR message timestamp minus the sensor delay
    range: float  # Calculated range
    beam_rr: float  # Beam 1: rear right (-X, +Y)
    beam_rl: float  # Beam 2: rear left (-X, -Y)
    beam_fl: float  # Beam 3: front left (+X, -Y)
    beam_fr: float  # Beam 4: front right (+X, +Y)

    @staticmethod
    def from_array(t_capture: float, ranges: typing.List[float]):
        return BeamSet(t_capture, ranges[0], ranges[1], ranges[2], ranges[3], ranges[4])

    def beams4(self) -> typing.List[float]:
        """Return just the 4 beams as a list."""
        return [self.beam_rr, self.beam_rl, self.beam_fl, self.beam_fr]


class DVLModel:
    """
    Model for a DVL with "X" beam configuration.
    Defaults are for the WaterLinked A50 DVL.
    """

    def __init__(
        self,
        beam_angle_deg: float = 22.5,
        sensor_delay: float = 0.2,
        beam_variance: float = 0.1,
        sensor_rate: float = 9.0,
    ):
        """
        Initialize the DVL model.
        """
        self.beam_angle_deg = beam_angle_deg
        self.sensor_delay = sensor_delay
        self.beam_variance = beam_variance
        self.sensor_rate = sensor_rate
        self.sensor_dt = 1.0 / sensor_rate

        # Precompute geometry factors
        theta_rad = math.radians(self.beam_angle_deg)
        self.cot_theta = 1.0 / math.tan(theta_rad)
        self.sin_theta = math.sin(theta_rad)
        self.cos_theta = math.cos(theta_rad)

        # Slope calculation scale factor
        # Slope = (diff / sum) * cot(theta) * sqrt(2)
        self.slope_scale_factor = self.cot_theta * math.sqrt(2)

    def calc_terrain_slope(self, beam_set: BeamSet) -> typing.Tuple[typing.Optional[float], typing.Optional[float]]:
        """
        Calculate the terrain slope in FRD (Forward, Right, Down) body coordinates
        using the four sonar beam ranges.

        Args:
            beam_set (BeamSet): BeamSet object containing the four beam ranges

        Returns:
            tuple: (slope_x, slope_y)
                   - slope_x: Slope in the forward direction (positive = pitching up)
                   - slope_y: Slope in the right direction (positive = banking right)
                   Returns (None, None) if the ranges are invalid.
        """
        if beam_set.beam_rr <= 0 or beam_set.beam_rl <= 0 or beam_set.beam_fl <= 0 or beam_set.beam_fr <= 0:
            print("All beam ranges must be positive values")
            return None, None

        try:
            sum_front = beam_set.beam_fl + beam_set.beam_fr
            sum_rear = beam_set.beam_rr + beam_set.beam_rl
            sum_right = beam_set.beam_rr + beam_set.beam_fr
            sum_left = beam_set.beam_rl + beam_set.beam_fl

            total_sum = sum_front + sum_rear

            slope_x = ((sum_rear - sum_front) / total_sum) * self.slope_scale_factor
            slope_y = ((sum_left - sum_right) / total_sum) * self.slope_scale_factor

            return slope_x, slope_y

        except Exception as e:
            print(f"Error during slope calculation: {e}")
            return None, None

    def get_beam_vectors_body(self) -> typing.List[np.ndarray]:
        """
        Return beam vectors in body frame.

        Beam Configuration (X-config):
        - Beam 1: Rear-Right (-X, +Y) -> Azimuth 135 deg
        - Beam 2: Rear-Left  (-X, -Y) -> Azimuth 225 deg
        - Beam 3: Front-Left (+X, -Y) -> Azimuth 315 deg
        - Beam 4: Front-Right(+X, +Y) -> Azimuth 45 deg
        """
        azimuths = [135, 225, 315, 45]
        beam_vectors_body = []

        for az in azimuths:
            az_rad = math.radians(az)
            vx = self.sin_theta * math.cos(az_rad)
            vy = self.sin_theta * math.sin(az_rad)
            vz = self.cos_theta  # Pointing down
            beam_vectors_body.append(np.array([vx, vy, vz]))

        return beam_vectors_body
