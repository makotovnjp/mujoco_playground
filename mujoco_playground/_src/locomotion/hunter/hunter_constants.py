# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hunter constants."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "hunter" / "xmls"
HUNTER_XML = ROOT_PATH / "pdd_mjx.xml"

# Feet sites and geoms for foot standing tasks
FEET_SITES = [
    "left_foot",
    "right_foot",
]

LEFT_FEET_GEOMS = [
    "leg_l5_link",  # Left foot end effector
]
RIGHT_FEET_GEOMS = [
    "leg_r5_link",  # Right foot end effector
]

# Robot body names based on pdd.xml
ROOT_BODY = "base_link"

# Sensor names from pdd.xml
GRAVITY_SENSOR = "orientation"  # framequat sensor
GLOBAL_LINVEL_SENSOR = "linear-velocity"  # velocimeter sensor
GLOBAL_ANGVEL_SENSOR = "angular-velocity"  # gyro sensor
LOCAL_LINVEL_SENSOR = "linear-velocity"  # velocimeter sensor
ACCELEROMETER_SENSOR = "linear-acceleration"  # accelerometer sensor
GYRO_SENSOR = "angular-velocity"  # gyro sensor
