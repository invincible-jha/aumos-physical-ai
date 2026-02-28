"""URDF/SDF robot description format parser.

GAP-354: URDF/SDF Robot Model Support.
Parses URDF (Universal Robot Description Format) and SDF (Simulation Description Format)
XML files into RobotModel Pydantic objects. Generates MuJoCo XML for GPU physics simulation.

URDF is the standard format published by robot manufacturers for ROS-compatible models.
Every production robot (UR5, Franka Panda, Boston Dynamics Spot) has a URDF model.

lxml license: BSD 3-Clause.
"""
from __future__ import annotations

from lxml import etree
from pydantic import BaseModel


class RobotModel(BaseModel):
    """Parsed robot kinematic model.

    Attributes:
        name: Robot name from the URDF/SDF root element.
        joints: List of joint dicts with name, joint_type, parent_link, child_link,
            lower_limit, upper_limit, velocity_limit, effort_limit.
        links: List of link dicts with name and mass (kg).
        source_format: "urdf" or "sdf".
        dof: Degrees of freedom — count of revolute, prismatic, and continuous joints.
    """

    name: str
    joints: list[dict]
    links: list[dict]
    source_format: str  # "urdf" or "sdf"
    dof: int


class URDFParser:
    """Parser for URDF and SDF robot description files.

    Converts robot description XML into RobotModel instances and generates
    MuJoCo XML for physics simulation via MuJoCoMJXAdapter.

    Usage::

        parser = URDFParser()
        robot = parser.parse_urdf(urdf_xml_string)
        mujoco_xml = parser.to_mujoco_xml(robot)
    """

    # URDF joint types that consume degrees of freedom
    CONTROLLABLE_TYPES = frozenset({"revolute", "prismatic", "continuous"})

    def parse_urdf(self, urdf_xml: str) -> RobotModel:
        """Parse URDF XML string into a RobotModel.

        Args:
            urdf_xml: Full URDF XML string starting with <robot ...>.

        Returns:
            RobotModel with all joints and links extracted.
        """
        root = etree.fromstring(urdf_xml.encode())
        robot_name = root.get("name", "unnamed_robot")

        joints: list[dict] = []
        for joint_el in root.findall("joint"):
            joint_type = joint_el.get("type", "revolute")
            limit_el = joint_el.find("limit")
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            joints.append({
                "name": joint_el.get("name"),
                "joint_type": joint_type,
                "parent_link": parent_el.get("link", "") if parent_el is not None else "",
                "child_link": child_el.get("link", "") if child_el is not None else "",
                "lower_limit": float(limit_el.get("lower", "-3.14159")) if limit_el is not None else -3.14159,
                "upper_limit": float(limit_el.get("upper", "3.14159")) if limit_el is not None else 3.14159,
                "velocity_limit": float(limit_el.get("velocity", "1.0")) if limit_el is not None else 1.0,
                "effort_limit": float(limit_el.get("effort", "100.0")) if limit_el is not None else 100.0,
            })

        links: list[dict] = []
        for link_el in root.findall("link"):
            inertial = link_el.find("inertial")
            mass_el = inertial.find("mass") if inertial is not None else None
            links.append({
                "name": link_el.get("name"),
                "mass": float(mass_el.get("value", "1.0")) if mass_el is not None else 1.0,
            })

        controllable_count = sum(1 for j in joints if j["joint_type"] in self.CONTROLLABLE_TYPES)
        return RobotModel(
            name=robot_name,
            joints=joints,
            links=links,
            source_format="urdf",
            dof=controllable_count,
        )

    def parse_sdf(self, sdf_xml: str) -> RobotModel:
        """Parse SDF XML string into a RobotModel.

        SDF (Simulation Description Format) is used by Gazebo and Ignition.
        Extracts the first <model> element from the SDF document.

        Args:
            sdf_xml: Full SDF XML string starting with <sdf ...>.

        Returns:
            RobotModel with joints and links extracted from the first model.
        """
        root = etree.fromstring(sdf_xml.encode())
        model_el = root.find("model") or root
        robot_name = model_el.get("name", "unnamed_robot")

        joints: list[dict] = []
        for joint_el in model_el.findall("joint"):
            joint_type = joint_el.get("type", "revolute")
            axis_el = joint_el.find("axis")
            limit_el = axis_el.find("limit") if axis_el is not None else None
            joints.append({
                "name": joint_el.get("name"),
                "joint_type": joint_type,
                "parent_link": joint_el.findtext("parent", ""),
                "child_link": joint_el.findtext("child", ""),
                "lower_limit": float(limit_el.findtext("lower", "-3.14159")) if limit_el is not None else -3.14159,
                "upper_limit": float(limit_el.findtext("upper", "3.14159")) if limit_el is not None else 3.14159,
                "velocity_limit": float(limit_el.findtext("velocity", "1.0")) if limit_el is not None else 1.0,
                "effort_limit": float(limit_el.findtext("effort", "100.0")) if limit_el is not None else 100.0,
            })

        links: list[dict] = []
        for link_el in model_el.findall("link"):
            inertial = link_el.find("inertial")
            mass_val = inertial.findtext("mass", "1.0") if inertial is not None else "1.0"
            links.append({
                "name": link_el.get("name"),
                "mass": float(mass_val),
            })

        controllable_count = sum(1 for j in joints if j["joint_type"] in self.CONTROLLABLE_TYPES)
        return RobotModel(
            name=robot_name,
            joints=joints,
            links=links,
            source_format="sdf",
            dof=controllable_count,
        )

    def to_mujoco_xml(self, robot_model: RobotModel) -> str:
        """Convert RobotModel to MuJoCo XML for physics simulation.

        Generates a minimal MuJoCo XML with worldbody, joints, and actuators.
        Feed the output directly to MuJoCoMJXAdapter.simulate_parallel().

        Args:
            robot_model: Parsed robot model from parse_urdf() or parse_sdf().

        Returns:
            MuJoCo XML string suitable for mujoco.MjModel.from_xml_string().
        """
        controllable_joints = [
            j for j in robot_model.joints if j["joint_type"] in self.CONTROLLABLE_TYPES
        ]

        joint_elements = "\n".join(
            f'    <joint name="{j["name"]}" type="{self._to_mujoco_type(j["joint_type"])}" '
            f'range="{j["lower_limit"]} {j["upper_limit"]}" />'
            for j in controllable_joints
        )

        actuator_elements = "".join(
            f'<motor joint="{j["name"]}" gear="1" />'
            for j in controllable_joints
        )

        return (
            f'<mujoco model="{robot_model.name}">\n'
            f"  <worldbody>\n"
            f'    <body name="base_link">\n'
            f"{joint_elements}\n"
            f"    </body>\n"
            f"  </worldbody>\n"
            f"  <actuator>{actuator_elements}</actuator>\n"
            f"</mujoco>"
        )

    @staticmethod
    def _to_mujoco_type(urdf_type: str) -> str:
        """Map URDF joint type to MuJoCo joint type.

        Args:
            urdf_type: URDF joint type string.

        Returns:
            MuJoCo joint type string.
        """
        mapping = {
            "revolute": "hinge",
            "prismatic": "slide",
            "continuous": "hinge",
            "fixed": "fixed",
        }
        return mapping.get(urdf_type, "hinge")
