"""Robot MCP Server — 12 Layer 1 intrinsic tools (Spot SDK wrappers).

Each tool maps 1-to-1 to a real Boston Dynamics Spot SDK service call.
For the benchmark (no live robot), each tool reads from a CouchDB document
that simulates the Spot SDK response. Scenario state is injected externally
by the eval harness via init_data.py before each run.
The tools never know which scenario is active — they just read CouchDB.

CouchDB document types (all in the iot DB):
    profile:{asset_id}      — per-asset physical layout and panel state
    robot_state:{robot_id}  — per-robot battery, pose, power and stance state
    waypoints               — singleton inspection waypoint map

SDK service mapping:
    navigate_to  → bosdyn.client.graph_nav.GraphNavClient.navigate_to()
    open_panel   → bosdyn.client.robot_command.RobotCommandClient (arm manipulation)
    get_battery  → bosdyn.client.robot_state.RobotStateClient (battery_states)
    get_pose     → bosdyn.client.robot_state.RobotStateClient (kinematic_state)
    list_waypoints → bosdyn.client.graph_nav.GraphNavClient.download_graph()
    capture_image  → bosdyn.client.image.ImageClient.get_image_from_sources()
    power_on     → bosdyn.client.power.PowerClient.power_on_motors()
    power_off    → bosdyn.client.power.PowerClient.power_off_motors()
    stand        → bosdyn.client.robot_command.RobotCommandBuilder.synchro_stand_command()
    sit          → bosdyn.client.robot_command.RobotCommandBuilder.synchro_sit_command()
    dock         → bosdyn.client.docking.DockingClient.blocking_dock_robot()
    undock       → bosdyn.client.docking.DockingClient.blocking_undock()

Live-robot connection env vars (not used in benchmark mode):
    SPOT_HOSTNAME   — robot IP, e.g. 192.168.80.3
    SPOT_USERNAME   — SDK auth username (default: admin)
    SPOT_PASSWORD   — SDK auth password
    MAXIMO_URL      — IBM Maximo base URL
    MAXIMO_APIKEY   — Maximo API key
"""

import logging
import math
import os
from typing import Dict, List, Optional, Union

import couchdb3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("robot-mcp-server")

# ---------------------------------------------------------------------------
# Environment — CouchDB (benchmark) + Spot SDK (live robot, not used in bench)
# ---------------------------------------------------------------------------

COUCHDB_URL      = os.environ.get("COUCHDB_URL")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")
IOT_DBNAME       = os.environ.get("IOT_DBNAME", "iot")
ROBOT_ID         = os.environ.get("ROBOT_ID", "spot_1")

# Live-robot connection vars — read but not consumed in benchmark mode
SPOT_HOSTNAME = os.environ.get("SPOT_HOSTNAME")
SPOT_USERNAME = os.environ.get("SPOT_USERNAME", "admin")
SPOT_PASSWORD = os.environ.get("SPOT_PASSWORD")
MAXIMO_URL    = os.environ.get("MAXIMO_URL")
MAXIMO_APIKEY = os.environ.get("MAXIMO_APIKEY")

try:
    db = couchdb3.Database(
        IOT_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info("Connected to IoT CouchDB: %s", IOT_DBNAME)
except Exception as exc:
    logger.error("Failed to connect to IoT CouchDB: %s", exc)
    db = None

# ---------------------------------------------------------------------------
# Asset ID normalisation
# ---------------------------------------------------------------------------

_DISPLAY_TO_KEY: Dict[str, str] = {
    "Chiller 6":        "chiller_6",
    "Metro Pump 1":     "metro_pump_1",
    "Hydraulic Pump 1": "hydraulic_pump_1",
    "Motor 01":         "motor_01",
    "chiller_6":        "chiller_6",
    "metro_pump_1":     "metro_pump_1",
    "hydraulic_pump_1": "hydraulic_pump_1",
    "motor_01":         "motor_01",
}


def _profile_key(asset_id: str) -> str:
    return _DISPLAY_TO_KEY.get(asset_id, asset_id.lower().replace(" ", "_"))


def _get_profile(asset_id: str) -> Optional[Dict]:
    if db is None:
        return None
    key = _profile_key(asset_id)
    try:
        return db.get(f"profile:{key}")
    except Exception as exc:
        logger.error("Profile lookup failed for %s: %s", asset_id, exc)
        return None


def _get_robot_state() -> Optional[Dict]:
    if db is None:
        return None
    try:
        return db.get(f"robot_state:{ROBOT_ID}")
    except Exception as exc:
        logger.error("robot_state lookup failed for %s: %s", ROBOT_ID, exc)
        return None


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "robot",
    instructions=(
        "Spot robot inspection tools — 12 Layer 1 intrinsic tools. "
        "Typical inspection sequence: power_on → undock → stand → list_waypoints "
        "→ get_battery → navigate_to → get_pose → open_panel → capture_image "
        "→ sit → dock → power_off. "
        "Abort if get_battery returns low_battery=True. "
        "Abort if get_pose returns localization_ok=False. "
        "Abort if list_waypoints shows the target waypoint as active=False. "
        "After capture_image, pass gauge_path to the VLM agent for reading. "
        "Call dock before power_off to return the robot to the charging station."
    ),
)

# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------


class ErrorResult(BaseModel):
    error: str


class NavigateResult(BaseModel):
    asset_id: str
    success: bool
    waypoint_id: Optional[str]
    distance_m: float
    steps_taken: int
    blocked_reason: Optional[str] = None
    message: str


class OpenPanelResult(BaseModel):
    asset_id: str
    success: bool
    access_granted: bool
    stuck_reason: Optional[str] = None
    message: str


class BatteryResult(BaseModel):
    robot_id: str
    battery_charge_pct: float
    battery_low_threshold: float
    battery_estimated_runtime_s: float
    at_charge_station: bool
    low_battery: bool
    message: str


class PoseResult(BaseModel):
    robot_id: str
    x: float
    y: float
    theta: float
    frame: str
    localization_ok: bool
    pose_drift_m: float
    fault_state: Optional[str]
    message: str


class WaypointEntry(BaseModel):
    waypoint_id: str
    waypoint_name: str
    asset_id: Optional[str]
    location_description: str
    x: float
    y: float
    active: bool


class ListWaypointsResult(BaseModel):
    total: int
    active_count: int
    waypoints: List[WaypointEntry]
    message: str


class CaptureImageResult(BaseModel):
    asset_id: str
    gauge_path: Optional[str]
    gauge_range: List[float]
    gauge_description: str
    occlusion_flag: bool
    image_available: bool
    vlm_hint: str
    message: str


class PowerResult(BaseModel):
    robot_id: str
    power_state: str   # "POWERED_ON" | "POWERED_OFF"
    transition: str    # "power_on" | "power_off"
    success: bool
    message: str


class StanceResult(BaseModel):
    robot_id: str
    stance_state: str  # "STANDING" | "SITTING"
    transition: str    # "stand" | "sit"
    power_state: str
    success: bool
    message: str


class DockResult(BaseModel):
    robot_id: str
    docked: bool
    at_charge_station: bool
    dock_waypoint_id: Optional[str]
    battery_charge_pct: float
    message: str


# ---------------------------------------------------------------------------
# Tool 1: navigate_to
# ---------------------------------------------------------------------------


@mcp.tool(title="Navigate To Asset")
def navigate_to(asset_id: str) -> Union[NavigateResult, ErrorResult]:
    """Navigate the robot to an asset's physical location.

    Simulates a Spot GraphNav NavigateTo service call.
    Returns blocked if physical_location is not set in the profile.

    Call list_waypoints first to confirm the waypoint exists and is active.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    loc = profile.get("physical_location")
    if loc is None:
        return NavigateResult(
            asset_id=asset_id,
            success=False,
            waypoint_id=None,
            distance_m=0.0,
            steps_taken=0,
            blocked_reason="physical_location not set in profile",
            message=f"Navigation blocked: no floor-plan coordinates for '{asset_id}'",
        )

    x, y, z    = float(loc.get("x", 0)), float(loc.get("y", 0)), float(loc.get("z", 0))
    distance_m = round(math.sqrt(x**2 + y**2 + z**2), 2)
    steps      = max(1, int(distance_m / 0.5))
    room       = loc.get("room_id", "unknown")
    wp_id      = profile.get("waypoint_id")

    return NavigateResult(
        asset_id=asset_id,
        success=True,
        waypoint_id=wp_id,
        distance_m=distance_m,
        steps_taken=steps,
        message=f"Navigated to '{asset_id}' in {room} ({distance_m} m, {steps} steps)",
    )


# ---------------------------------------------------------------------------
# Tool 2: open_panel
# ---------------------------------------------------------------------------


@mcp.tool(title="Open Inspection Panel")
def open_panel(asset_id: str) -> Union[OpenPanelResult, ErrorResult]:
    """Open the asset's physical inspection panel using the Spot Arm.

    Simulates a Spot robot-command manipulation request.
    panel_stuck=True in the profile means the panel cannot be opened (FM-1).
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    panel_stuck = bool(profile.get("panel_stuck", False))

    if panel_stuck:
        return OpenPanelResult(
            asset_id=asset_id,
            success=False,
            access_granted=False,
            stuck_reason="panel_stuck=True in profile (FM-1)",
            message=(
                f"Panel failed to open for '{asset_id}'. "
                "Escalate and raise work order — do not retry more than 3 times."
            ),
        )

    return OpenPanelResult(
        asset_id=asset_id,
        success=True,
        access_granted=True,
        message=f"Panel opened successfully for '{asset_id}' — access granted",
    )


# ---------------------------------------------------------------------------
# Tool 3: get_battery
# ---------------------------------------------------------------------------


@mcp.tool(title="Get Robot Battery State")
def get_battery() -> Union[BatteryResult, ErrorResult]:
    """Return the robot's battery charge and estimated runtime.

    Simulates a Spot robot-state battery_states query.
    Reads robot_state:{ROBOT_ID} from CouchDB (seeded by seed_robot_profiles.py).

    when low_battery=True (charge_pct < threshold), abort the mission
    and dock immediately. Do not proceed to navigate_to or open_panel.

    Call at mission start and after long navigate_to calls.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    charge    = float(state.get("battery_charge_pct", 100.0))
    threshold = float(state.get("battery_low_threshold", 20.0))
    runtime_s = float(state.get("battery_estimated_runtime_s", 0.0))
    at_dock   = bool(state.get("at_charge_station", False))
    low       = charge < threshold

    if low:
        msg = (
            f"LOW BATTERY: {charge:.1f}% (threshold {threshold:.0f}%). "
            f"~{runtime_s:.0f}s remaining. Abort and dock immediately."
        )
    else:
        msg = (
            f"Battery OK: {charge:.1f}%, ~{round(runtime_s / 60)} min remaining "
            f"({'docked' if at_dock else 'on mission'})"
        )

    return BatteryResult(
        robot_id=ROBOT_ID,
        battery_charge_pct=charge,
        battery_low_threshold=threshold,
        battery_estimated_runtime_s=runtime_s,
        at_charge_station=at_dock,
        low_battery=low,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 4: get_pose
# ---------------------------------------------------------------------------


@mcp.tool(title="Get Robot Pose")
def get_pose() -> Union[PoseResult, ErrorResult]:
    """Return the robot's current pose and localisation status.

    Simulates a Spot robot-state kinematic_state query.
    Reads robot_state:{ROBOT_ID} from CouchDB.

    when localization_ok=False (pose_drift_m > 0.5 m), abort the
    inspection and raise a work order for robot remapping. Readings taken
    at an unverified pose come from the wrong asset location.

    Call after navigate_to to confirm the robot arrived correctly.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    pose   = state.get("pose", {"x": 0.0, "y": 0.0, "theta": 0.0, "frame": "map"})
    loc_ok = bool(state.get("localization_ok", True))
    drift  = float(state.get("pose_drift_m", 0.0))
    fault  = state.get("fault_state")

    if not loc_ok:
        msg = (
            f"LOCALIZATION FAILURE: drift={drift:.2f} m. "
            "Cannot confirm position in facility map. "
            "Abort and raise work order for robot remapping."
        )
    elif drift > 0.5:
        msg = (
            f"WARNING: pose drift {drift:.2f} m (>0.5 m). "
            "Consider re-localising before opening panel."
        )
    else:
        msg = (
            f"Pose OK: x={float(pose.get('x', 0)):.2f} m, "
            f"y={float(pose.get('y', 0)):.2f} m, "
            f"θ={float(pose.get('theta', 0)):.3f} rad "
            f"(frame={pose.get('frame', 'map')})"
        )

    return PoseResult(
        robot_id=ROBOT_ID,
        x=float(pose.get("x", 0.0)),
        y=float(pose.get("y", 0.0)),
        theta=float(pose.get("theta", 0.0)),
        frame=str(pose.get("frame", "map")),
        localization_ok=loc_ok,
        pose_drift_m=drift,
        fault_state=fault,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 5: list_waypoints
# ---------------------------------------------------------------------------


@mcp.tool(title="List Inspection Waypoints")
def list_waypoints(
    asset_id: Optional[str] = None,
) -> Union[ListWaypointsResult, ErrorResult]:
    """List all known GraphNav waypoints from the facility map.

    Simulates a Spot graph-nav download_graph response.
    Reads the singleton 'waypoints' document from CouchDB.

    when the target asset's waypoint is missing or active=False
    (deleted after plant reconfiguration), raise a work order for waypoint
    remapping. Do NOT call navigate_to for an inactive waypoint.

    Pass asset_id to filter to that asset's waypoints. Omit to list all.
    Call before the first navigate_to in a session.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")

    try:
        wps_doc = db.get("waypoints")
    except Exception as exc:
        logger.error("Waypoints doc lookup failed: %s", exc)
        wps_doc = None

    if wps_doc is None:
        return ErrorResult(
            error="'waypoints' document not found — run seed_robot_profiles.py to initialise"
        )

    raw = wps_doc.get("waypoints", [])
    if asset_id is not None:
        key = _profile_key(asset_id)
        raw = [w for w in raw if w.get("asset_id") in (key, asset_id)]

    entries = [
        WaypointEntry(
            waypoint_id=w.get("waypoint_id", ""),
            waypoint_name=w.get("waypoint_name", ""),
            asset_id=w.get("asset_id"),
            location_description=w.get("location_description", ""),
            x=float(w.get("x", 0.0)),
            y=float(w.get("y", 0.0)),
            active=bool(w.get("active", True)),
        )
        for w in raw
    ]

    active_count = sum(1 for e in entries if e.active)
    inactive     = [e.waypoint_id for e in entries if not e.active]

    if inactive:
        msg = (
            f"{len(entries)} waypoint(s), {active_count} active. "
            f"INACTIVE (FM-11): {inactive}. "
            "Raise work order for waypoint remapping before navigating."
        )
    else:
        msg = f"{len(entries)} waypoint(s), all active."

    return ListWaypointsResult(
        total=len(entries),
        active_count=active_count,
        waypoints=entries,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 6: capture_image
# ---------------------------------------------------------------------------


@mcp.tool(title="Capture Asset Gauge Image")
def capture_image(asset_id: str) -> Union[CaptureImageResult, ErrorResult]:
    """Capture an image of the asset's gauge panel.

    Simulates a Spot image-service get_image_from_sources call.
    Returns gauge_path from the CouchDB profile (None until field data collected).

    image_available=False when gauge_path is None or the panel is occluded.
    When image_available=True, pass gauge_path and vlm_hint to the VLM
    perception agent for gauge reading. This tool does NOT run the VLM.

    occlusion_flag=True when panel_stuck=True in the profile (panel blocks view).
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    profile = _get_profile(asset_id)
    if profile is None:
        return ErrorResult(error=f"No robot profile found for asset '{asset_id}'")

    gauge_path  = profile.get("gauge_path")
    gauge_range = profile.get("gauge_range", [0, 100])
    gauge_desc  = profile.get("gauge_description", "")
    panel_stuck = bool(profile.get("panel_stuck", False))

    occlusion       = panel_stuck
    image_available = gauge_path is not None and not occlusion

    vlm_hint = (
        f"Gauge range: {gauge_range[0]}–{gauge_range[1]}. "
        f"Description: {gauge_desc}. "
        "Read the needle position and return a float value. "
        "If the image is occluded or unreadable return null."
    )

    if occlusion:
        msg = (
            f"Image capture for '{asset_id}' BLOCKED — panel is stuck/occluded. "
            "Cannot capture gauge image. Escalate FM-2."
        )
    elif not image_available:
        msg = (
            f"Image for '{asset_id}': gauge_path not yet set in profile "
            "(field data pending). VLM path unavailable."
        )
    else:
        msg = f"Image ready for '{asset_id}': {gauge_path}. Pass to VLM agent with vlm_hint."

    return CaptureImageResult(
        asset_id=asset_id,
        gauge_path=gauge_path,
        gauge_range=gauge_range,
        gauge_description=gauge_desc,
        occlusion_flag=occlusion,
        image_available=image_available,
        vlm_hint=vlm_hint,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 7: power_on
# ---------------------------------------------------------------------------


@mcp.tool(title="Power On Robot Motors")
def power_on() -> Union[PowerResult, ErrorResult]:
    """Power on the Spot robot's drive motors.

    Simulates a Spot PowerClient.power_on_motors() call.
    Reads power_state from robot_state:{ROBOT_ID} in CouchDB.

    Call at mission start before stand or navigate_to. Safe to call
    when already powered on — returns success with the current state.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    current = state.get("power_state", "POWERED_ON")
    if current == "POWERED_ON":
        msg = f"Motors already powered on (robot '{ROBOT_ID}')"
    else:
        msg = f"Motors powered on successfully (robot '{ROBOT_ID}')"

    return PowerResult(
        robot_id=ROBOT_ID,
        power_state="POWERED_ON",
        transition="power_on",
        success=True,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 8: power_off
# ---------------------------------------------------------------------------


@mcp.tool(title="Power Off Robot Motors")
def power_off() -> Union[PowerResult, ErrorResult]:
    """Power off the Spot robot's drive motors.

    Simulates a Spot PowerClient.power_off_motors() call.
    The robot must be sitting before powering off — call sit() first.

    After power_off the robot cannot move until power_on() is called again.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    current = state.get("power_state", "POWERED_ON")
    stance  = state.get("stance_state", "STANDING")

    if stance == "STANDING":
        return PowerResult(
            robot_id=ROBOT_ID,
            power_state=current,
            transition="power_off",
            success=False,
            message=(
                f"Cannot power off: robot '{ROBOT_ID}' is STANDING. "
                "Call sit() before power_off()."
            ),
        )

    if current == "POWERED_OFF":
        msg = f"Motors already powered off (robot '{ROBOT_ID}')"
    else:
        msg = f"Motors powered off successfully (robot '{ROBOT_ID}')"

    return PowerResult(
        robot_id=ROBOT_ID,
        power_state="POWERED_OFF",
        transition="power_off",
        success=True,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 9: stand
# ---------------------------------------------------------------------------


@mcp.tool(title="Command Robot to Stand")
def stand() -> Union[StanceResult, ErrorResult]:
    """Command the Spot robot to stand upright.

    Simulates a Spot RobotCommandBuilder.synchro_stand_command() call via
    RobotCommandClient.

    Robot must be powered on before standing. Call power_on() first if needed.
    Required before navigate_to or open_panel.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    power = state.get("power_state", "POWERED_ON")
    if power != "POWERED_ON":
        return StanceResult(
            robot_id=ROBOT_ID,
            stance_state=state.get("stance_state", "SITTING"),
            transition="stand",
            power_state=power,
            success=False,
            message=f"Cannot stand: robot '{ROBOT_ID}' is not powered on. Call power_on() first.",
        )

    current_stance = state.get("stance_state", "SITTING")
    if current_stance == "STANDING":
        msg = f"Robot '{ROBOT_ID}' is already standing"
    else:
        msg = f"Robot '{ROBOT_ID}' stood up successfully"

    return StanceResult(
        robot_id=ROBOT_ID,
        stance_state="STANDING",
        transition="stand",
        power_state=power,
        success=True,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 10: sit
# ---------------------------------------------------------------------------


@mcp.tool(title="Command Robot to Sit")
def sit() -> Union[StanceResult, ErrorResult]:
    """Command the Spot robot to sit down.

    Simulates a Spot RobotCommandBuilder.synchro_sit_command() call via
    RobotCommandClient.

    Call before dock() at the end of a mission, or in an emergency battery abort.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    power = state.get("power_state", "POWERED_ON")
    if power != "POWERED_ON":
        return StanceResult(
            robot_id=ROBOT_ID,
            stance_state=state.get("stance_state", "SITTING"),
            transition="sit",
            power_state=power,
            success=False,
            message=f"Cannot sit: robot '{ROBOT_ID}' is not powered on.",
        )

    current_stance = state.get("stance_state", "STANDING")
    if current_stance == "SITTING":
        msg = f"Robot '{ROBOT_ID}' is already sitting"
    else:
        msg = f"Robot '{ROBOT_ID}' sat down successfully"

    return StanceResult(
        robot_id=ROBOT_ID,
        stance_state="SITTING",
        transition="sit",
        power_state=power,
        success=True,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Tool 11: dock
# ---------------------------------------------------------------------------


@mcp.tool(title="Dock Robot at Charging Station")
def dock() -> Union[DockResult, ErrorResult]:
    """Return the robot to its charging dock.

    Simulates a Spot DockingClient.blocking_dock_robot() call.
    Navigates to the dock waypoint (asset_id=None) in the waypoints document.

    Call at mission end or when get_battery() returns low_battery=True.
    The dock waypoint must be active in the waypoints document.
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    try:
        wps_doc = db.get("waypoints")
    except Exception as exc:
        logger.error("Waypoints doc lookup failed: %s", exc)
        wps_doc = None

    dock_wp = None
    if wps_doc:
        for wp in wps_doc.get("waypoints", []):
            if wp.get("asset_id") is None and "dock" in wp.get("waypoint_id", "").lower():
                dock_wp = wp
                break

    if dock_wp is None or not dock_wp.get("active", True):
        return DockResult(
            robot_id=ROBOT_ID,
            docked=False,
            at_charge_station=False,
            dock_waypoint_id=dock_wp.get("waypoint_id") if dock_wp else None,
            battery_charge_pct=float(state.get("battery_charge_pct", 0.0)),
            message=f"Dock waypoint not found or inactive — cannot dock robot '{ROBOT_ID}'",
        )

    charge = float(state.get("battery_charge_pct", 0.0))
    return DockResult(
        robot_id=ROBOT_ID,
        docked=True,
        at_charge_station=True,
        dock_waypoint_id=dock_wp.get("waypoint_id"),
        battery_charge_pct=charge,
        message=(
            f"Robot '{ROBOT_ID}' docked at '{dock_wp.get('waypoint_name', 'dock')}'. "
            f"Battery: {charge:.1f}%."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 12: undock
# ---------------------------------------------------------------------------


@mcp.tool(title="Undock Robot from Charging Station")
def undock() -> Union[DockResult, ErrorResult]:
    """Release the robot from the charging dock.

    Simulates a Spot DockingClient.blocking_undock() call.
    Checks at_charge_station and battery_charge_pct before undocking.

    Call at mission start, after power_on(). Abort if battery is too low
    to complete the mission (check get_battery() before proceeding).
    """
    if db is None:
        return ErrorResult(error="IoT database unavailable")
    state = _get_robot_state()
    if state is None:
        return ErrorResult(
            error=f"No robot_state document for robot '{ROBOT_ID}' — "
                  "run seed_robot_profiles.py to initialise"
        )

    charge    = float(state.get("battery_charge_pct", 100.0))
    threshold = float(state.get("battery_low_threshold", 20.0))

    if charge < threshold:
        return DockResult(
            robot_id=ROBOT_ID,
            docked=True,
            at_charge_station=True,
            dock_waypoint_id=None,
            battery_charge_pct=charge,
            message=(
                f"UNDOCK REFUSED: battery at {charge:.1f}% (threshold {threshold:.0f}%). "
                "Charge before undocking."
            ),
        )

    return DockResult(
        robot_id=ROBOT_ID,
        docked=False,
        at_charge_station=False,
        dock_waypoint_id=None,
        battery_charge_pct=charge,
        message=f"Robot '{ROBOT_ID}' undocked successfully. Battery: {charge:.1f}%.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
