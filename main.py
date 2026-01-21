#!/usr/bin/env python3

from datetime import datetime, timezone
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    WindowProperties,
    loadPrcFile,
    Vec3,
    LVecBase3,
    Shader,
    DirectionalLight,
    AmbientLight,
    InputDevice,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomPoints, GeomNode,
    GraphicsPipe, GraphicsOutput, Texture, FrameBufferProperties,
    ShaderAttrib, LColor, CardMaker,
    TransparencyAttrib, OrthographicLens, Camera, NodePath, LineSegs,
    TextNode,
    GamepadButton, KeyboardButton
)
from direct.task import Task
from panda3d.vrpn import VrpnClient
from panda3d.core import Quat, Camera, TrackerNode, ButtonNode, AnalogNode, Transform2SG, loadPrcFileData, PerspectiveLens, LVecBase3, Lens, ColorAttrib
from sgp4 import omm
from sgp4.propagation import gstime

from sgp4.api import Satrec, SatrecArray
from sgp4.conveniences import jday_datetime
import numpy as np
import logging
import os
import sys

# Constants
SCALE = 0.001  # 1 unit = 1000 km
EARTH_RADIUS = 6371.0
EARTH_TEXTURE_OFFSET = 160 # 148 TODO fix with better texture

# Shader
ENABLE_SHADER = True
VERT_SHADER = "shader/satellites.vert"
FRAG_SHADER = "shader/satellites.frag"

# Data source
OMM_FILE = "all.csv"
if len(sys.argv) > 1:
    OMM_FILE = sys.argv[1]
OMM_PATH = os.path.join("res", OMM_FILE)
print(f"Loading omm resource file: {OMM_FILE}")

# Logging
np.set_printoptions(precision=2)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Panda Config
if ENABLE_SHADER:
    loadPrcFile("config/Config_Shader.prc")
else:
    loadPrcFile("config/Config.prc")


class SatelliteVisualizer(ShowBase):

    IOD_METERS = 0.064
    LEDWALL_WIDTH_METERS = 4.6
    LEDWALL_HEIGHT_METERS = 2.6
    Z_CALIBRATION_METERS = 3.0
    TRANSFORMATION_BUTTON_NUMBER = 0
    TRANSFORMATION_SPEED = 2
    NEAR_PLANE = 0.1
    FAR_PLANE = 1000

    def __init__(self):
        ShowBase.__init__(self)

        self.setup_vrpn()
        self.setup_stereo()

        # COVISE-style navigation
        self.transformation_started = False
        self.reference_pos = LVecBase3(0, 0, 0)
        self.reference_quat = Quat()


        # self.disableMouse()
        self.setBackgroundColor(Vec3(0,0,0))

        self.init_gmst()
        # self.setup_camera()
        self.setup_light() # TODO fix rotation
        self.setup_earth()
        self.setup_gamepad()

        # Filtering initialization
        self.selected_global_id = -1
        self.filter_orbit = 0 # 0: All, 1: LEO, 2: MEO, 3: GEO
        self.filter_constellation = 0 # 0: All
        self.ORBIT_TYPES = ["All", "LEO", "MEO", "GEO"]
        self.CONSTELLATIONS = ["All", "STARLINK", "IRIDIUM", "GPS", "ONEWEB", "GALILEO", "BEIDOU"]
        self.visible_orbits = []
        self.visible_indices = []
        self.orbit_visual_node = None

        self.load_omm_model()
        self.setup_shader()
        # self.setup_ray()
        self.setup_ui()
        
        # self.accept("space", self.process_selection)
        # self.accept("gamepad-face_a", self.process_selection)
        self.accept("o", self.toggle_orbit_filter)
        self.accept("c", self.toggle_constellation_filter)

        # self.taskMgr.add(self.process_inputs, "input")
        self.taskMgr.add(self.render_satellites, "render")


    def setup_vrpn(self):
        # connect to vrpn server
        self.vrpnclient = VrpnClient("localhost")
        
        # register update method used to poll new data from the VRPN server via the VrpnClient
        taskMgr.add(self.poll_vrpn_server, "poll_vrpn_server")
        
        # CAVE parent rig, moves according to tracking and covise-style scene navigation
        self.rig = self.render.attachNewNode("rig")

        # VRPN trackers:
        
        # In Panda3D, input is handled via the data graph, a tree that lives separately to the scene graph.
        # Therefore, attaching a tracker node input (data graph) to a scene graph node will not work.
        # Instead we need to create a node in the data graph for the input, a node in the scene graph for the actual scene, and convert
        # from the data graph node to the scene graph node using a Transform2SG object ("Transform to Scene Graph").

        # head glasses tracker (sensor #1)
        self.head_tracker = TrackerNode(self.vrpnclient, "DTrack:2")        # in DATA graph: create a tracker node using the VRPN name (TrackerName:SensorNumber)
        self.dataRoot.node().addChild(self.head_tracker)                    # add tracker to DATA graph
        self.head = render.attachNewNode("head")                            # create a corresponding node for the SCENE graph
        d2s_head = Transform2SG("d2s_head")                                 # create a transform object to transform from DATA graph to SCENE graph (SG)
        self.head_tracker.addChild(d2s_head)                                # add the transform object as a child to the DATA graph node
        d2s_head.setNode(self.head.node())                                  # set the output node to the SCENE graph node
        self.head.reparentTo(self.rig)

        # attach left and righ eye trackers (only used for projection matrix calculation)
        self.left = self.head.attachNewNode("left")
        self.left.setPos(-SatelliteVisualizer.IOD_METERS / 2.0, 0.0, 0.0)
        self.right = self.head.attachNewNode("right")
        self.right.setPos(+SatelliteVisualizer.IOD_METERS / 2.0, 0.0, 0.0)

        # Flystick 8 tracker (sensor #8)
        self.flystick8_tracker = TrackerNode(self.vrpnclient, "DTrack:8")   # in DATA graph: create a tracker node using the VRPN name (TrackerName:SensorNumber)
        self.dataRoot.node().addChild(self.flystick8_tracker)               # add tracker to DATA graph
        self.flystick8 = render.attachNewNode("flystick8")                  # create a corresponding node for the SCENE graph
        d2s_flystick8 = Transform2SG("d2s_flystick8")                       # create a transform object to transform from DATA graph to SCENE graph (SG)
        self.flystick8_tracker.addChild(d2s_flystick8)                      # add the transform object as a child to the DATA graph node
        d2s_flystick8.setNode(self.flystick8.node())                        # set the output node to the SCENE graph node
        self.flystick8.reparentTo(self.rig)

        # Flystick 9 tracker (sensor #9)
        self.flystick9_tracker = TrackerNode(self.vrpnclient, "DTrack:9")   # in DATA graph: create a tracker node using the VRPN name (TrackerName:SensorNumber)
        self.dataRoot.node().addChild(self.flystick9_tracker)               # add tracker to DATA graph
        self.flystick9 = render.attachNewNode("flystick9")                  # create a corresponding node for the SCENE graph
        d2s_flystick9 = Transform2SG("d2s_flystick9")                       # create a transform object to transform from DATA graph to SCENE graph (SG)
        self.flystick9_tracker.addChild(d2s_flystick9)                      # add the transform object as a child to the DATA graph node
        d2s_flystick9.setNode(self.flystick9.node())                        # set the output node to the SCENE graph node
        self.flystick9.reparentTo(self.rig)

        self.flystick8_pointer = self.create_pointer(self.flystick8)
        self.flystick9_pointer = self.create_pointer(self.flystick9)
        
        # VRPN buttons:
        self.buttons = ButtonNode(self.vrpnclient, "DTrack")
        taskMgr.add(self.handle_vrpn_buttons, "handle_vrpn_buttons")
        
        # VRPN analogs:
        self.analogs = AnalogNode(self.vrpnclient, "DTrack")
        taskMgr.add(self.handle_vrpn_analogs, "handle_vrpn_analogs")


    def setup_stereo(self):

        # disable default cam
        base.camNode.setActive(False)
        base.cam.node().setActive(False)

        # make_display_region automatically returns a StereoDisplayRegion if framebuffer-stereo is set to 1 in config
        # both left and right display region share the default camera initially
        self.stereo_display_region = base.win.make_display_region()
        left_display_region = self.stereo_display_region.getLeftEye()
        right_display_region = self.stereo_display_region.getRightEye()
        
        # create and configure left and right eye camera:

        self.left_cam = self.left.attachNewNode(Camera("left_cam"))
        self.right_cam = self.right.attachNewNode(Camera("right_cam"))

        screen_z = SatelliteVisualizer.Z_CALIBRATION_METERS
        screen_left = -SatelliteVisualizer.LEDWALL_WIDTH_METERS / 2.0
        screen_right = SatelliteVisualizer.LEDWALL_WIDTH_METERS / 2.0
        screen_top = SatelliteVisualizer.LEDWALL_HEIGHT_METERS / 2.0
        screen_bottom = -SatelliteVisualizer.LEDWALL_HEIGHT_METERS / 2.0

        self.screen_ul = LVecBase3(screen_left, screen_z, screen_top)
        self.screen_ur = LVecBase3(screen_right, screen_z, screen_top)
        self.screen_ll = LVecBase3(screen_left, screen_z, screen_bottom)
        self.screen_lr = LVecBase3(screen_right, screen_z, screen_bottom)

        left_lens = PerspectiveLens()
        left_lens.setNear(SatelliteVisualizer.NEAR_PLANE)
        left_lens.setFar(SatelliteVisualizer.FAR_PLANE)

        right_lens = PerspectiveLens()
        right_lens.setNear(SatelliteVisualizer.NEAR_PLANE)
        right_lens.setFar(SatelliteVisualizer.FAR_PLANE)

        self.left_cam.node().setLens(left_lens)
        self.right_cam.node().setLens(right_lens)

        left_display_region.setCamera(self.left_cam)
        right_display_region.setCamera(self.right_cam)

        taskMgr.add(self.update_projection_matrix, "update_projection_matrix")


    def update_projection_matrix(self, t):

        ul_l = self.left.getRelativePoint(self.rig, self.screen_ul)
        ur_l = self.left.getRelativePoint(self.rig, self.screen_ur)
        ll_l = self.left.getRelativePoint(self.rig, self.screen_ll)
        lr_l = self.left.getRelativePoint(self.rig, self.screen_lr)

        ul_r = self.right.getRelativePoint(self.rig, self.screen_ul)
        ur_r = self.right.getRelativePoint(self.rig, self.screen_ur)
        ll_r = self.right.getRelativePoint(self.rig, self.screen_ll)
        lr_r = self.right.getRelativePoint(self.rig, self.screen_lr)

        flags = Lens.FC_off_axis

        self.left_cam.node().getLens().setFrustumFromCorners(
            ul_l, ur_l, ll_l, lr_l, flags
        )

        self.right_cam.node().getLens().setFrustumFromCorners(
            ul_r, ur_r, ll_r, lr_r, flags
        )

        return Task.cont


    def poll_vrpn_server(self, t):
        self.vrpnclient.poll()
        self.head.setQuat(Quat()) # discard tracked orientation (cannot face away from wall)
        return Task.cont
        
    
    # DTrack exposes a single button device for all Flystick buttons, that means:
    # Buttons 0-8 are for Flystick 9 and buttons 8-16 are for Flystick 8.
    def handle_vrpn_buttons(self, t):

        # Default COVISE-style scene navigation:

        tbtn_pressed = self.buttons.getButtonState(SatelliteVisualizer.TRANSFORMATION_BUTTON_NUMBER)

        if not self.transformation_started and tbtn_pressed:
            self.transformation_started = True

        if not tbtn_pressed:
            self.transformation_started = False
            self.reference_pos = LVecBase3(self.flystick9.getPos())
            self.reference_quat = Quat(self.flystick9.getQuat())       

        if self.transformation_started:
            dt = globalClock.getDt()
            
            moved = (self.flystick9.getPos() - self.reference_pos) * SatelliteVisualizer.TRANSFORMATION_SPEED * dt

            self.orbit_h += moved.getX()
            self.orbit_p += moved.getY()
            self.orbit_distance += moved.getZ()

            rad_h = np.radians(self.orbit_h)
            rad_p = np.radians(self.orbit_p)
            
            x = self.orbit_distance * np.sin(rad_h) * np.cos(rad_p)
            y = -self.orbit_distance * np.cos(rad_h) * np.cos(rad_p)
            z = self.orbit_distance * np.sin(rad_p)
            
            self.camera.setPos(x, y, z)
            self.camera.lookAt(0, 0, 0)

            # translation
            self.rig.setPos(self.rig, LVecBase3(x,y,z))
            self.rig.lookAt(0,0,0)

            # rotation
            # q_old_inv = Quat(self.reference_quat)
            # q_old_inv.invertInPlace()
            # q_diff = q_old_inv * self.flystick9.getQuat()
            # h, _, _ = q_diff.getHpr()
            # q_yaw_only = Quat()
            # q_yaw_only.setHpr((h * SatelliteVisualizer.TRANSFORMATION_SPEED * dt, 0, 0))
            # self.rig.setQuat(self.rig.getQuat() * q_yaw_only)

        # handle your Flystick button presses here
        # example:
        #if self.buttons.getButtonState(0): # trigger Flystick 9
            #print("Flystick ID 9 button 0 pressed")
        return Task.cont
        

    
    # DTrack exposes a single analog device for all Flystick joysticks, that means:
        # Flystick 9 x-Axis: 0
        # Flystick 9 y-Axis: 1
        # Flystick 8 x-Axis: 2
        # Flystick 8 y-Axis: 3
    def handle_vrpn_analogs(self, t):
        
        # handle your Flystick joystick inputs here
        # examples:

        x8 = self.analogs.getControlState(2)
        y8 = self.analogs.getControlState(3)
        #print(f"Flystick ID 8 joystick: {x8}|{y8}")
        
        x9 = self.analogs.getControlState(0)
        y9 = self.analogs.getControlState(1)
        #print(f"Flystick ID 9 joystick: {x9}|{y9}")
        
        return Task.cont









    def setup_ui(self):
        self.ui_text = OnscreenText(
            text="No Satellite Selected",
            parent=self.a2dTopRight,
            style=1,
            fg=(1, 1, 1, 1),
            pos=(-0.05, -0.1),
            align=TextNode.ARight,
            scale=0.06,
            shadow=(0, 0, 0, 1),
            mayChange=True
        )
        
        self.filter_text = OnscreenText(
            text="Filter: All\nConstellation: All",
            parent=self.a2dTopLeft,
            style=1,
            fg=(1, 1, 1, 1),
            pos=(0.05, -0.1),
            align=TextNode.ALeft,
            scale=0.05,
            shadow=(0, 0, 0, 1),
            mayChange=True
        )


    def update_ui(self):
        # Update filter text
        orbit_str = self.ORBIT_TYPES[self.filter_orbit]
        const_str = self.CONSTELLATIONS[self.filter_constellation]
        count_str = f"Showing: {len(self.visible_orbits)}"
        self.filter_text.setText(f"Orbit: {orbit_str}\nConstellation: {const_str}\n{count_str}")

        if self.selected_global_id == -1:
            self.ui_text.setText("No Satellite Selected")
            return

        sat = self.satellite_orbits[self.selected_global_id]
        info = self.satellite_infos[self.selected_global_id]
        
        # Calculate current state
        now = datetime.now(timezone.utc)
        jd, fr = jday_datetime(now)
        error, pos, vel = sat.sgp4(jd, fr)
        
        if error != 0:
            self.ui_text.setText(f"Error calculating state\nSGP4 Error: {error}")
            return

        # pos is in km, vel is in km/s
        pos_km = np.array(pos)
        vel_km_s = np.array(vel)
        
        altitude = np.linalg.norm(pos_km) - EARTH_RADIUS
        velocity = np.linalg.norm(vel_km_s)
        
        name = info.get('OBJECT_NAME', 'Unknown')
        
        period_h = 2 * np.pi / sat.no_kozai / 60
        
        text = (
            f"{name}\n"
            f"Alt: {altitude:.1f} km\n"
            f"Vel: {velocity:.2f} km/s\n"
            f"Period: {period_h:.1f} h\n"
            f"Incl: {np.degrees(sat.inclo):.1f}°"
        )
        self.ui_text.setText(text)


    def init_gmst(self):
        start_time = datetime.now(timezone.utc)
        jd, fr = jday_datetime(start_time)
        self.gstime = gstime(jd + fr)


    def setup_camera(self):
        self.orbit_distance = 150.0
        self.orbit_h = 0.0
        self.orbit_p = 0.0
        
        self.update_camera_pos()


    def update_camera_pos(self):
        # Convert Spherical to Cartesian
        # H (Azimuth), P (Elevation)
        # In Panda3D: 
        # X = dist * -sin(H) * cos(P)
        # Y = dist * cos(H) * cos(P)
        # Z = dist * sin(P)
        
        rad_h = np.radians(self.orbit_h)
        rad_p = np.radians(self.orbit_p)
        
        x = self.orbit_distance * np.sin(rad_h) * np.cos(rad_p)
        y = -self.orbit_distance * np.cos(rad_h) * np.cos(rad_p)
        z = self.orbit_distance * np.sin(rad_p)
        
        self.camera.setPos(x, y, z)
        self.camera.lookAt(0, 0, 0)

    

    def setup_light(self):
        time = datetime.now(timezone.utc)
        day_of_year = time.timetuple().tm_yday
        sun_declination = 23.44 * np.sin(np.radians((360/365)*(day_of_year - 81)))
        sun_angle = np.degrees(self.gstime)

        sun_light = DirectionalLight("sun_light")
        sun_light.setColor(Vec3(1, 1, 0.9))
        sun_node = self.render.attachNewNode(sun_light)
        sun_node.setHpr(-90 + sun_angle, -sun_declination, 0)
        self.render.setLight(sun_node)

        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor(Vec3(0.2, 0.2, 0.3))
        ambient_node = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_node)


    def setup_background(self):
        self.stars = self.loader.loadModel("models/solar_sky_sphere")
        self.stars_tex = self.loader.loadTexture("textures/2k_stars.jpg")
        self.stars.setTexture(self.stars_tex, 1)
        self.stars.setScale(10000)
        self.stars.reparentTo(self.render)


    def setup_earth(self):
        time = datetime.now(timezone.utc)
        hours = time.hour + time.minute/60 + time.second/3600
        earth_angle = np.fmod(hours * 15.0, 360.0)
        earth_angle = np.degrees(self.gstime)
        earth_angle = 0

        self.earth = self.loader.loadModel("models/planet_sphere")
        self.earth_tex = self.loader.loadTexture("textures/earth_1k_tex.jpg")
        self.earth.setTexture(self.earth_tex, 1)
        self.earth.setScale(EARTH_RADIUS * SCALE)
        self.earth.setHpr(EARTH_TEXTURE_OFFSET + earth_angle, 0, 0)
        self.earth.reparentTo(self.render)


    def setup_gamepad(self):
        self.gamepad = None
        self.accept("connect-device", self.connect_device)
        self.accept("disconnect-device", self.disconnect_device)
        # Check if devices are already available
        gamepads = self.devices.getDevices(InputDevice.DeviceClass.gamepad)
        if gamepads:
            self.connect_device(gamepads[0])


    def connect_device(self, device):
        if not self.gamepad and device.device_class == InputDevice.DeviceClass.gamepad:
            print(f"Gamepad connected: {device.name}")
            self.gamepad = device
            self.attachInputDevice(device, prefix="gamepad")


    def disconnect_device(self, device):
        if self.gamepad == device:
            print(f"Gamepad disconnected: {device.name}")
            self.detachInputDevice(device)
            self.gamepad = None


    def load_omm_model(self):
        self.satellite_infos: list[dict[str,str]] = []
        self.satellite_orbits: list[Satrec] = []
        with open(OMM_PATH) as f:
            parsed = omm.parse_csv(f)
            for fields in parsed:
                sat = Satrec()
                omm.initialize(sat, fields)
                self.satellite_infos.append(fields)
                self.satellite_orbits.append(sat)


        vformat = GeomVertexFormat.getV3()
        self.vertex_data = GeomVertexData("points", vformat, Geom.UHDynamic)
        self.vertex_data.setNumRows(len(self.satellite_orbits))

        points = GeomPoints(Geom.UHDynamic)
        points.addNextVertices(len(self.satellite_orbits))
        points.closePrimitive()

        geom = Geom(self.vertex_data)
        geom.addPrimitive(points)

        node = GeomNode("points")
        node.addGeom(geom)
        self.points_np = self.render.attachNewNode(node)
        
        self.apply_filters()

    
    def toggle_orbit_filter(self):
        self.filter_orbit = (self.filter_orbit + 1) % len(self.ORBIT_TYPES)
        self.apply_filters()

    def toggle_constellation_filter(self):
        self.filter_constellation = (self.filter_constellation + 1) % len(self.CONSTELLATIONS)
        self.apply_filters()

    def apply_filters(self):
        self.visible_orbits = []
        self.visible_indices = []
        
        orbit_type = self.ORBIT_TYPES[self.filter_orbit]
        constellation = self.CONSTELLATIONS[self.filter_constellation]
        
        # Pre-calc limits for orbit types
        # Mean Motion (revs/day)
        # LEO: > 11.25
        # GEO: 0.99 - 1.01 (approx 1.0)
        # MEO: 1.01 < mm < 11.25
        
        for i, info in enumerate(self.satellite_infos):
            # Check Constellation
            if constellation != "All":
                name = info.get("OBJECT_NAME", "").upper()
                if constellation not in name:
                    continue
            
            # Check Orbit
            if orbit_type != "All":
                mm = float(info.get("MEAN_MOTION", 0))
                if orbit_type == "LEO" and mm <= 11.25:
                    continue
                elif orbit_type == "MEO" and (mm >= 11.25 or mm <= 1.1):
                    continue
                elif orbit_type == "GEO" and (mm < 0.9 or mm > 1.1): # Rough GEO filter
                    continue
            
            self.visible_orbits.append(self.satellite_orbits[i])
            self.visible_indices.append(i)
            
        # Update Geometry size
        num_visible = len(self.visible_orbits)
        self.vertex_data.setNumRows(num_visible)
        
        if num_visible > 0:
            points_primitive = self.points_np.node().modifyGeom(0).modifyPrimitive(0)
            points_primitive.clearVertices()
            points_primitive.addNextVertices(num_visible)
            points_primitive.closePrimitive()
        else:
            points_primitive = self.points_np.node().modifyGeom(0).modifyPrimitive(0)
            points_primitive.clearVertices()
            points_primitive.closePrimitive()

    
    def setup_shader(self):
        if not ENABLE_SHADER:
            self.points_np.setLightOff()
            self.points_np.setRenderModeThickness(0.025)
            self.points_np.setRenderModePerspective(True)
            self.points_np.setColor(1, 1, 1, 1)
            return
        
        shader = Shader.load(Shader.SL_GLSL, VERT_SHADER, FRAG_SHADER)
        attrib = ShaderAttrib.make(shader)
        attrib = attrib.setFlag(ShaderAttrib.F_shader_point_size, True)

        self.points_np.setAttrib(attrib)
        self.points_np.setShader(shader)
        self.points_np.setShaderInputs(
            point_size=100,
            border_size=0.05,
            point_color=(1,1,1,1),
            border_color=(0,0,0,1),
            selected_color=(1,0,1,1),
            selected_id = -1,
        )
        self.points_np.setTransparency(TransparencyAttrib.MNone) # TODO test 


    def setup_ray(self):
        self.camLens.setNearFar(1.0, 10000.0)
        self.ray_h = 0.0
        self.ray_p = 0.0
        # Node attached to camera to track its movement, we rotate it locally for the ray
        
        self.ray_pivot = self.camera.attachNewNode("ray_pivot")

        # Create the line pointing forward (+Y)
        segs = LineSegs()
        segs.setColor(1, 0, 1, 1)           # purple
        segs.setThickness(3.0)
        segs.moveTo(0, 1, 0)
        segs.drawTo(0, 5000, 0)             # long line

        ray_geom = NodePath(segs.create())
        ray_geom.setLightOff()
        ray_geom.setTransparency(TransparencyAttrib.MAlpha)
        ray_geom.setTwoSided(True)
        ray_geom.reparentTo(self.ray_pivot)

        self.ray_pivot.hide()


    def draw_orbit(self, sat_idx):
        if self.orbit_visual_node:
            self.orbit_visual_node.removeNode()
            self.orbit_visual_node = None

        if sat_idx < 0 or sat_idx >= len(self.satellite_orbits):
            return

        sat = self.satellite_orbits[sat_idx]
        if sat.no_kozai == 0:
            return

        # Period in minutes: 2*pi / mean_motion (rad/min)
        period_mins = 2 * np.pi / sat.no_kozai
        period_days = period_mins / 1440.0
        
        num_points = 250
        offsets = np.linspace(0, period_days, num_points)
        
        now = datetime.now(timezone.utc)
        jd_start, fr_start = jday_datetime(now)
        
        jd_array = np.full(num_points, jd_start)
        fr_array = fr_start + offsets
        
        error, positions, velocities = sat.sgp4_array(np.array(jd_array), np.array(fr_array))
        
        positions = (positions * SCALE).astype(np.float32)
        
        segs = LineSegs()
        segs.setColor(0, 1, 0, 1)
        segs.setThickness(2.0)
        
        # Start line
        segs.moveTo(positions[0][0], positions[0][1], positions[0][2])
        
        for i in range(1, num_points):
            segs.drawTo(positions[i][0], positions[i][1], positions[i][2])
            
        self.orbit_visual_node = self.render.attachNewNode(segs.create())
        self.orbit_visual_node.setLightOff()
        self.orbit_visual_node.setBin("fixed", 100)
        self.orbit_visual_node.setDepthTest(True)


    def process_selection(self):
        if not hasattr(self, 'scaled_positions') or len(self.visible_orbits) == 0:
            return

        cam_pos = np.array(self.camera.getPos(self.render), dtype=np.float32)
        
        # Get ray direction from the visual ray node
        quat = self.ray_pivot.getQuat(self.render)
        fwd = quat.getForward()
        ray_dir = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)

        local_selected_id = -1
        vecs = self.scaled_positions.reshape(-1,3) - cam_pos # reshape removes extra array dimension for time
        dists = np.linalg.norm(vecs, axis=1)
        dists = np.maximum(dists, 1e-6)
        
        dots = np.sum(vecs * ray_dir, axis=1)
        cos_angles = dots / dists
        
        threshold_cos = np.cos(np.radians(2.0))
        
        mask = cos_angles > threshold_cos
        candidate_indices = np.where(mask)[0]

        if len(candidate_indices) > 0:
            candidates_dists = dists[candidate_indices]
            
            best_local_idx = np.argmin(candidates_dists)
            local_selected_id = int(candidate_indices[best_local_idx])

            # Logic to cycle if clicking same spot?
            # Requires tracking local id selection or mapping?
            # For now simpler logic: just pick closest
            
            self.selected_global_id = self.visible_indices[local_selected_id]
            self.draw_orbit(self.selected_global_id)
            
            sat_name = self.satellite_infos[self.selected_global_id].get('OBJECT_NAME', 'Unknown')
            print(f"Selected: {sat_name} (ID: {self.selected_global_id})")


    def render_satellites(self, task):
        self.update_ui()
        if task.frame % 2 == 0:
            return task.cont

        if not self.visible_orbits:
             return task.cont

        time = datetime.now(timezone.utc)
        jd, fr = jday_datetime(time)
        orbits = SatrecArray(self.visible_orbits).sgp4(np.array([jd]), np.array([fr]))
        _, positions, velocities = orbits

        vertex_writer = GeomVertexWriter(self.vertex_data, "vertex")
        vertex_writer.setRow(0)

        self.scaled_positions = (positions * SCALE).astype(np.float32)
        num_rows = len(self.scaled_positions)
        # vertex_data already resized in apply_filters
        
        array_handle = self.vertex_data.modifyArray(0)
        memory_view = memoryview(array_handle)
        np_buffer = np.frombuffer(memory_view, dtype=np.float32)
        np_buffer[:] = self.scaled_positions.flatten()
        
        # Update shader selection
        shader_selected_id = -1
        if self.selected_global_id != -1:
            try:
                # Find if global ID is currently visible
                # visible_indices is sorted if created in order
                # We can use binary search or just index if we trust it, but index is slow on list
                # Since we rebuild every filter change, let's just use try/except on index
                shader_selected_id = self.visible_indices.index(self.selected_global_id)
            except ValueError:
                shader_selected_id = -1
        
        if ENABLE_SHADER:
             self.points_np.setShaderInput("selected_id", int(shader_selected_id))

        return task.cont


    def process_inputs(self, task):
        dt = self.clock.dt
        orbit_speed = 90.0 * dt
        zoom_speed = 100.0 * dt
        ray_speed = 90.0 * dt
        
        h_input, p_input = 0.0, 0.0
        zoom_input = 0.0
        ray_h_input, ray_p_input = 0.0, 0.0
        
        # Gamepad Input
        if self.gamepad:
            g_lx = self.gamepad.findAxis(InputDevice.Axis.left_x).value
            g_ly = self.gamepad.findAxis(InputDevice.Axis.left_y).value
            
            # Triggers
            l_trig = self.gamepad.findAxis(InputDevice.Axis.left_trigger).value
            r_trig = self.gamepad.findAxis(InputDevice.Axis.right_trigger).value
            
            # Right Stick (Ray)
            g_rx = self.gamepad.findAxis(InputDevice.Axis.right_x).value
            g_ry = self.gamepad.findAxis(InputDevice.Axis.right_y).value

            # Orbit (Left Stick)
            if abs(g_lx) > 0.1: h_input += g_lx 
            if abs(g_ly) > 0.1: p_input += g_ly
            
            # Zoom
            if abs(l_trig) > 0.05: zoom_input += l_trig
            if abs(r_trig) > 0.05: zoom_input -= r_trig
            
            # Ray
            if abs(g_rx) > 0.1: ray_h_input -= g_rx
            if abs(g_ry) > 0.1: ray_p_input += g_ry

        # Keyboard Input
        mw = self.mouseWatcherNode
        if mw.is_button_down(KeyboardButton.ascii_key('w')): p_input += 1.0
        if mw.is_button_down(KeyboardButton.ascii_key('s')): p_input -= 1.0
        if mw.is_button_down(KeyboardButton.ascii_key('d')): h_input += 1.0
        if mw.is_button_down(KeyboardButton.ascii_key('a')): h_input -= 1.0
        
        if mw.is_button_down(KeyboardButton.ascii_key('q')): zoom_input += 1.0
        if mw.is_button_down(KeyboardButton.ascii_key('e')): zoom_input -= 1.0

        #########################################
        if self.mDown:
            x=self.mouseWatcherNode.getMouseX()-self.oX
            y=self.mouseWatcherNode.getMouseY()-self.oY
            h_input += x
            p_input += y
        #########################################
        
        # Update Orbit State
        self.orbit_h += h_input * orbit_speed
        self.orbit_p += p_input * orbit_speed
        self.orbit_distance += zoom_input * zoom_speed
        
        self.orbit_p = max(-89.0, min(89.0, self.orbit_p))
        self.orbit_distance = max(10.0, min(500.0, self.orbit_distance))
        
        self.update_camera_pos()

        # Update Ray
        self.ray_h += ray_h_input * ray_speed
        self.ray_p += ray_p_input * ray_speed
        self.ray_pivot.setHpr(self.ray_h, self.ray_p, 0)

        return task.cont
    

SatelliteVisualizer().run()
