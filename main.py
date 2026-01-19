#!/usr/bin/env python3

from datetime import datetime, timezone
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    loadPrcFile,
    Vec3,
    Shader,
    DirectionalLight,
    AmbientLight,
    InputDevice,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomPoints, GeomNode,
    GraphicsPipe, GraphicsOutput, Texture, FrameBufferProperties,
    ShaderAttrib, LColor, CardMaker,
    TransparencyAttrib, OrthographicLens, Camera, NodePath, LineSegs,
    GamepadButton, KeyboardButton
)
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
    def __init__(self):
        ShowBase.__init__(self)

        self.disableMouse()
        self.setBackgroundColor(Vec3(0,0,0))

        self.init_gmst()
        self.setup_camera()
        self.setup_light() # TODO fix rotation
        self.setup_earth()

        self.gamepad = None
        self.setup_gamepad()

        self.load_omm_model()

        self.setup_shader()
        self.setup_ray()
        
        self.accept("space", self.process_selection)
        self.accept("gamepad-face_a", self.process_selection)
        self.taskMgr.add(self.process_inputs, "input")
        self.taskMgr.add(self.render_satellites, "render")


    def init_gmst(self):
        start_time = datetime.now(timezone.utc)
        jd, fr = jday_datetime(start_time)
        self.gstime = gstime(jd + fr)


    def setup_camera(self):
        # Orbit Camera State
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
        self.selected_satellite = -1
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
        segs.moveTo(0, 0, 0)
        segs.drawTo(0, 5000, 0)             # long line

        ray_geom = NodePath(segs.create())
        ray_geom.setLightOff()
        ray_geom.setTransparency(TransparencyAttrib.MAlpha)
        ray_geom.setTwoSided(True)
        ray_geom.reparentTo(self.ray_pivot)


    def process_selection(self):
        cam_pos = np.array(self.camera.getPos(self.render), dtype=np.float32)
        
        # Get ray direction from the visual ray node
        quat = self.ray_np.getQuat(self.render)
        fwd = quat.getForward()
        ray_dir = np.array([fwd.x, fwd.y, fwd.z], dtype=np.float32)

        selected_id = -1
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
            selected_id = int(candidate_indices[best_local_idx])

            if selected_id == self.selected_satellite and len(candidates_dists) > 1:
                candidates_dists[best_local_idx] = np.inf
                
                best_local_idx = np.argmin(candidates_dists)
                print(f"dist {candidates_dists[best_local_idx]}")
                selected_id = int(candidate_indices[best_local_idx])

            self.selected_satellite = selected_id
            self.points_np.setShaderInput("selected_id", self.selected_satellite)


    def render_satellites(self, task):
        if task.frame % 2 == 0:
            return task.cont

        time = datetime.now(timezone.utc)
        jd, fr = jday_datetime(time)
        orbits = SatrecArray(self.satellite_orbits).sgp4(np.array([jd]), np.array([fr]))
        _, positions, velocities = orbits

        vertex_writer = GeomVertexWriter(self.vertex_data, "vertex")
        vertex_writer.setRow(0)

        self.scaled_positions = (positions * SCALE).astype(np.float32)
        num_rows = len(self.scaled_positions)
        self.vertex_data.setNumRows(num_rows)
        array_handle = self.vertex_data.modifyArray(0)
        memory_view = memoryview(array_handle)
        np_buffer = np.frombuffer(memory_view, dtype=np.float32)
        np_buffer[:] = self.scaled_positions.flatten()

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
