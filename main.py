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
from math import sin, radians, degrees
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
        self.setup_selection_ray()

        self.load_omm_model()

        self.setup_shader()

        self.accept("space", self.process_selection)
        self.accept("gamepad-face_a", self.process_selection)
        self.taskMgr.add(self.process_inputs, "input")
        self.taskMgr.add(self.render_satellites, "render")


    def init_gmst(self):
        start_time = datetime.now(timezone.utc)
        jd, fr = jday_datetime(start_time)
        self.gstime = gstime(jd + fr)


    def setup_camera(self):
        self.camera.setPos(100,50,50)
        self.camera.lookAt(0,0,0)
    

    def setup_light(self):
        time = datetime.now(timezone.utc)
        day_of_year = time.timetuple().tm_yday
        sun_declination = 23.44 * sin(radians((360/365)*(day_of_year - 81)))
        sun_angle = degrees(self.gstime)

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
        earth_angle = degrees(self.gstime)
        earth_angle = 0

        self.earth = self.loader.loadModel("models/planet_sphere")
        self.earth_tex = self.loader.loadTexture("textures/earth_1k_tex.jpg")
        self.earth.setTexture(self.earth_tex, 1)
        self.earth.setScale(EARTH_RADIUS * SCALE)
        self.earth.setHpr(EARTH_TEXTURE_OFFSET + earth_angle, 0, 0)
        self.earth.reparentTo(self.render)


    def setup_selection_ray(self):
        self.selection_ray = LineSegs()
        self.selection_ray.setThickness(2.0)
        self.selection_ray.setColor(1, 0, 0, 1) # Red color
        self.selection_ray.moveTo(0, 0, 0)
        self.selection_ray.drawTo(0, 100, 0) # Initial direction
        self.selection_ray_node = self.render.attachNewNode(self.selection_ray.create())
        self.selection_ray_node.hide()


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
        speed = 50.0 * dt
        rot_speed = 90.0 * dt
        
        move_x, move_y = 0.0, 0.0
        rot_h, rot_p = 0.0, 0.0
        selection_pressed = False

        # Gamepad Input
        if self.gamepad:
            g_lx = self.gamepad.findAxis(InputDevice.Axis.left_x).value
            g_ly = self.gamepad.findAxis(InputDevice.Axis.left_y).value
            g_rx = self.gamepad.findAxis(InputDevice.Axis.right_x).value
            g_ry = self.gamepad.findAxis(InputDevice.Axis.right_y).value
            
            if abs(g_lx) > 0.1: move_x += g_lx
            if abs(g_ly) > 0.1: move_y += g_ly
            if abs(g_rx) > 0.1: rot_h -= g_rx
            if abs(g_ry) > 0.1: rot_p += g_ry
            
            if self.gamepad.findButton(GamepadButton.face_a()).pressed:
                selection_pressed = True

        # Keyboard Input
        mw = self.mouseWatcherNode
        if mw.is_button_down(KeyboardButton.ascii_key('w')): move_y += 1.0
        if mw.is_button_down(KeyboardButton.ascii_key('s')): move_y -= 1.0
        if mw.is_button_down(KeyboardButton.ascii_key('a')): move_x -= 1.0
        if mw.is_button_down(KeyboardButton.ascii_key('d')): move_x += 1.0
        
        if mw.is_button_down(KeyboardButton.up()):    rot_p += 1.0
        if mw.is_button_down(KeyboardButton.down()):  rot_p -= 1.0
        if mw.is_button_down(KeyboardButton.left()):  rot_h += 1.0
        if mw.is_button_down(KeyboardButton.right()): rot_h -= 1.0
        
        if mw.is_button_down(KeyboardButton.space()) or mw.is_button_down(KeyboardButton.enter()):
            selection_pressed = True
        
        # Clamp combined inputs
        move_x = max(-1.0, min(1.0, move_x))
        move_y = max(-1.0, min(1.0, move_y))
        rot_h = max(-1.0, min(1.0, rot_h))
        rot_p = max(-1.0, min(1.0, rot_p))

        # Move Camera
        self.camera.setPos(self.camera, Vec3(move_x * speed, move_y * speed, 0))
        
        # Rotate Camera
        self.camera.setH(self.camera.getH() + rot_h * rot_speed)
        self.camera.setP(self.camera.getP() + rot_p * rot_speed)
        
        # Update Visual Ray
        self.selection_ray_node.show()
        self.selection_ray_node.setPos(self.camera.getPos(self.render))
        self.selection_ray_node.setQuat(self.camera.getQuat(self.render))
        self.selection_ray_node.setScale(10.0) 

        return task.cont
    

    def process_selection(self):
        cam_pos = np.array(self.camera.getPos(self.render), dtype=np.float32)
        quat = self.camera.getQuat(self.render)
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


SatelliteVisualizer().run()