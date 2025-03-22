import mujoco
import mujoco.viewer
import numpy as np
import time

xml_path = "ant.xml"
with open(xml_path, 'r') as f:
    xml_string = f.read()
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
ctrl_range = model.actuator_ctrlrange  

with mujoco.viewer.launch_passive(model, data) as viewer:
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,'torso')
    
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = torso_id
    viewer.cam.distance = 7.0
    viewer.cam.azimuth = 0.0
    viewer.cam.elevation = -20.0
    time.sleep(3)
    start_time = time.time()
    while viewer.is_running():
        data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
