from mjpy import MjViewer, MjModel, register_license
import os

register_license(os.path.join(os.path.dirname(__file__),
                              '../vendor/mujoco_osx/LICENSE_DEEPRL.TXT'))


if __name__ == "__main__":
    viewer = MjViewer()
    viewer.start()
    model = MjModel(os.path.join(os.path.dirname(__file__),
                              '../vendor/mujoco_osx/humanoid.xml'))
    viewer.set_model(model)
    while True:
        viewer.loop_once()
        model.step()
    viewer.finish()
