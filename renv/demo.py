import time
import pathlib
from multiprocessing.managers import SharedMemoryManager

from ril_env.spacemouse import Spacemouse
from ril_env.xarm_controller import XArm, XArmConfig
from ril_env.multi_realsense import MultiRealsense
from ril_env.video_recorder import VideoRecorder


def main():
    xarm_config = XArmConfig()

    video_dir = pathlib.Path("recordings")
    video_dir.mkdir(parents=True, exist_ok=True)

    video_recorder = VideoRecorder.create_h264(
        fps=30,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=21,
        thread_type="FRAME",
        thread_count=3,
    )

    with SharedMemoryManager() as shm_manager, Spacemouse(deadzone=0.4) as sm, XArm(
        xarm_config
    ) as arm, MultiRealsense(
        shm_manager=shm_manager,
        record_fps=30,
        capture_fps=30,
        put_fps=30,
        put_downsample=True,
        video_recorder=video_recorder,
        verbose=True,
    ) as multi:

        print("MultiRealsense cameras started and ready.")
        multi.set_exposure(exposure=120, gain=0)
        multi.set_white_balance(white_balance=5900)

        multi.start_recording(str(video_dir), start_time=time.time())
        print(f"Recording started. Videos will be saved in {video_dir.absolute()}.")

        try:
            while True:
                loop_start = time.monotonic()

                time.sleep(0.01)

                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]
                drot = sm_state[3:]
                grasp = sm.grasp

                if sm.is_button_pressed(1):
                    arm.home()
                    continue

                arm.step(dpos, drot, grasp)

                elapsed = time.monotonic() - loop_start
                sleep_time = max(0, 0.02 - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            multi.stop_recording()
            print("Recording stopped.")


if __name__ == "__main__":
    main()
