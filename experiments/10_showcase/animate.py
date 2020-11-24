"""Build animation from cockpit .json log."""

import glob
import os

from PIL import Image

from cockpit import CockpitPlotter


def animate(tproblem):
    """Build an animation from the logged .json file."""
    HERE = os.path.abspath(__file__)
    HEREDIR = os.path.dirname(HERE)

    RUN_DIR = os.path.join(HEREDIR, "results", tproblem, "SGD")
    RUN_PATTERN = os.path.join(RUN_DIR, "*/*__log.json")

    RUN_MATCHES = glob.glob(RUN_PATTERN)

    assert len(RUN_MATCHES) == 1, f"Found no or multiple files: {RUN_MATCHES}"
    RUN_MATCHES = RUN_MATCHES[0]

    cp = CockpitPlotter(RUN_MATCHES.replace(".json", ""))
    cp._read_tracking_results()

    # regenerate plots
    track_events = list(cp.tracking_data["iteration"])

    for idx, global_step in enumerate(track_events):
        print(f"Plotting {idx:05d}/{len(track_events):05d}")

        cp.plot(
            show_plot=False,
            block=False,
            show_log_iter=True,
            save_plot=True,
            savename_append=f"_animation_frame_{idx:05d}",
            discard=global_step,
        )

    # load frames
    frame_dir = os.path.dirname(os.path.splitext(cp.logpath)[0])
    pattern = os.path.join(frame_dir, "*_animation_frame_*.png")
    frame_paths = sorted(glob.glob(pattern))
    frame, *frames = [Image.open(f) for f in frame_paths]

    animation_savepath = os.path.join(frame_dir, "showcase.gif")

    # Collect images and create Animation
    frame.save(
        fp=animation_savepath,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=200,
        loop=0,
    )
