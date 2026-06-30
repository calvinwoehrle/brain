
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pydicom import dcmread
from pydicom.data import get_testdata_file

sample_image = Path('/Users/calvinwoehrle/Documents/MRI/CRM15_2026_06_23/CRM15.MR.Pi\'s_Shomstein_.20.28.2026.06.26.15.12.11.475.39609681.dcm')


# path = get_testdata_file(sample_image)


ds = dcmread(sample_image)


if False:
    print(str(ds))

    # Output str(ds) to a text file
    output_file = Path('Output/dicom_metadata.txt')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(str(ds))

    # type(ds.PixelData)

# Decode pixel data into a NumPy array.
# This is a multi-frame Enhanced MR object, so the array shape is
# (num_frames, rows, columns) -> e.g. (72, 96, 96).
pixels = ds.pixel_array
print("Pixel array shape:", pixels.shape)

# Pick which frame/slice to show first (0-based).
n_frames = pixels.shape[0]
start_frame = n_frames // 2

# Leave room at the bottom for the slider.
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

img = ax.imshow(pixels[start_frame], cmap="gray")
ax.axis("off")
title = ax.set_title(f"{ds.SeriesDescription} - frame {start_frame}")

# Add a horizontal slider to scroll through the frames.
slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])
frame_slider = Slider(
    ax=slider_ax,
    label="Frame",
    valmin=0,
    valmax=n_frames - 1,
    valinit=start_frame,
    valstep=1,
)


def update(val):
    frame = int(frame_slider.val)
    img.set_data(pixels[frame])
    title.set_text(f"{ds.SeriesDescription} - frame {frame}")
    fig.canvas.draw_idle()


frame_slider.on_changed(update)

# Optional: scroll the mouse wheel over the image to step frames.
def on_scroll(event):
    step = 1 if event.button == "up" else -1
    new_frame = int(min(max(frame_slider.val + step, 0), n_frames - 1))
    frame_slider.set_val(new_frame)


fig.canvas.mpl_connect("scroll_event", on_scroll)

plt.show()
