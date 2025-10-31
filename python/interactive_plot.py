import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.ndimage import gaussian_filter
from util_graphics import intensities2detint_e4c, intensities2detint_e6c

# ErRuSi2
cif_path = '/epics/crystals/ErRu2Si2/EntryWithCollCode55782.cif'
hkl_path = '/epics/crystals/ErRu2Si2/EntryWithCollCode55782.hkl'

# parameters
gauss_sig = 2
#E6C
gamma_axis = [0, 0, 1]
delta_axis = [0, -1, 0]
#E4C
tth_axis = [0,-1,0]

R = 70 
geom='E6C'
#geom='E4CV' #E4CV axes align with E6C
#wavelength = 1.55
wavelength = 1.486
#wavelength =  1.4
darwidth=2
cyl_center = (0, 0)
ray_origin = np.array([0.0, 0.0, 0.0])
min_intensity = 10
window_width = 120

angle_deg = 20
zmax = R * np.tan(np.deg2rad(angle_deg))
zmin = -zmax
y_range = 2 * zmax

det_angle_deg = 7.5
det_zmax = R * np.tan(np.deg2rad(det_angle_deg))
det_zmin = -det_zmax
det_height = det_zmax - det_zmin

if (geom=='E4CV') or (geom=='E4CH'):
    lst = intensities2detint_e4c(
        cif_path, hkl_path, wavelength, min_intensity, R, geom,
        cyl_center, ray_origin, zmin, zmax, tth_axis)
    if lst != []:
        data = np.array(lst)
        theta, z, intensity, h, k, l, omega, chi, phi, tth = (
            data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5],
            data[:, 6], data[:, 7], data[:, 8], data[:,9]
        )
    else:
        print("NO DATA")
elif geom=='E6C':
    lst = intensities2detint_e6c(
        cif_path, hkl_path, wavelength, min_intensity, R, geom,
        cyl_center, ray_origin, zmin, zmax, gamma_axis, delta_axis)
    if lst != []:
        data = np.array(lst)
        theta, z, intensity, h, k, l, mu, omega, chi, phi, gamma, delta = (
            data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5],
            data[:, 6], data[:, 7], data[:, 8], data[:, 9], data[:, 10], data[:, 11]
        )
    else:
        print("NO DATA")

mult=5
nx, ny = int(mult*window_width), int(mult*y_range)
print(nx)
theta_grid = np.linspace(0, 360, nx, endpoint=False)
z_grid = np.linspace(zmin, zmax, ny)

initial_y_offset = 0
initial_angle = -180
initial_angle_center = initial_angle + 60

visible_peaklist = []


def compute_heatmap(threshold, cur_angle, geom):
    peaklist = []
    heatmap = np.zeros((ny, nx))
    if (geom=='E4CV') or (geom=='E4CH'):
        for t, zz, inten, o, hh, kk, ll, tt in zip(theta, z, intensity, omega, h, k, l, tth):
            if (inten > threshold) and ((cur_angle - darwidth) <= o <= (cur_angle + darwidth)):
                i = int(nx * t / 360) % nx
                j = np.searchsorted(z_grid, zz)
                if 0 <= j < ny:
                    heatmap[j, i] += inten
                    peaklist.append({
                        'h': hh, 'k': kk, 'l': ll, \
                        'theta': t, 'z': zz, 'intensity': inten, 'omega':o, \
                        'tth':tt})
        blurred = gaussian_filter(heatmap, sigma=gauss_sig)
        if blurred.max() != 0:
            blurred /= blurred.max()
        return heatmap, blurred, peaklist
    elif (geom=='E6C'):
        for t, zz, inten, o, hh, kk, ll, ga, de in zip(theta, z, intensity, omega, h, k, l, gamma, delta):
            if (inten > threshold) and ((cur_angle - darwidth) <= o <= (cur_angle + darwidth)):
                i = int(nx * t / 360) % nx
                j = np.searchsorted(z_grid, zz)
                if 0 <= j < ny:
                    heatmap[j, i] += inten
                    peaklist.append({
                        'h': hh, 'k': kk, 'l': ll, \
                        'theta': t, 'z': zz, 'intensity': inten, 'omega':o, \
                        'gamma':ga, 'delta':de})
        blurred = gaussian_filter(heatmap, sigma=gauss_sig)
        #blurred = heatmap
        if blurred.max() != 0:
            blurred /= blurred.max()
        return heatmap, blurred, peaklist


heatmap, heatmap_blurred, peaklist = compute_heatmap(min_intensity, initial_angle_center, geom)
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.4)

def slice_window(hmap, y_offset, det_zmin, det_zmax,
                 nx, ny, y_range, window_width, det_height):
    z_start = det_zmin + y_offset
    z_end = z_start + (det_zmax - det_zmin)
    j_start = int(ny * z_start / y_range) + int(ny / 2)
    j_end = j_start + int(ny * det_height / y_range)
    i_start = 0
    i_end = int(nx*window_width/360)
    hmap_window = hmap[j_start:j_end, i_start:i_end]
    theta_extent = [0, 120] # detector width
    z_extent = [z_start, z_end]
    return hmap_window, theta_extent, z_extent

heatmap_window, theta_extent, z_extent = slice_window(heatmap_blurred, initial_y_offset, \
    det_zmin, det_zmax, nx, ny, y_range, window_width, det_height)

heatmap_im = ax.imshow(
    heatmap_window,
    extent=[theta_extent[0], theta_extent[1], z_extent[0], z_extent[1]],
    origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1
)

#heatmap_im = ax.imshow(
#    heatmap_window,
#    extent=[theta_extent[0], theta_extent[1], z_extent[0], z_extent[1]],
#    origin='lower', aspect='auto', cmap='viridis')

plt.colorbar(heatmap_im, ax=ax, label="Intensity")
ax.set_xlabel("detector theta (°)")
ax.set_ylabel("z")

axcolor = 'lightgoldenrodyellow'
ax_slider_angle = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)
slider_angle = Slider(ax_slider_angle, 'sample omega (°)', -180, 60, valinit=initial_angle)

ax_slider_y = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
slider_y = Slider(ax_slider_y, 'z offset', -10, 10, valinit=initial_y_offset)

ax_text_minint = plt.axes([0.2, 0.12, 0.1, 0.03], facecolor=axcolor)
text_minint = TextBox(ax_text_minint, 'min Intensity', initial=str(min_intensity))


def update(val):
    global heatmap_blurred, visible_peaklist
    angle_start = slider_angle.val
    cur_angle_center = angle_start + 60
    y_offset = slider_y.val
    try:
        current_min_intensity = float(text_minint.text)
    except ValueError:
        current_min_intensity = min_intensity

    heatmap, heatmap_blurred, peaklist = compute_heatmap(current_min_intensity, cur_angle_center, geom)

    hmap_win, theta_extent, z_extent = slice_window(
        heatmap_blurred, y_offset,
        det_zmin, det_zmax, nx, ny, y_range, window_width, det_height
    )
    heatmap_im.set_data(hmap_win)
    heatmap_im.set_extent([theta_extent[0], theta_extent[1], z_extent[0], z_extent[1]])
    ax.set_xlim(*theta_extent)
    ax.set_ylim(*z_extent)

    visible_peaklist.clear()
    for peak in peaklist:
        if (z_extent[0] <= peak['z'] <= z_extent[1]) and (theta_extent[0] <= peak['theta'] <= theta_extent[1]):
            visible_peaklist.append(peak)

    fig.canvas.draw_idle()


def hover(event):
    if event.inaxes != ax:
        annot.set_visible(False)
        fig.canvas.draw_idle()
        return

    xdata, ydata = event.xdata, event.ydata # theta, z
    if xdata is None or ydata is None or not visible_peaklist:
        annot.set_visible(False)
        fig.canvas.draw_idle()
        return

    threshold = 0.1
    nearby_peaks = []

    for peak in visible_peaklist:
        dist = np.sqrt(((peak['theta'] - (xdata%360))/window_width)**2 + ((peak['z'] - ydata)/y_range)**2)
        #dist = np.hypot(peak['theta'] - xdata, peak['z'] - ydata)
        if dist < threshold:
            nearby_peaks.append(peak)

    if nearby_peaks:
        annot.xy = (nearby_peaks[0]['theta'], nearby_peaks[0]['z'])
        if (geom=='E4CV') or (geom=='E4CH'):
            text_lines = [
                f"hkl=({p['h']},{p['k']},{p['l']}) θ={p['theta']:.1f} z={p['z']:.2f},\nome={p['omega']:.1f}, tth={p['tth']:.1f}" for p in nearby_peaks]
        elif (geom=='E6C'):
            text_lines = [
                f"hkl=({p['h']},{p['k']},{p['l']}) θ={p['theta']:.1f} z={p['z']:.2f},\nome={p['omega']:.1f}, gamma={p['gamma']:.1f}, delta={p['delta']:.1f}" for p in nearby_peaks]
        annot.set_text('\n'.join(text_lines))
        annot.set_visible(True)
    else:
        annot.set_visible(False)

    fig.canvas.draw_idle()

annot = ax.annotate(
    "", xy=(0, 0), xytext=(20, 20),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->")
)
annot.set_visible(False)

slider_angle.on_changed(update)
slider_y.on_changed(update)
text_minint.on_submit(lambda text: update(None))
update(0)
fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()

