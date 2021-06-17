import matplotlib.pyplot as plt
import numpy as np

def energy_rank(data, marker_width=.5, color='blue'):
    y_data = np.repeat(data, 2)
    x_data = np.empty_like(y_data)
    x_data[0::2] = np.arange(1, len(data)+1) - (marker_width/2)
    x_data[1::2] = np.arange(1, len(data)+1) + (marker_width/2)
    lines = []
    lines.append(plt.Line2D(x_data, y_data, lw=1, linestyle='dashed', color=color))
    for x in range(0,len(data)*2, 2):
        lines.append(plt.Line2D(x_data[x:x+2], y_data[x:x+2], lw=2, linestyle='solid', color=color))
    return lines

data = np.random.rand(4,8) * 4 # 4 lines with 8 datapoints from 0 - 4

artists = []
for row, color in zip(data, ('red','blue','green','magenta')):
    artists.extend(energy_rank(row, color=color))

fig, ax = plt.subplots()

for artist in artists:
    ax.add_artist(artist)
ax.set_ybound([0,4])
ax.set_xbound([.5,8.5])
plt.show()
