import matplotlib.pyplot as plt
import numpy as np

def getNumFromOneHot(inp):
    for i in range(10):
        if inp[i] == 1:
            return i


def draw_stroke_sequence(sequence, save_path=None, show=True):
    """
    sequence: numpy array or list of shape (T, 4) where each row is [dx, dy, eos, eod]
    save_path: optional path to save the plot as an image
    show: whether to display the plot
    """
    x, y = 0, 0
    xs, ys = [], []

    for dx, dy, eos, eod in sequence:
        x += dx*28
        y += dy*28
        xs.append(x)
        ys.append(y)

        if eos > 0.5:  # end of stroke
            xs.append(None)
            ys.append(None)

        if eod > 0.5:
            break

    plt.plot(xs, ys, linewidth=2)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.axis('equal')
    
datas = [[] for _ in range(10)]

for i in range(10000):
    try:
        data = np.loadtxt(f'sequences/testimg-{i}-targetdata.txt', delimiter=' ')
    except FileNotFoundError:
        print(f"❌ File not found at path: {i}")
        continue
    
    inputOneshot = data[0, 0:10]
    outputStrokes = data[:, 10:]
    outputStrokes[:, 0] = outputStrokes[:, 0]/28
    outputStrokes[:, 1] = outputStrokes[:, 1]/28
    
    datas[getNumFromOneHot(inputOneshot)].append(outputStrokes)
    
    
input_data = []
images = []

for i in range(10):
    temp_onehot = np.zeros(10)
    temp_onehot[i] = 1
    
    smallest_10 = sorted(datas[i], key=len)[:100]
    for k in smallest_10:
        input_data.append(temp_onehot)
        images.append(k)

# Storage for accepted and rejected images
accepted = []
rejected = []

# Iterator index
index = 0


def on_key(event):
    global index
    
    if event.key.lower() == 'y':
        accepted.append(images[index])
    elif event.key.lower() == 'n':
        rejected.append(images[index])
    else:
        print("Press 'y' for yes, 'n' for no.")
        return
    
    index += 1
    plt.clf()

    if index < len(images):
        draw_stroke_sequence(images[index])
        plt.title(f"Image {index+1}/{len(images)} — Press Y (yes) or N (no)")
        plt.draw()
    else:
        print("Done reviewing images!")
        plt.close()

# Show the first image
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)
draw_stroke_sequence(images[index])
plt.title(f"Image {index+1}/{len(images)} — Press Y (yes) or N (no)")
plt.show()
