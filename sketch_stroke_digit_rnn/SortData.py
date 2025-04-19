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
    

datas = []
actual = []

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
    
    actual.append(getNumFromOneHot(inputOneshot))
    datas.append((i, outputStrokes))

# Storage for accepted and rejected images
accepted = []
rejected = []

# Iterator index
index = 548


def on_key(event):
    global index
    
    if event.key.lower() == 'y':
        accepted.append(datas[index][0])
    elif event.key.lower() == 'n':
        rejected.append(datas[index][0])
    else:
        print("Press 'y' for yes, 'n' for no.")
        return
    
    index += 1
    plt.clf()

    if index < len(datas):
        draw_stroke_sequence(datas[index][1])
        plt.title(f"Image {index+1}/{len(datas)} — Press Y (yes) or N (no) Number: {actual[index]}")
        plt.draw()
    else:
        print("Done reviewing images!")
        plt.close()

# Show the first image
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)
draw_stroke_sequence(datas[index][1])
plt.title(f"Image {index+1}/{len(datas)} — Press Y (yes) or N (no) {actual[index]}")
plt.show()

print(accepted)