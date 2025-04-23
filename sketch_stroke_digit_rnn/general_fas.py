import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import copy
from PIL import Image
import io
from torchvision import transforms
import torch.nn.functional as F
import os

def getNumFromOneHot(inp):
    for i in range(10):
        if inp[i] == 1:
            return i
        
def draw_stroke_sequencefilepath(sequence, save_path=None, show=True):
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

    plt.figure(figsize=(2, 2))
    plt.plot(xs, ys, linewidth=2)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Function to draw the number using the RNN

def generate_text_savefile(model, number, filepath):
    model.eval()
    
    temp_onehot = np.zeros(10)
    temp_onehot[number] = 1
    temp_onehot = torch.tensor(temp_onehot, dtype=torch.float32).to(device)
    
    initial_input = torch.tensor([0, 0, 0, 0], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(1)
    
    outputs = []
    
    output, hidden = model(initial_input, onehot_digit=temp_onehot)
    output[..., 2:] = (torch.sigmoid(output[..., -1, 2:]) > 0.5).float()

    outputs.append(output[:, -1, :].detach().cpu().numpy()[0])

    for i in range(62-1):
        output, hidden = model(output, hidden=hidden)
        output[..., 2:] = (torch.sigmoid(output[..., -1, 2:]) > 0.5).float()
        outputs.append(output[:, -1, :].detach().cpu().numpy()[0])
        
        # print(outputs[-1])
        if output[:, -1, 3] == 1:
            # print("HI")
            break
    
    draw_stroke_sequencefilepath(outputs, save_path=filepath)

# DAMAGE WEIGHTS

def damage_smallest(model, p_smallest): # energy constraint
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            if p_smallest == 0:
                continue

            tensor = param.data
            weight_magnitudes = tensor.abs().view(-1)
            k = int(weight_magnitudes.numel() * p_smallest)

            if k == 0:
                continue
            threshold = weight_magnitudes.kthvalue(k).values.item()

            mask = tensor.abs() >= threshold
            param.data.mul_(mask)

def damage_fas(model, p_block, p_reflect, p_filter):
    for name, param in model.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            if p_block + p_reflect + p_filter == 0:
                continue

            tensor = param.data
            flat_weights = tensor.view(-1)
            nonzero_indices = (flat_weights!=0).nonzero(as_tuple=True)[0]
            num_nonzero_indices = nonzero_indices.numel()
            if num_nonzero_indices == 0:
                continue

            # percentage of weights damaged will be taken from the number of nonzero weights
            # simulated fas damage occurs after energy constraint blockage
            num_block = int(num_nonzero_indices * p_block)
            num_reflect = int(num_nonzero_indices * p_reflect)
            num_filter = int(num_nonzero_indices * p_filter)

            shuffled_indices = nonzero_indices[torch.randperm(num_nonzero_indices, device=flat_weights.device)]

            indices_block = shuffled_indices[:num_block]
            indices_reflect = shuffled_indices[num_block:num_block+num_reflect]
            indices_filter = shuffled_indices[num_block+num_reflect:num_block+num_reflect+num_filter]

            # do damage
            # blockage: set weights to 0
            if p_block != 0:
                flat_weights[indices_block] = 0

            # reflect: halve weights
            if p_reflect != 0:
                flat_weights[indices_reflect] *= 0.5

            # filter: low pass filter (lusch et al)
            if p_filter != 0:
                weights_to_filter = flat_weights[indices_filter]            # get weights before transformation
                signs = torch.sign(weights_to_filter)                       # get signs of weights
                abs_weights_to_filter = weights_to_filter.abs()             # get high_weight, should be in the 95th percentile for all weights
                high_weight = torch.quantile(flat_weights.abs(), 0.95)      # scale weights to mostly between -1 and 1
                x = abs_weights_to_filter / high_weight
                transformed_weights = -0.2744 * x**2 + 0.9094 * x - 0.0192
                gaussian_noise = torch.randn_like(transformed_weights) * 0.05
                transformed_weights += gaussian_noise
                transformed_weights = transformed_weights * signs * high_weight # rescale
                flat_weights[indices_filter] = transformed_weights

class DigitToStrokeLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, batch_size=32):
        super(DigitToStrokeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.embedding = nn.Linear(10, hidden_size)  # From one-hot to hidden dim
        
        # LSTM
        # Output layer: predicts [dx, dy, eos, eod]
        # Inital hidden state is the one-hot of number
        # Initial input is [0, 0, 0, 0, 0]
        # Input at t > 0 is output from t-1
        
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        # Output layer: predicts [dx, dy, eos, eod]
        self.output_head = nn.Linear(hidden_size, 4)
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()  # For eos/eod
        self.tanh = nn.Tanh()


    def forward(self, x, hidden=None, onehot_digit=None):
        
        if onehot_digit != None and hidden == None:
            # Embed the digit
            h0 = self.embedding(onehot_digit)
            h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)

        elif hidden == None and onehot_digit == None:
            hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                      torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
            
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        
        out = self.output_head(out)
        
        out[:, :, 0:2] = self.tanh(out[:, :, 0:2])
        # out[:, :, 2:] = self.sigmoid(out[:, :, 2:])
        
        return out, hidden
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(__file__)



# generate images

# model_path = os.path.join(base_dir, "model_weights", f"sketch_model_weights2.pth")
# image_path = os.path.join(base_dir, "images", "model2_drawings", "filter")
# for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#     for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         print(x)
#         model = DigitToStrokeLSTM(hidden_size=512, num_layers=2).to(device)
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()

#         # damage_smallest(model, x) # might need to change??
#         damage_fas(model, 0.0, 0.0, x) # might need to change??

#         filename = f"{i}_{x:.1f}.png"
#         filepath = os.path.join(image_path, filename)

#         generate_text_savefile(model, i, filepath)





def generate_text(model, number):
    model.eval()
    
    temp_onehot = np.zeros(10)
    temp_onehot[number] = 1
    temp_onehot = torch.tensor(temp_onehot, dtype=torch.float32).to(device)
    
    initial_input = torch.tensor([0, 0, 0, 0], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(1)
    
    outputs = []
    
    output, hidden = model(initial_input, onehot_digit=temp_onehot)
    output[..., 2:] = (torch.sigmoid(output[..., -1, 2:]) > 0.5).float()

    outputs.append(output[:, -1, :].detach().cpu().numpy()[0])

    for i in range(62-1):
        output, hidden = model(output, hidden=hidden)
        output[..., 2:] = (torch.sigmoid(output[..., -1, 2:]) > 0.5).float()
        outputs.append(output[:, -1, :].detach().cpu().numpy()[0])
        
        # print(outputs[-1])
        if output[:, -1, 3] == 1:
            # print("HI")
            break
    
    return draw_stroke_sequence(outputs)

def draw_stroke_sequence(sequence):
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

    # Load onto variable img_array
    plt.figure(figsize=(1, 1), dpi=28)  # 1 inch * 28 dpi = 28 pixels
    plt.plot(xs, ys, color = "black", linewidth=2)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.axis('equal')
    # Use a BytesIO buffer to save the plot into memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=False, facecolor='white')
    plt.close()
    buf.seek(0)  # Rewind the buffer to the beginning
    img = Image.open(buf)  # Open the image from the buffer
    
    img_array = np.array(img.convert('L'))  # Convert to grayscale (1 channel) as a numpy array
    
    buf.close()  # Close the buffer
    
    return img_array

def evaluate_img(model, img, display=False):
    # Step 2: Preprocess the image
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert to PyTorch tensor (scales pixels to [0, 1])
    ])

    img_tensor = transform(img).to(device)

    img_tensor[img_tensor<0.6] = 0
    img_tensor[img_tensor>=0.6] = 1

    img_tensor = 1-img_tensor

    if display:
        imgDisplay = img_tensor.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions

        # Display the image using matplotlib
        plt.imshow(imgDisplay, cmap='gray')
        plt.axis('off')  # Turn off axis labels
        plt.show()


    # Step 3: Add batch dimension (PyTorch models expect a batch of images)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)

    # Step 4: Pass the tensor to the model
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Turn off gradients since we're in inference mode
        output = model(img_tensor)  # Pass the image tensor to the model for prediction

    # Step 5: Interpret the output
    _, predicted_class = torch.max(output, 1)  # Get the predicted class index
    
    # First return is the predicted class (int), and the second is an array containing confidences for each digit
    return predicted_class.item(), nn.Softmax(dim=1)(output).detach().cpu().numpy()[0]

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.model = nn.Sequential(
            # First Convolution Block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 32 filters, 3x3 kernel, 'same' padding
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32 filters, 3x3 kernel, 'same' padding
            nn.BatchNorm2d(32),
            
            nn.MaxPool2d(2, 2),  # Max pooling (2x2) with stride 2
            nn.Dropout(0.25),

            # Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 filters, 3x3 kernel, 'same' padding
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64 filters, 3x3 kernel, 'same' padding
            nn.BatchNorm2d(64),

            nn.MaxPool2d(2, 2),  # Max pooling (2x2) with stride 2
            nn.Dropout(0.25),

            # Fully Connected (Dense) layers
            nn.Flatten(),
            
            nn.Linear(64 * 7 * 7, 512),  # Input size depends on the output of convolutional layers
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 10)  # Output layer (10 classes)
        )
            
            

    def forward(self, x):
        x = self.model(x)
        return x





# GENERAL (fas) SIMULATION STARTS HERE!!!!!!

def test_model(model, model_version):
    cs = []
    times2 = []
    for mv in ['1', '2', '3', '4', '5']:

        cnn_path = os.path.join(base_dir, "cnn_weights", f"cnn_weights{mv}.pth")

        cnnEvaluator = CNNModel().to(device)
        cnnEvaluator.load_state_dict(torch.load(cnn_path, weights_only=True, map_location=torch.device('cpu')))
        cnnEvaluator.eval()  # set to evaluation mode if you're doing inference

        confidences = []
        times = []

        for digit in range(10):
            drawing, time = generate_text(model, digit)
            predicted_class, confidence = evaluate_img(cnnEvaluator, drawing, display=False)
            # print(f'Predicted Class: {predicted_class}')
            # print(f'Confidence of desired_digit = {confidence[digit]}')
            confidences.append(confidence[digit])
            times.append(time)
            
        cs.append(sum(confidences)/len(confidences))
        times2.append(sum(times)/len(times))

    return sum(cs)/len(cs), sum(times2)/len(times2)


cs_block = []
ts_block = []
cs_reflect = []
ts_reflect = []
cs_filter = []
ts_filter = []
for model_version in ['1', '2', '3', '4', '5']:
    print("YAY NEW MODEL", model_version)
    ccs_block = []
    tts_block = []
    ccs_reflect = []
    tts_reflect = []
    ccs_filter = []
    tts_filter = []
    for i in range(10):
        print("repeataf", i)
        if model_version == '1': version = 1
        if model_version == '2': version = 2
        if model_version == '3': version = 3
        if model_version == '4': version = 4
        if model_version == '5': version = 5

        block_confidences = []
        reflect_confidences = []
        filter_confidences = []
        block_times = []
        reflect_times = []
        filter_times = []

        model_path = os.path.join(base_dir, "model_weights", f"sketch_model_weights{model_version}.pth")

        model = DigitToStrokeLSTM(hidden_size=512, num_layers=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        damage_smallest(model, 0.1) # base energy constraint

        # blockage
        for p in [i / 100 for i in range(0, 101, 5)]:
            copy_model = copy.deepcopy(model)
            damage_fas(copy_model, p, 0.0, 0.0)

            correct, time = test_model(copy_model, version)
            block_confidences.append(correct)
            block_times.append(time)

        # reflect
        for p in [i / 100 for i in range(0, 101, 5)]:
            copy_model = copy.deepcopy(model)
            damage_fas(copy_model, 0.0, p, 0.0)

            correct, time = test_model(copy_model, version)
            reflect_confidences.append(correct)
            reflect_times.append(time)

        # filter
        for p in [i / 100 for i in range(0, 101, 5)]:
            copy_model = copy.deepcopy(model)
            damage_fas(copy_model, 0.0, 0.0, p)

            correct, time = test_model(copy_model, version)
            filter_confidences.append(correct)
            filter_times.append(time)

        ccs_block.append(block_confidences)
        tts_block.append(block_times)
        ccs_reflect.append(reflect_confidences)
        tts_reflect.append(reflect_times)
        ccs_filter.append(filter_confidences)
        tts_filter.append(filter_times)

    cs_block.append(ccs_block)
    ts_block.append(tts_block)
    cs_reflect.append(ccs_reflect)
    ts_reflect.append(tts_reflect)
    cs_filter.append(ccs_filter)
    ts_filter.append(tts_filter)

print("CONFIDENCES, BLOCKAGE")
print(cs_block)
print("TIMES, BLOCKAGE")
print(ts_block)
print("CONFIDENCES, REFLECT")
print(cs_reflect)
print("TIMES, REFLECT")
print(ts_reflect)
print("CONFIDENCES, FILTER")
print(cs_filter)
print("TIMES, FILTER")
print(ts_filter)