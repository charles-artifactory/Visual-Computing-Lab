import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.model_selection import KFold

# ---------- DATA PREPROCESSING ----------#


def read_pfm(file):
    """Read a PFM file"""
    with open(file, 'rb') as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file: ' + file)

        # Line 2: dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header: ' + file)

        # Line 3: scale factor (negative for little-endian)
        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'  # big-endian

        # Data
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # Flip vertically

        return data, scale


class MiddleburyDataset(Dataset):
    def __init__(self, root_dir, scenes, transform=None, phase='train'):
        """
        Args:
            root_dir (string): Directory with all the scenes
            scenes (list): List of scene names to include
            transform (callable, optional): Optional transform to be applied on samples
            phase (str): 'train' or 'test'
        """
        self.root_dir = root_dir
        self.scenes = scenes
        self.transform = transform
        self.phase = phase
        self.samples = []

        # Collect all valid samples
        for scene in scenes:
            scene_dir = os.path.join(root_dir, scene)
            # Get default left and right images
            left_img_path = os.path.join(scene_dir, 'im0.png')
            right_img_path = os.path.join(scene_dir, 'im1.png')
            left_disp_path = os.path.join(scene_dir, 'disp0.pfm')

            if os.path.exists(left_img_path) and os.path.exists(right_img_path) and os.path.exists(left_disp_path):
                self.samples.append({
                    'left_img': left_img_path,
                    'right_img': right_img_path,
                    'left_disp': left_disp_path,
                    'scene': scene
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        left_img = Image.open(sample['left_img']).convert('RGB')
        right_img = Image.open(sample['right_img']).convert('RGB')

        # Load disparity map
        disparity, _ = read_pfm(sample['left_disp'])
        disparity = disparity.astype(np.float32)

        # Apply transformations
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        # Convert disparity to tensor
        disparity = torch.from_numpy(disparity)

        return {
            'left': left_img,
            'right': right_img,
            'disparity': disparity,
            'scene': sample['scene']
        }


def get_data_loaders(root_dir, fold_idx, k=5, batch_size=1):
    """
    Create train and validation data loaders for k-fold cross-validation

    Args:
        root_dir (str): Path to the dataset
        fold_idx (int): Current fold index (0 to k-1)
        k (int): Number of folds
        batch_size (int): Batch size

    Returns:
        train_loader, val_loader
    """
    # List all scenes
    scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Set up k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Get train and validation indices for the current fold
    train_indices = []
    val_indices = []

    for i, (train_idx, val_idx) in enumerate(kf.split(scenes)):
        if i == fold_idx:
            train_indices = train_idx
            val_indices = val_idx
            break

    train_scenes = [scenes[i] for i in train_indices]
    val_scenes = [scenes[i] for i in val_indices]

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize for memory efficiency
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = MiddleburyDataset(root_dir, train_scenes, transform=transform, phase='train')
    val_dataset = MiddleburyDataset(root_dir, val_scenes, transform=transform, phase='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, val_scenes

# ---------- MODEL DEFINITION ----------#


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.inplanes = 64

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4


class CostVolume(nn.Module):
    def __init__(self, max_disp=192):
        super(CostVolume, self).__init__()
        self.max_disp = max_disp

    def forward(self, left_feature, right_feature):
        B, C, H, W = left_feature.size()
        cost_volume = torch.zeros(B, C*2, self.max_disp//4, H, W, device=left_feature.device)

        for i in range(self.max_disp//4):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_feature[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_feature[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_feature
                cost_volume[:, C:, i, :, :] = right_feature

        cost_volume = cost_volume.contiguous()
        return cost_volume


class CostAggregation(nn.Module):
    def __init__(self, in_channels):
        super(CostAggregation, self).__init__()

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        x = self.conv3d_4(x)
        return x


class DisparityRegression(nn.Module):
    def __init__(self, max_disp):
        super(DisparityRegression, self).__init__()
        self.max_disp = max_disp

    def forward(self, x):
        B, _, D, H, W = x.size()
        x = F.softmax(x.squeeze(1), dim=1)

        # Create disparity values [0, 1, 2, ..., max_disp-1]
        disp_values = torch.arange(0, self.max_disp//4, dtype=torch.float32, device=x.device)
        disp_values = disp_values.view(1, D, 1, 1)

        # Compute expected disparity value
        disparity = torch.sum(x * disp_values, dim=1)

        # Scale to full resolution
        disparity = disparity * 4

        return disparity


class StereoNet(nn.Module):
    def __init__(self, max_disp=192):
        super(StereoNet, self).__init__()
        self.max_disp = max_disp

        # Feature extraction
        self.feature_extraction = FeatureExtraction()

        # Cost volume construction
        self.cost_volume = CostVolume(max_disp)

        # Cost volume aggregation
        self.cost_aggregation = CostAggregation(512*2)

        # Disparity regression
        self.disparity_regression = DisparityRegression(max_disp)

    def forward(self, left, right):
        # Extract features
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)

        # Construct cost volume
        cost_volume = self.cost_volume(left_feature, right_feature)

        # Aggregate cost
        cost = self.cost_aggregation(cost_volume)

        # Regression
        disparity = self.disparity_regression(cost)

        # Upsample to original size
        disparity = F.interpolate(disparity.unsqueeze(1), size=(left.size(2), left.size(3)),
                                  mode='bilinear', align_corners=False).squeeze(1)

        return disparity

# ---------- TRAINING AND EVALUATION ----------#


def train_model(model, train_loader, val_loader, fold_idx, max_disp=192, num_epochs=20):
    """Train the stereo model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function
    criterion = nn.SmoothL1Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_disparities = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        for sample in tqdm(train_loader, desc=f'Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs} (Training)'):
            left = sample['left'].to(device)
            right = sample['right'].to(device)
            target_disp = sample['disparity'].to(device)

            # Forward pass
            optimizer.zero_grad()
            output_disp = model(left, right)

            # Create mask for valid disparities
            mask = (target_disp > 0) & (target_disp < max_disp)

            # Compute loss
            loss = criterion(output_disp[mask], target_disp[mask])

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_disp_errors = []

        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f'Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs} (Validation)'):
                left = sample['left'].to(device)
                right = sample['right'].to(device)
                target_disp = sample['disparity'].to(device)

                # Forward pass
                output_disp = model(left, right)

                # Create mask for valid disparities
                mask = (target_disp > 0) & (target_disp < max_disp)

                # Compute loss
                loss = criterion(output_disp[mask], target_disp[mask])
                val_loss += loss.item()

                # Compute disparity error
                disp_error = torch.abs(output_disp[mask] - target_disp[mask]).mean().item()
                val_disp_errors.append(disp_error)

                # Save predicted disparities for the last epoch
                if epoch == num_epochs - 1:
                    for i in range(output_disp.size(0)):
                        val_disparities.append({
                            'scene': sample['scene'][i],
                            'disparity': output_disp[i].cpu().numpy(),
                            'target': target_disp[i].cpu().numpy()
                        })

        avg_val_loss = val_loss / len(val_loader)
        avg_disp_error = np.mean(val_disp_errors)
        val_losses.append(avg_val_loss)

        print(f'Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Disparity Error: {avg_disp_error:.4f} pixels')

        # Update learning rate
        scheduler.step()

        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_fold{fold_idx+1}.pth')

    return val_disparities, avg_disp_error


def run_cross_validation(data_root, output_dir, k=5, max_disp=192, num_epochs=20):
    """Run k-fold cross-validation"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Results
    all_val_disparities = []
    fold_errors = []
    test_scenes = []

    # Run k-fold cross-validation
    for fold_idx in range(k):
        print(f"Starting fold {fold_idx+1}/{k}")

        # Get data loaders for current fold
        train_loader, val_loader, val_scenes = get_data_loaders(data_root, fold_idx, k=k)
        test_scenes.append(val_scenes)

        # Initialize model
        model = StereoNet(max_disp=max_disp)

        # Train model
        val_disparities, fold_error = train_model(model, train_loader, val_loader, fold_idx, max_disp, num_epochs)
        all_val_disparities.extend(val_disparities)
        fold_errors.append(fold_error)

        print(f"Fold {fold_idx+1} completed. Disparity Error: {fold_error:.4f} pixels")

    # Save cross-validation results
    cv_results = pd.DataFrame({
        'Fold': range(1, k+1),
        'Test_Scenes': [str(scenes) for scenes in test_scenes],
        'Disparity_Error': fold_errors
    })
    cv_results.to_csv(os.path.join(output_dir, 'ex4c_crossvalidation.csv'), index=False)

    # Generate sample disparity maps (first 3 examples)
    for i, disp_data in enumerate(all_val_disparities[:3]):
        scene = disp_data['scene']
        pred_disp = disp_data['disparity']

        # Normalize disparity for visualization
        normalized_disp = pred_disp / max_disp

        # Save as heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(normalized_disp, cmap='plasma')
        plt.colorbar(label='Normalized Disparity')
        plt.title(f'Predicted Disparity Map - Scene: {scene}')
        plt.savefig(os.path.join(output_dir, f'ex4b_{scene}_disparitymap.png'))
        plt.close()

    print("Cross-validation completed. Results saved to", output_dir)

# ---------- VISUALIZATION ----------#


def create_architecture_diagram(output_path="./results/ex4a_architecture.png"):
    """Create a visualization of the PSMNet architecture"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a large empty image
    img_width = 1200
    img_height = 800
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Define box dimensions and positions
    box_width = 160
    box_height = 80

    # Draw architecture components

    # Input images
    draw_box(draw, 100, 100, box_width, box_height, "Left Image\n(Input)", "lightblue")
    draw_box(draw, 100, 250, box_width, box_height, "Right Image\n(Input)", "lightblue")

    # Feature extraction (shared weights)
    draw_box(draw, 350, 100, box_width, box_height, "Feature\nExtraction\n(ResNet)", "lightgreen")
    draw_box(draw, 350, 250, box_width, box_height, "Feature\nExtraction\n(Shared weights)", "lightgreen")

    # Draw arrows connecting inputs to feature extraction
    draw_arrow(draw, 100+box_width, 100+box_height//2, 350, 100+box_height//2)
    draw_arrow(draw, 100+box_width, 250+box_height//2, 350, 250+box_height//2)

    # Cost volume
    draw_box(draw, 600, 175, box_width, box_height, "Cost Volume\nConstruction", "orange")

    # Draw arrows connecting feature extraction to cost volume
    draw_arrow(draw, 350+box_width, 100+box_height//2, 600, 175+box_height//2)
    draw_arrow(draw, 350+box_width, 250+box_height//2, 600, 175+box_height//2)

    # 3D CNN for cost aggregation
    draw_box(draw, 850, 175, box_width, box_height, "3D CNN\nCost Aggregation", "lightsalmon")

    # Draw arrow connecting cost volume to cost aggregation
    draw_arrow(draw, 600+box_width, 175+box_height//2, 850, 175+box_height//2)

    # Disparity regression
    draw_box(draw, 850, 350, box_width, box_height, "Disparity\nRegression", "lightpink")

    # Draw arrow connecting cost aggregation to disparity regression
    draw_arrow(draw, 850+box_width//2, 175+box_height, 850+box_width//2, 350)

    # Output disparity map
    draw_box(draw, 600, 500, box_width, box_height, "Disparity Map\n(Output)", "gold")

    # Draw arrow connecting disparity regression to output
    draw_arrow(draw, 850+box_width//2, 350+box_height, 850+box_width//2, 500+box_height//2)
    draw_arrow(draw, 850+box_width//2, 500+box_height//2, 600+box_width, 500+box_height//2)

    # Title and legends
    font = ImageFont.load_default()
    draw.text((450, 30), "PSMNet Architecture for Stereo Depth Estimation", fill=(0, 0, 0))

    # Save the diagram
    img.save(output_path)
    print(f"Architecture diagram saved to {output_path}")


def draw_box(draw, x, y, width, height, text, color):
    """Draw a box with text inside"""
    # Convert color name to RGB
    color_dict = {
        "lightblue": (173, 216, 230),
        "lightgreen": (144, 238, 144),
        "orange": (255, 165, 0),
        "lightsalmon": (255, 160, 122),
        "lightpink": (255, 182, 193),
        "gold": (255, 215, 0)
    }
    rgb_color = color_dict.get(color, (200, 200, 200))

    # Draw rectangle
    draw.rectangle([x, y, x+width, y+height], fill=rgb_color, outline=(0, 0, 0))

    # Draw text
    lines = text.split('\n')
    font = ImageFont.load_default()
    line_height = height // (len(lines) + 1)

    for i, line in enumerate(lines):
        text_width = len(line) * 6  # Approximate text width
        text_x = x + (width - text_width) // 2
        text_y = y + (i + 1) * line_height
        draw.text((text_x, text_y), line, fill=(0, 0, 0))


def draw_arrow(draw, x1, y1, x2, y2):
    """Draw an arrow from (x1, y1) to (x2, y2)"""
    draw.line([x1, y1, x2, y2], fill=(0, 0, 0), width=2)

    # Draw arrowhead
    angle = np.arctan2(y2 - y1, x2 - x1)
    arrow_length = 10
    arrow_width = 5

    x_end = x2
    y_end = y2

    # Calculate points for arrowhead
    x_arrow1 = x_end - arrow_length * np.cos(angle) + arrow_width * np.sin(angle)
    y_arrow1 = y_end - arrow_length * np.sin(angle) - arrow_width * np.cos(angle)

    x_arrow2 = x_end - arrow_length * np.cos(angle) - arrow_width * np.sin(angle)
    y_arrow2 = y_end - arrow_length * np.sin(angle) + arrow_width * np.cos(angle)

    # Draw arrowhead
    draw.polygon([(x_end, y_end), (x_arrow1, y_arrow1), (x_arrow2, y_arrow2)], fill=(0, 0, 0))

# ---------- MAIN FUNCTION ----------#


INPUT_DIR = '../inputs/MiddleburyDataset/data/'
OUTPUT_DIR = '../results'


def main():
    parser = argparse.ArgumentParser(description='Stereo Depth Estimation with Deep Learning')
    parser.add_argument('--data_root', type=str, default=INPUT_DIR,
                        help='Path to Middlebury dataset')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Path to save results')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--max_disp', type=int, default=290,
                        help='Maximum disparity value')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create architecture diagram
    create_architecture_diagram(os.path.join(args.output_dir, 'ex4a_architecture.png'))

    # Run cross-validation
    run_cross_validation(args.data_root, args.output_dir, k=args.k_folds,
                         max_disp=args.max_disp, num_epochs=args.epochs)

    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
