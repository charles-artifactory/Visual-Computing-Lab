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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

INPUT_DIR = '../inputs/MiddleburyDataset/data/'
OUTPUT_DIR = '../results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def read_calib_file(calib_file):
    """Read calibration file and extract parameters."""
    with open(calib_file, 'r') as f:
        lines = f.readlines()

    params = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        key_value = line.split('=')
        if len(key_value) != 2:
            continue

        key, value = key_value

        if key == 'cam0' or key == 'cam1':
            # Example: cam0=[1733.74 0 792.27; 0 1733.74 541.89; 0 0 1]
            value = value.replace('[', '').replace(']', '')
            matrix_values = []
            for row in value.split(';'):
                if row:
                    row_values = [float(v) for v in row.split()]
                    matrix_values.extend(row_values)
            params[key] = matrix_values
        else:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value

    if 'width' not in params:
        params['width'] = 1920
    if 'height' not in params:
        params['height'] = 1080
    if 'ndisp' not in params:
        params['ndisp'] = 192

    return params


def read_pfm(file):
    """Read PFM file."""
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file: ' + file)

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header: ' + file)

        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'
        else:
            endian = '>'

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


class MiddleburyDataset(Dataset):
    def __init__(self, root_dir, scenes, transform=None, phase='train', target_size=(256, 512)):
        """
        Args:
            root_dir (string): Directory containing all scenes
            scenes (list): List of scene names to include
            transform (callable, optional): Optional transform to be applied on a sample
            phase (str): 'train' or 'test'
            target_size (tuple): Target size (height, width) for resizing
        """
        self.root_dir = root_dir
        self.scenes = scenes
        self.transform = transform
        self.phase = phase
        self.target_size = target_size
        self.samples = []

        for scene in scenes:
            scene_dir = os.path.join(root_dir, scene)

            calib_file = os.path.join(scene_dir, 'calib.txt')

            left_img_path = os.path.join(scene_dir, 'im0.png')
            right_img_path = os.path.join(scene_dir, 'im1.png')
            left_disp_path = os.path.join(scene_dir, 'disp0.pfm')

            if os.path.exists(left_img_path) and os.path.exists(right_img_path) and os.path.exists(left_disp_path) and os.path.exists(calib_file):
                calib_params = read_calib_file(calib_file)

                self.samples.append({
                    'left_img': left_img_path,
                    'right_img': right_img_path,
                    'left_disp': left_disp_path,
                    'scene': scene,
                    'calib_params': calib_params
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        calib_params = sample['calib_params']
        original_width = int(calib_params['width'])
        original_height = int(calib_params['height'])
        max_disp = int(calib_params['ndisp'])

        left_img = Image.open(sample['left_img']).convert('RGB')
        right_img = Image.open(sample['right_img']).convert('RGB')

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        disparity, _ = read_pfm(sample['left_disp'])
        disparity = disparity.astype(np.float32)

        disparity_pil = Image.fromarray(disparity)
        disparity_pil = disparity_pil.resize(self.target_size, Image.BILINEAR)
        disparity = np.array(disparity_pil)

        width_scale = self.target_size[1] / original_width
        disparity = disparity * width_scale

        disparity = torch.from_numpy(disparity)

        return {
            'left': left_img,
            'right': right_img,
            'disparity': disparity,
            'scene': sample['scene'],
            'calib_params': calib_params,
            'max_disp': max_disp
        }


def get_data_loaders(root_dir, fold_idx, k=5, batch_size=1):
    """
    Create train and validation data loaders for k-fold cross-validation

    Args:
        root_dir (str): Dataset path
        fold_idx (int): Current fold index (0 to k-1)
        k (int): Number of folds
        batch_size (int): Batch size

    Returns:
        train_loader, val_loader, val_scenes, max_disp
    """
    scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    scenes.sort()  # Ensure consistent order

    print(f"Found {len(scenes)} scenes: {scenes[:5]}...")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    train_indices = []
    val_indices = []

    for i, (train_idx, val_idx) in enumerate(kf.split(scenes)):
        if i == fold_idx:
            train_indices = train_idx
            val_indices = val_idx
            break

    train_scenes = [scenes[i] for i in train_indices]
    val_scenes = [scenes[i] for i in val_indices]

    print(f"Fold {fold_idx+1}: Train scenes: {len(train_scenes)}, Validation scenes: {len(val_scenes)}")

    target_size = (256, 512)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MiddleburyDataset(root_dir, train_scenes, transform=transform,
                                      phase='train', target_size=target_size)
    val_dataset = MiddleburyDataset(root_dir, val_scenes, transform=transform, phase='test', target_size=target_size)

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        calib_params = sample['calib_params']
        print(f"Scene '{sample['scene']}' calibration info:")
        print(f"  Original size: {int(calib_params['width'])}x{int(calib_params['height'])}")
        print(f"  Max disparity: {int(calib_params['ndisp'])}")
        print(f"  Disparity range: {calib_params.get('vmin', 'N/A')} - {calib_params.get('vmax', 'N/A')}")
        print(f"  Baseline: {calib_params.get('baseline', 'N/A')} mm")

        max_disp = int(calib_params['ndisp'])
    else:
        max_disp = 192

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, val_scenes, max_disp


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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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

        disp_values = torch.arange(0, self.max_disp//4, dtype=torch.float32, device=x.device)
        disp_values = disp_values.view(1, D, 1, 1)

        disparity = torch.sum(x * disp_values, dim=1)

        disparity = disparity * 4

        return disparity


class StereoNet(nn.Module):
    def __init__(self, max_disp=192):
        super(StereoNet, self).__init__()
        self.max_disp = max_disp

        self.feature_extraction = FeatureExtraction()

        self.cost_volume = CostVolume(max_disp)

        self.cost_aggregation = CostAggregation(512*2)

        self.disparity_regression = DisparityRegression(max_disp)

    def forward(self, left, right):
        left_feature = self.feature_extraction(left)
        right_feature = self.feature_extraction(right)

        cost_volume = self.cost_volume(left_feature, right_feature)

        cost = self.cost_aggregation(cost_volume)

        disparity = self.disparity_regression(cost)

        disparity = F.interpolate(disparity.unsqueeze(1), size=(left.size(2), left.size(3)),
                                  mode='bilinear', align_corners=False).squeeze(1)

        return disparity


def visualize_stereo_results(left_img, right_img, raw_disparity, scene_name, output_dir):
    """
    Create a visualization with left image, right image, and disparity maps (raw and normalized).

    Args:
        left_img: Left image
        right_img: Right image
        raw_disparity: Raw disparity map
        scene_name: Scene name
        output_dir: Output directory
    """
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(left_img)
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(right_img)
    plt.title('Right Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    im = plt.imshow(raw_disparity, cmap='plasma')
    plt.title('Predicted Disparity')
    plt.axis('off')
    plt.colorbar(im, label='Disparity (pixels)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ex4b_{scene_name}_stereo_result.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(raw_disparity, cmap='plasma')
    plt.title('Original Disparity')
    plt.axis('off')
    plt.colorbar(im1, label='Disparity (pixels)')

    norm_disp = raw_disparity / np.max(raw_disparity)
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(norm_disp, cmap='plasma')
    plt.title('Normalized Disparity')
    plt.axis('off')
    plt.colorbar(im2, label='Normalized Disparity')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ex4b_{scene_name}_disparitymap.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved disparity visualizations for scene {scene_name}")


def train_model(model, train_loader, val_loader, fold_idx, max_disp=192, num_epochs=20):
    """Train the stereo model"""
    model.to(device)

    criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_disparities = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Replace tqdm with print statements
        print(f'Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs} (Training)')
        train_batch_count = len(train_loader)

        for i, sample in enumerate(train_loader):
            # Print progress periodically
            if (i + 1) % max(1, train_batch_count // 10) == 0:
                print(f'  Training batch {i+1}/{train_batch_count} ({(i+1)/train_batch_count*100:.1f}%)')

            left = sample['left'].to(device)
            right = sample['right'].to(device)
            target_disp = sample['disparity'].to(device)

            optimizer.zero_grad()
            output_disp = model(left, right)

            if output_disp.shape[1:] != target_disp.shape[1:]:
                target_disp = F.interpolate(target_disp.unsqueeze(1),
                                            size=output_disp.shape[1:],
                                            mode='bilinear',
                                            align_corners=False).squeeze(1)

            batch_max_disp = max_disp
            mask = (target_disp > 0) & (target_disp < batch_max_disp)

            loss = criterion(output_disp[mask], target_disp[mask])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_disp_errors = []

        # Replace tqdm with print statements for validation
        print(f'Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs} (Validation)')
        val_batch_count = len(val_loader)

        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                # Print progress periodically
                if (i + 1) % max(1, val_batch_count // 5) == 0:
                    print(f'  Validation batch {i+1}/{val_batch_count} ({(i+1)/val_batch_count*100:.1f}%)')

                left = sample['left'].to(device)
                right = sample['right'].to(device)
                target_disp = sample['disparity'].to(device)

                # Forward pass
                output_disp = model(left, right)

                # Ensure shape match - resize target disparity if needed
                if output_disp.shape[1:] != target_disp.shape[1:]:
                    target_disp = F.interpolate(target_disp.unsqueeze(1),
                                                size=output_disp.shape[1:],
                                                mode='bilinear',
                                                align_corners=False).squeeze(1)

                # Create mask for valid disparity
                batch_max_disp = max_disp
                mask = (target_disp > 0) & (target_disp < batch_max_disp)

                # Compute loss
                loss = criterion(output_disp[mask], target_disp[mask])
                val_loss += loss.item()

                # Compute disparity error
                disp_error = torch.abs(output_disp[mask] - target_disp[mask]).mean().item()
                val_disp_errors.append(disp_error)

                # Save predicted disparity for the last epoch
                if epoch == num_epochs - 1:
                    for i in range(output_disp.size(0)):
                        calib_params = {k: v for k, v in sample['calib_params'].items()}
                        val_disparities.append({
                            'scene': sample['scene'][i],
                            'disparity': output_disp[i].cpu().numpy(),
                            'target': target_disp[i].cpu().numpy(),
                            'calib_params': calib_params
                        })

        avg_val_loss = val_loss / len(val_loader)
        avg_disp_error = np.mean(val_disp_errors)
        val_losses.append(avg_val_loss)

        print(f'Fold {fold_idx+1}, Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Disparity Error: {avg_disp_error:.4f} pixels')

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_model_fold{fold_idx+1}.pth'))

    return val_disparities, avg_disp_error


def save_cv_results(folds, test_scenes, disparity_errors, output_path):
    """Save cross-validation results to CSV without using pandas"""
    with open(output_path, 'w') as f:
        # Write header
        f.write('Fold,Test_Scenes,Disparity_Error\n')

        # Write data rows
        for i in range(len(folds)):
            f.write(f'{folds[i]},{test_scenes[i]},{disparity_errors[i]}\n')


def read_cv_results(file_path):
    """Read cross-validation results from CSV without using pandas"""
    if not os.path.exists(file_path):
        return None

    results = {
        'Fold': [],
        'Test_Scenes': [],
        'Disparity_Error': []
    }

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            first_comma_idx = line.find(',')
            if first_comma_idx == -1:
                continue

            last_comma_idx = line.rfind(',')
            if last_comma_idx == first_comma_idx:
                continue

            fold = line[:first_comma_idx]
            test_scenes = line[first_comma_idx+1:last_comma_idx]
            disp_error = line[last_comma_idx+1:]

            try:
                results['Fold'].append(int(fold))
                results['Test_Scenes'].append(test_scenes)
                results['Disparity_Error'].append(float(disp_error))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse line: {line}")
                continue

    return results


def run_cross_validation(data_root, output_dir, k=5, num_epochs=20):
    """Run k-fold cross-validation"""
    os.makedirs(output_dir, exist_ok=True)

    all_val_disparities = []
    fold_errors = []
    test_scenes = []

    for fold_idx in range(k):
        print(f"Starting fold {fold_idx+1}/{k}")

        train_loader, val_loader, val_scenes, max_disp = get_data_loaders(data_root, fold_idx, k=k)
        test_scenes.append(str(val_scenes))

        print(f"Using max disparity: {max_disp}")

        model = StereoNet(max_disp=max_disp)

        val_disparities, fold_error = train_model(model, train_loader, val_loader, fold_idx, max_disp, num_epochs)
        all_val_disparities.extend(val_disparities)
        fold_errors.append(fold_error)

        print(f"Fold {fold_idx+1} completed. Disparity Error: {fold_error:.4f} pixels")

    # Save cross-validation results without pandas
    save_cv_results(
        folds=range(1, k+1),
        test_scenes=test_scenes,
        disparity_errors=fold_errors,
        output_path=os.path.join(output_dir, 'ex4c_crossvalidation.csv')
    )

    for i, disp_data in enumerate(all_val_disparities[:3]):
        scene = disp_data['scene']
        pred_disp = disp_data['disparity']
        calib_params = disp_data['calib_params']

        baseline = float(calib_params.get('baseline', 100.0))
        focal_length = float(calib_params['cam0'][0])
        doffs = float(calib_params.get('doffs', 0.0))

        scene_dir = os.path.join(data_root, scene)
        left_img = Image.open(os.path.join(scene_dir, 'im0.png'))
        right_img = Image.open(os.path.join(scene_dir, 'im1.png'))

        visualize_stereo_results(left_img, right_img, pred_disp, scene, output_dir)

        # Compute depth: Z = (baseline * f) / (d + doffs)
        valid_disparity = np.where(pred_disp > 0.1, pred_disp, 0.1)
        depth = (baseline * focal_length) / (valid_disparity + doffs)

        depth_cap = np.percentile(depth, 95)
        depth_vis = np.clip(depth, 0, depth_cap)
        norm_depth = depth_vis / np.max(depth_vis)

        plt.figure(figsize=(10, 6))
        plt.imshow(norm_depth, cmap='viridis')
        plt.colorbar(label='Normalized Depth (mm)')
        plt.title(f'Predicted Depth Map - Scene: {scene}')
        plt.savefig(os.path.join(output_dir, f'ex4b_{scene}_depthmap.png'))
        plt.close()

    print("Cross-validation completed. Results saved to", output_dir)


def draw_box(draw, x, y, width, height, text, color):
    """Draw a box with text"""
    color_dict = {
        "lightblue": (173, 216, 230),
        "lightgreen": (144, 238, 144),
        "orange": (255, 165, 0),
        "lightsalmon": (255, 160, 122),
        "lightpink": (255, 182, 193),
        "gold": (255, 215, 0)
    }
    rgb_color = color_dict.get(color, (200, 200, 200))

    draw.rectangle([x, y, x+width, y+height], fill=rgb_color, outline=(0, 0, 0))

    lines = text.split('\n')
    font = ImageFont.load_default()
    line_height = height // (len(lines) + 1)

    for i, line in enumerate(lines):
        text_width = len(line) * 6
        text_x = x + (width - text_width) // 2
        text_y = y + (i + 1) * line_height
        draw.text((text_x, text_y), line, fill=(0, 0, 0))


def draw_arrow(draw, x1, y1, x2, y2):
    """Draw an arrow from (x1, y1) to (x2, y2)"""
    draw.line([x1, y1, x2, y2], fill=(0, 0, 0), width=2)

    angle = np.arctan2(y2 - y1, x2 - x1)
    arrow_length = 10
    arrow_width = 5

    x_end = x2
    y_end = y2

    x_arrow1 = x_end - arrow_length * np.cos(angle) + arrow_width * np.sin(angle)
    y_arrow1 = y_end - arrow_length * np.sin(angle) - arrow_width * np.cos(angle)

    x_arrow2 = x_end - arrow_length * np.cos(angle) - arrow_width * np.sin(angle)
    y_arrow2 = y_end - arrow_length * np.sin(angle) + arrow_width * np.cos(angle)

    draw.polygon([(x_end, y_end), (x_arrow1, y_arrow1), (x_arrow2, y_arrow2)], fill=(0, 0, 0))


def create_architecture_diagram(output_path=os.path.join(OUTPUT_DIR, 'ex4a_architecture.png')):
    """Create a visualization of the PSMNet architecture"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a large blank image
    img_width = 1200
    img_height = 800
    img = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Define box size and position
    box_width = 160
    box_height = 80

    # Draw architecture components

    # Input images
    draw_box(draw, 100, 100, box_width, box_height, "Left Image\n(Input)", "lightblue")
    draw_box(draw, 100, 250, box_width, box_height, "Right Image\n(Input)", "lightblue")

    # Feature extraction (shared weights)
    draw_box(draw, 350, 100, box_width, box_height, "Feature\nExtraction\n(ResNet)", "lightgreen")
    draw_box(draw, 350, 250, box_width, box_height, "Feature\nExtraction\n(Shared weights)", "lightgreen")

    # Draw arrows from input to feature extraction
    draw_arrow(draw, 100+box_width, 100+box_height//2, 350, 100+box_height//2)
    draw_arrow(draw, 100+box_width, 250+box_height//2, 350, 250+box_height//2)

    # Cost volume
    draw_box(draw, 600, 175, box_width, box_height, "Cost Volume\nConstruction", "orange")

    # Draw arrows from feature extraction to cost volume
    draw_arrow(draw, 350+box_width, 100+box_height//2, 600, 175+box_height//2)
    draw_arrow(draw, 350+box_width, 250+box_height//2, 600, 175+box_height//2)

    # 3D CNN for cost aggregation
    draw_box(draw, 850, 175, box_width, box_height, "3D CNN\nCost Aggregation", "lightsalmon")

    # Draw arrow from cost volume to cost aggregation
    draw_arrow(draw, 600+box_width, 175+box_height//2, 850, 175+box_height//2)

    # Disparity regression
    draw_box(draw, 850, 350, box_width, box_height, "Disparity\nRegression", "lightpink")

    # Draw arrow from cost aggregation to disparity regression
    draw_arrow(draw, 850+box_width//2, 175+box_height, 850+box_width//2, 350)

    # Output disparity map
    draw_box(draw, 600, 500, box_width, box_height, "Disparity Map\n(Output)", "gold")

    # Draw arrows from disparity regression to output
    draw_arrow(draw, 850+box_width//2, 350+box_height, 850+box_width//2, 500+box_height//2)
    draw_arrow(draw, 850+box_width//2, 500+box_height//2, 600+box_width, 500+box_height//2)

    # Title and legend
    font = ImageFont.load_default()
    draw.text((450, 30), "PSMNet Architecture for Stereo Depth Estimation", fill=(0, 0, 0))

    # Save diagram
    img.save(output_path)
    print(f"Architecture diagram saved to {output_path}")


def visualize_results():
    """Visualize training and validation results and save to files without displaying."""
    cv_path = os.path.join(OUTPUT_DIR, 'ex4c_crossvalidation.csv')
    if os.path.exists(cv_path):
        cv_results = read_cv_results(cv_path)

        if cv_results:
            print("Cross-validation results:")
            print(f"{'Fold':<10} {'Disparity_Error':<20}")
            print("-" * 30)

            for i in range(len(cv_results['Fold'])):
                print(f"{cv_results['Fold'][i]:<10} {cv_results['Disparity_Error'][i]:<20.4f}")

            mean_error = sum(cv_results['Disparity_Error']) / len(cv_results['Disparity_Error'])
            print(f"\nMean disparity error: {mean_error:.4f} pixels")

            plt.figure(figsize=(10, 6))
            plt.bar(cv_results['Fold'], cv_results['Disparity_Error'])
            plt.xlabel('Fold')
            plt.ylabel('Disparity Error (pixels)')
            plt.title('Cross-Validation Results')
            plt.xticks(cv_results['Fold'])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(OUTPUT_DIR, 'disparity_errors.png'), dpi=200, bbox_inches='tight')
            plt.close()
    else:
        print("Cross-validation results file not found. Please run training first.")

    disp_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('ex4b_') and f.endswith('_disparitymap.png')]
    if disp_files:
        print(f"\nFound {len(disp_files)} predicted disparity maps:")

        if len(disp_files) > 0:
            plt.figure(figsize=(15, 5*min(len(disp_files), 3)))

            for i, f in enumerate(disp_files[:3]):
                img = plt.imread(os.path.join(OUTPUT_DIR, f))
                plt.subplot(min(len(disp_files), 3), 1, i+1)
                plt.imshow(img)
                plt.title(f.replace('ex4b_', '').replace('_disparitymap.png', ''))
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'combined_disparity_maps.png'), dpi=200, bbox_inches='tight')
            plt.close()
    else:
        print("No predicted disparity maps found. Please run training first.")


k = 5
num_epochs = 100

create_architecture_diagram()

scenes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
if scenes:
    test_scene = scenes[0]
    calib_file = os.path.join(INPUT_DIR, test_scene, 'calib.txt')
    if os.path.exists(calib_file):
        print(f"\nReading sample calibration file: {calib_file}")
        calib_params = read_calib_file(calib_file)
        print("Calibration parameters:")
        for key, value in calib_params.items():
            print(f"  {key}: {value}")

run_cross_validation(INPUT_DIR, OUTPUT_DIR, k=k, num_epochs=num_epochs)
visualize_results()
