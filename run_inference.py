import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from functools import partial
from timm.models.vision_transformer import Block, VisionTransformer

# Constants
START_AGE = 0
END_AGE = 77


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RetinalDataset(Dataset):
    def __init__(self, csv_file, data_dir, size=224, is_train=False):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['is_train'] == 0]
        self.data_dir = data_dir
        self.size = size
        self.is_train = is_train

        # Define transforms
        self.transform = transforms.Compose([
            # RemoveEyeBackground(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        image = self.transform(image)

        # Get target (age)
        age = self.df.iloc[idx]['label'] - 15

        # Return image, age, weight, image path, data source
        return image, torch.tensor(age), self.df.iloc[idx]['filename']


class CascadeHead(nn.Module):
    def __init__(self, embedding_dim, input_dim, num_class=8, full_class=77, age_bin=10, age_prompt='hard', depth_ca=1):
        super(CascadeHead, self).__init__()

        self.age_bin = age_bin
        self.age_prompt = age_prompt

        self.norm_base = nn.Identity()
        self.fc_0 = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_class))
        self.fc_1 = nn.Linear(num_class, embedding_dim // 2)
        self.ca_blocks = Block(dim=embedding_dim, num_heads=6)
        self.age_prototypes = torch.nn.Parameter(torch.randn(num_class, embedding_dim))
        self.fc_2 = nn.Sequential(nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim), nn.ReLU())
        self.fc_3 = nn.Linear(embedding_dim, full_class)

    def forward(self, x):
        # coarse prediction with class token
        feat_1 = self.norm_base(x[:, 0])  # B C
        out1 = self.fc_0(feat_1)  # coarse logits  B C --> B num_class

        age_span = torch.arange(START_AGE, END_AGE, step=self.age_bin, dtype=torch.float32).to(x.device)
        out1_ = nn.Softmax(dim=-1)(out1) * age_span  # B * class   coarse predictions
        out1_feat = self.fc_1(out1_)

        # pick class with largest prob
        if self.age_prompt == 'hard':
            cls_index = torch.div(out1_.sum(1), self.age_bin, rounding_mode='floor').long().detach()
            prompt_token = self.age_prototypes[cls_index]  # B * dim
        elif self.age_prompt == 'soft':
            prompt_token = torch.matmul(nn.Softmax(dim=-1)(out1).detach(), self.age_prototypes)
        else:
            raise ValueError

        # x = torch.cat((prompt_token.unsqueeze(1), x), dim=1)
        x = self.ca_blocks(x)  # B dim

        feat_2 = x[:, 0]
        feat_ = torch.concat([out1_feat, feat_2], dim=1)
        feat_ = self.fc_2(feat_)
        out2 = self.fc_3(feat_)  # final prediction

        return out1, out2, feat_, (feat_1, feat_2)


class RegModel(VisionTransformer):
    def __init__(self, img_size=224, num_class=8, age_bin=10, embedding_dim=768, dropout=0.2):
        super().__init__(img_size=img_size, drop_path_rate=dropout, patch_size=16,
                         embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.age_bin = age_bin
        self.head = CascadeHead(self.embed_dim, self.embed_dim, num_class, age_bin=self.age_bin)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, images):
        x = self.forward_features(images)
        out1, out2, feat_, feats = self.head(x)
        # For inference, we don't need the loss term
        return out1, out2, feat_, feats


def run_inference(model_path, csv_file, data_dir, batch_size=16, img_size=224, age_bin=10, num_workers=4):
    """
    Simple function to run inference with a trained model

    Args:
        model_path: Path to the model checkpoint
        csv_file: Path to CSV file with image paths and metadata
        data_dir: Directory containing the images
        batch_size: Batch size for inference
        img_size: Input image size
        age_bin: Age bin size used during training
        num_workers: Number of workers for data loading

    Returns:
        Dictionary containing prediction results
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {csv_file}")
    dataset = RetinalDataset(csv_file=csv_file, data_dir=data_dir, size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Dataset loaded with {len(dataset)} images")

    # Initialize model
    print("Initializing model")
    num_class = len(torch.arange(START_AGE, END_AGE, step=age_bin))
    model = RegModel(img_size=img_size, num_class=num_class, age_bin=age_bin)

    # Load model weights
    print(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif 'model_without_ddp' in checkpoint:
        model.load_state_dict(checkpoint['model_without_ddp'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Prepare result containers
    pred_ages = []
    pred_ages_coarse = []
    coarse_probs = []
    fine_probs = []
    targets = []
    img_files = []

    mae = AverageMeter()
    coarse_mae = AverageMeter()

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for i, (image, target, img_dir) in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")

            image = image.to(device)
            target = target.to(device)

            outputs1, outputs2, feat, feats = model(image)

            # Calculate predictions
            age_span = torch.arange(START_AGE, END_AGE, step=age_bin, dtype=torch.float32).to(device)
            coarse_pred = (nn.Softmax(dim=1)(outputs1) * age_span).sum(1)

            age_span = torch.arange(START_AGE, END_AGE, step=1, dtype=torch.float32).to(device)
            final_pred = (nn.Softmax(dim=1)(outputs2) * age_span).sum(1)

            # Calculate errors
            mean_abs_error_coarse = torch.abs(target.squeeze() - coarse_pred.squeeze()).mean()
            mean_abs_error = torch.abs(target.squeeze() - final_pred.squeeze()).mean()

            # Update metrics
            mae.update(mean_abs_error.item(), image.size(0))
            coarse_mae.update(mean_abs_error_coarse.item(), image.size(0))

            # Store results
            pred_ages_coarse.extend([pred.item() for pred in coarse_pred.cpu()])
            pred_ages.extend([pred.item() for pred in final_pred.cpu()])

            fine_probs.append(nn.Softmax(dim=1)(outputs2).cpu())
            coarse_probs.append(nn.Softmax(dim=1)(outputs1).cpu())

            targets.extend([t.item() for t in target.cpu()])
            img_files.extend(img_dir)

    # Concatenate tensors
    fine_probs = torch.concat(fine_probs, dim=0).numpy()
    coarse_probs = torch.concat(coarse_probs, dim=0).numpy()

    # Create result dictionary
    result = {
        'prediction': pred_ages,
        'coarse_prediction': pred_ages_coarse,
        'targets': targets,
        'mean_abs_error': mae.avg,
        'coarse_mae': coarse_mae.avg,
        'filenames': img_files,
        'fine_probs': fine_probs,
        'coarse_probs': coarse_probs,
    }

    print(f"Inference complete!")
    print(f"Coarse MAE: {coarse_mae.avg:.3f}")
    print(f"Refined MAE: {mae.avg:.3f}")

    return result


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoint-38.pth"
    csv_file = "data_proc/sample.csv"
    data_dir = "/data_path/"

    # Run inference
    results = run_inference(
        model_path=model_path,
        csv_file=csv_file,
        data_dir=data_dir,
        batch_size=128,
        img_size=384,
        age_bin=10
    )

    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'inference_results.pt')
    torch.save(results, result_path)
    print(f"Results saved to {result_path}")

    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(results['prediction']))):
        print(f"File: {os.path.basename(results['filenames'][i])}")
        print(f"True age: {results['targets'][i]:.1f}")
        print(f"Predicted age: {results['prediction'][i]:.1f}")
        print("---")