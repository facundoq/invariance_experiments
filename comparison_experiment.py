import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from experiment_base import Experiment, RunConfig, DeviceConfig
from typing import List

class ModelComparisonExperiment(Experiment):
    """
    Subclass that implements the logic for comparing multiple invariant/equivariant models.
    """
    def __init__(self, experiment_name: str, configs: List[RunConfig], device_config: DeviceConfig = None):
        super().__init__(experiment_name, device_config=device_config)
        self.configs = configs
        self.device = self.get_device()

    def _get_dataloader(self, dataset_name, batch_size, subset_size=500, test_subset_size=100):
        if dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == "cifar100":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == "tinyimagenet":
            from tinyimagenet import TinyImageNet
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            train_dataset = TinyImageNet(root='./data', split='train', download=True, transform=transform)
            test_dataset = TinyImageNet(root='./data', split='val', download=True, transform=transform)
        else: # default cifar10
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        subset_size = min(subset_size, len(train_dataset))
        test_subset_size = min(test_subset_size, len(test_dataset))

        train_subset = Subset(train_dataset, range(0, subset_size))
        test_subset = Subset(test_dataset, range(0, test_subset_size))
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def train_one_epoch(self, model, loader, optimizer, epoch):
        model.train()
        total_loss, correct = 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
        return total_loss / len(loader), correct / len(loader.dataset)

    def evaluate(self, model, loader, transform=None):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                if transform: data = transform(data)
                output = model(data)
                correct += (output.argmax(1) == target).sum().item()
        return correct / len(loader.dataset)

    def run(self, subset_size=500, test_subset_size=100):
        for exp_config in self.configs:
            print(f"\n--- Running: {exp_config.run_name} on {exp_config.dataset_name} ---")
            
            # 1. Instantiate Model via its build() and Config
            model = exp_config.build_fn(exp_config.model_config).to(self.device)
            
            # 2. Get Data
            train_loader, test_loader = self._get_dataloader(
                exp_config.dataset_name,
                exp_config.batch_size, 
                subset_size=subset_size, 
                test_subset_size=test_subset_size
            )
            
            optimizer = optim.Adam(model.parameters(), lr=exp_config.lr)

            # 3. MLflow Tracking
            with self.start_run(exp_config.run_name):
                self.log_config(exp_config.model_config)
                self.log_config({
                    "epochs": exp_config.epochs,
                    "lr": exp_config.lr,
                    "batch_size": exp_config.batch_size
                })

                for epoch in range(1, exp_config.epochs + 1):
                    loss, acc = self.train_one_epoch(model, train_loader, optimizer, epoch)
                    self.log_metrics({"train_loss": loss, "train_acc": acc}, step=epoch)

                # Standard Test
                test_acc = self.evaluate(model, test_loader)
                self.log_metrics({"test_acc": test_acc})
                print(f"Standard Test Acc: {test_acc:.4f}")

                # Rotated Test
                rot_transform = transforms.RandomRotation(90)
                rot_acc = self.evaluate(model, test_loader, transform=rot_transform)
                self.log_metrics({"rotated_test_acc": rot_acc})
                print(f"Rotated Test Acc: {rot_acc:.4f}")

                self.log_model(model)
