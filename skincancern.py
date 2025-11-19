#To get OrderedDict because the torch.nn.Sequential model uses that to 
#  allow me to name the layers, which I found helpful in setting the 
#   learning rate.
import collections
import torch, torchvision
from torchvision.models import resnet50, ResNet50_Weights

# If you have a Mac laptop, this will make it use the GPU power.
torch.set_default_device(torch.device("mps"))
# Set random seed so that results are reproducible.
torch.manual_seed(37)

# Docs say that DEFAULT gives the most accurate version.
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights = weights)

def get_dataloaders(
            *, batch_size: int, train_proportion: float
    ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Set up dataloaders for train and validation sets."""
    path_to_images = "defungi"
    transform = torchvision.transforms.Compose([
            weights.transforms(),
        ])
    full_dataset = torchvision.datasets.ImageFolder(
            path_to_images, 
            transform=transform,
        )
    generator = torch.Generator(device="mps").manual_seed(37)
    train_set, valid_set = torch.utils.data.random_split(
            full_dataset,
            [train_proportion, 1-train_proportion],
            generator = generator,
    )
    train = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    valid = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    return train, valid

train, valid = get_dataloaders(batch_size=32, train_proportion=0.8)
# Number of batches
print(f"Train batches = {len(train)}")
print(f"Validation batches = {len(valid)}")

# Use model.children to see layers with their names
# Trying to delete last layer, but instead just make it not do anything.
model.fc = torch.nn.Identity()
# To see inheritance stack, use type(model).__mro__
# To see all known sublcasses that inherit from this class, use 
#       type(model).__subclasses__()
# model is of type torch.nn.Module, so override forward to include add'nl layer

# Set up model with named sections so we can optimize separately
model = torch.nn.Sequential(collections.OrderedDict([
        ('resnet', model),
        ('final', torch.nn.Linear(in_features=2048, out_features=5)),
        ('softmax', torch.nn.Softmax(dim=1)),
]))

lr = .0001
# Set learning rate for the two parts
param_groups = [
        {'params': model.resnet.parameters(), 'requires_grad': False},
        {'params': model.final.parameters(), 'lr': lr},
]
optimizer = torch.optim.Adam(param_groups)

model.train()
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(1):
    batch_losses = []
    batch_accuracies = []
    data_load_time = 0
    forward_time = 0
    backprop_time = 0
    print(f"=== Epoch {i+1} ===")
    for (image_batch, label_batch) in train:
        preds = model(image_batch)
        loss = loss_fn(preds, label_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(float(loss))
        cur_loss = sum(batch_losses)/len(batch_losses)
        batch_accuracies.append(int(sum(preds.argmax(1) == label_batch))/len(label_batch))
        cur_acc = sum(batch_accuracies)/len(batch_accuracies)
        print("Train:", end="\t\t")
        print(f"Batch: {len(batch_losses)}", end="\t")
        print(f"Loss: {round(cur_loss, 4)}", end="\t")
        print(f"Accuracy: {round(cur_acc, 4)}")
    print()
    batch_losses = []
    batch_accuracies = []
    for (image_batch, label_batch) in valid:
        with torch.no_grad():
            preds = model(image_batch)
            loss = loss_fn(preds, label_batch)
            batch_losses.append(float(loss))
            cur_loss = sum(batch_losses)/len(batch_losses)
            batch_accuracies.append(int(sum(preds.argmax(1) == label_batch))/len(label_batch))
            cur_acc = sum(batch_accuracies)/len(batch_accuracies)
            print("Validation:", end="\t")
            print(f"Batch: {len(batch_losses)}", end="\t")
            print(f"Loss: {round(cur_loss, 4)}", end="\t")
            print(f"Accuracy: {round(cur_acc, 4)}")
    print()
    # If you want the learning rate to decay over time
    # lr = 0*0.95
    # param_groups = [
    #         # {'params': model.preprocess.parameters(), 'lr': 0},
    #         {'params': model.resnet.parameters(), 'requires_grad': False},
    #         {'params': model.final.parameters(), 'lr': lr},
    # ]
    # optimizer = torch.optim.Adam(param_groups)