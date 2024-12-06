import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import default_collate
from torchdata.nodes.batch import Batcher
from torchdata.nodes.loader import Loader

from torchdata.nodes.samplers.multi_node_weighted_sampler import MultiNodeWeightedSampler


def map_fn_bert(item, max_len, tokenizer):
    """
    Maps a text sample to a BERT-compatible input format.
    Args:
        item (dict): A dictionary containing the text sample and its label.
            - "text" (str): The text sample.
            - "label" (int): The label associated with the text sample.
        max_len (int): The maximum length of the input sequence.
        tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): A pre-trained BERT tokenizer.
    Returns:
        dict: A dictionary containing the input IDs, attention mask, and labels in a BERT-compatible format.
            - "input_ids" (torch.tensor): The input IDs.
            - "attention_mask" (torch.tensor): The attention mask.
            - "labels" (torch.tensor): The labels.
    """
    text = item["text"]
    label = item["label"]
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
        "labels": torch.tensor(label, dtype=torch.long),
    }


def train_bert(model, train_batcher, test_batcher, num_epochs, batch_size):
    """
    Trains a BERT model on a given dataset.
    Args:
        model (torch.nn.Module): The BERT model to be trained.
        train_batcher (torchdata.nodes.loader.Loader): A Loader for the training set.
        test_batcher (torchdata.nodes.loader.Loader): A Loader for the testing set.
        num_epochs (int): The number of epochs to train the model for.
        batch_size (int): The size of each batch.
    Returns:
        torch.nn.Module: The trained BERT model.
    Notes:
        This function trains the BERT model using the Adam optimizer and cross-entropy loss.
        It also evaluates the model's performance on the testing set after each epoch.
    """

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for num_loop, batch in enumerate(train_batcher):
            if num_loop == 128:
                # we just want to process 128 batches
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / num_loop}")
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            num_samples_tested = 0
            num_loops = 0
            for batch in test_batcher:
                if num_loops == 32:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fn(outputs.logits, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.logits, dim=1)
                correct += (predicted == labels).sum().item()
                num_samples_tested += batch_size
                num_loops += 1
        accuracy = correct / num_samples_tested
        print(f"Test Loss : {test_loss / num_loops}, Accuracy: {accuracy : .4f}")
    return model


def get_prediction_bert(review, model, max_len, tokenizer):
    """
    Gets the prediction of a BERT model for a given review.
    Args:
        review (str): The text review to be classified.
        model (torch.nn.Module): The pre-trained BERT model.
        max_len (int): The maximum length of the input sequence.
        tokenizer (transformers.models.bert.tokenization_bert.BertTokenizer): A pre-trained BERT tokenizer.
    Returns:
        None
    Notes:
        This function uses the provided BERT model and tokenizer to classify the given review as either positive or negative.
        It prints the predicted class label ("Positive" or "Negative") to the console.
    """
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].flatten().unsqueeze(0)
    attention_mask = encoding["attention_mask"].flatten().unsqueeze(0)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits)
        if predicted_class == 0:
            print("Negative")
        else:
            print("Positive")


def display_mnist_sample(dataset):
    """
    Displays a random sample from the MNIST dataset.
    Args:
        dataset (dict): A dictionary containing the MNIST dataset, with keys "image" and "label".
    Returns:
        None
    Notes:
        This function uses matplotlib to display the image and prints the image size and label.
    """
    torch.manual_seed(42)
    random_idx = torch.randint(0, len(dataset), size=[1]).item()
    img, label = dataset["image"][random_idx], dataset["label"][random_idx]
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label)
    plt.axis("Off")
    print(f"Image size: {img.shape}")
    print(f"Label: {label}, label size: {label.shape}")


class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=5), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.fc_block = nn.Sequential(nn.Flatten(), nn.Linear(320, 50), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(50, 10))

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = out.view(-1, 320)
        out = self.fc_block(out)
        return out


def train_mnist_model(model, num_epochs, train_loader, test_loader, loss_fn, optimizer):
    """
    Trains a PyTorch model on the MNIST dataset.
    Args:
        model (nn.Module): The PyTorch model to be trained.
        num_epochs (int): The number of epochs to train the model for.
        train_loader (nodes.Loader): A DataLoader object containing the training data.
        test_loader (nodes.Loader): A DataLoader object containing the testing data.
        loss_fn (nn.Module): The loss function to be used during training.
        optimizer (Optimizer): The optimizer to be used during training.
    Returns:
        tuple: A tuple containing the trained model, a list of losses at each epoch, and a list of accuracies at each epoch.
    Notes:
        This function trains the model using the provided training data and evaluates its performance on the provided testing data.
    """
    loss_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        num_loops = 0
        total_loss = 0
        for batch in train_loader:
            images = batch["image"]
            labels = batch["label"]
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_loops += 1
        total_loss /= num_loops
        loss_list.append(total_loss)

        # Test the model
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for batch in test_loader:
                images = batch["image"]
                labels = batch["label"]
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            accuracy_list.append(acc)
            if epoch == num_epochs - 1:
                print(f"Test Accuracy: {acc} %")
    return model, loss_list, accuracy_list


def csv_reader(filename):
    """
    Reads a CSV file and returns its contents as a list of lists.
    Args:
        filename (str): The path to the CSV file to be read.
    Returns:
        list: A list of lists, where each inner list represents a row in the CSV file.
    Notes:
        This function assumes that the CSV file has a header row that should be skipped.
        It also assumes that each row in the CSV file contains only numeric values.
    """
    with open(filename) as csvfile:
        data = []
        for line_number, line in enumerate(csvfile):
            if line_number == 0:
                continue
            row = [float(x) for x in line.strip().split(",")]
            data.append(row)
    return data


class MDS_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 128)  # input layer (1) -> hidden layer (128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 1)  # hidden layer (128) -> output layer (1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_mds_model(datasets, weights, batch_size=512):
    """
    Train a model using the provided multiple datasets and weights.
    Args:
        model (nn.Module): The MDS model to be trained. If None, a new MDS_Net model will be created.
        datasets (Dict[str, BaseNode]): A dictionary mapping dataset names to their corresponding BaseNode objects.
        weights (Dict[str, float]): A dictionary mapping dataset names to their corresponding weightages for sampling in the training batch.
        batch_size (int, optional): The batch size for training. Defaults to 512.
    Returns:
        nn.Module: The trained MDS model.
    """
    model = MDS_Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    node = MultiNodeWeightedSampler(datasets, weights)
    train_batcher = Loader(Batcher(node, batch_size, drop_last=True))
    # Train the model
    for epoch in range(100):
        for batch in train_batcher:
            batch = default_collate(batch)
            x = batch["x"]
            y = batch["y"]
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    return model
