import torch
from torch.utils.data import default_collate 

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
            batch = default_collate(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / num_loop}")
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            num_samples_tested=0
            num_loops=0
            for batch in test_batcher:
                
                batch = default_collate(batch)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fn(outputs.logits, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.logits, dim=1)
                correct += (predicted == labels).sum().item()
                num_samples_tested += 128
                num_loops += 1
        accuracy = correct / num_samples_tested
        print(f"Test Loss: {test_loss / num_loops}, Accuracy: {accuracy:.4f}")
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
    