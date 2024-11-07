import os
from flytekit import workflow, task, Deck
from dataclasses import dataclass
from flytekit import current_context
from flytekit import ImageSpec
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from flytekit.types.structured import StructuredDataset
import urllib.request
from pathlib import Path
import pandas as pd
import torch
from torch import nn
import sentence_transformers
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss
from ft_tasks import evaluate_model, plot_results
import tempfile

# Config variables
enable_cache = False


# Custom Classes
@dataclass
class TrainingConfig:
    learning_rate: float = 2.0e-65
    epochs: int = 2
    batch_size: int = 16


@dataclass
class TrainingResults:
    config: TrainingConfig
    average_loss: float
    train_logloss: float
    model: FlyteDirectory
    classifier: FlyteDirectory
    classifier_dim: int


@dataclass
class TotalDataset:
    train_data: StructuredDataset
    val_data: StructuredDataset


# Custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Custom classifier head
class ClassifierHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


image = ImageSpec(
    requirements=Path(__file__).parent / "requirements.txt",
)


@task(
    cache_version="1.0",
    cache=enable_cache,
    container_image=image,
)
def download_dataset() -> FlyteFile:
    url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
    file_path = "data.tsv"

    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0')
    req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8')
    req.add_header('Accept-Language', 'en-US,en;q=0.5')

    r = urllib.request.urlopen(req).read().decode('utf-8')
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(r)
    return FlyteFile(file_path)


@task(
    cache_version="1.0",
    cache=enable_cache,
    container_image=image,
)
def process_dataset(data: FlyteFile, nrows: int = 1000) -> TotalDataset:
    df = pd.read_csv(data, sep='\t', nrows=nrows*2)
    df["target"] = df["is_duplicate"]
    df['prompt'] =\
        df.apply(
            lambda row:
            f"Do these two questions mean the exact same thing?\n"
            f"Question 1: {row.question1}\n"
            f"Question 2: {row.question2}", axis=1
            )
    total_data = df[['prompt', 'target']]
    retVal = TotalDataset(
        StructuredDataset(dataframe=total_data[:nrows]),
        StructuredDataset(dataframe=total_data[nrows:]))
    return retVal


@task(
    cache_version="1.0",
    cache=enable_cache,
    container_image=image,
)
def get_model(model_name: str) -> FlyteDirectory:
    from huggingface_hub import login, snapshot_download

    ctx = current_context()
    # working_dir = Path(ctx.working_directory)
    working_dir = Path(tempfile.gettempdir())
    # working_dir = Path("models")
    model_cache_dir = working_dir / "model_cache"

    login(token="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")
    snapshot_download(model_name, local_dir=model_cache_dir)
    return model_cache_dir


@task(
    cache_version="1.0",
    cache=enable_cache,
    container_image=image,
    enable_deck=True,
)
def train_model(
    model_folder: FlyteDirectory, data: TotalDataset,
    train_cfg: TrainingConfig)\
        -> TrainingResults:

    df = data.train_data.open(pd.DataFrame).all()
    model_folder.download()
    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    train_dir = working_dir / "trained_models"

    model = sentence_transformers.SentenceTransformer(model_folder.path)

    # Create classifier head
    classifier = ClassifierHead(model.get_sentence_embedding_dimension())

    # Prepre dataset
    dataset = TextClassificationDataset(
        df['prompt'].tolist(),
        torch.tensor(df['target'].values, dtype=torch.float32))

    dataloader = DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': train_cfg.learning_rate},
        {'params': classifier.parameters(), 'lr': train_cfg.learning_rate * 10}
    ])

    # Loss function
    criterion = nn.BCELoss()

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    classifier = classifier.to(device)

    # Training loop
    for epoch in range(train_cfg.epochs):
        model.train()
        classifier.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_texts, batch_labels in dataloader:
            # Get embeddings
            embeddings = model.encode(batch_texts, convert_to_tensor=True)
            embeddings = embeddings.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = classifier(embeddings)
            loss = criterion(outputs.squeeze(), batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Store predictions and labels for log loss calculation
            all_preds.extend(outputs.squeeze().detach().cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        # Calculate epoch loss and log loss
        avg_loss = total_loss / len(dataloader)
        epoch_log_loss = log_loss(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{train_cfg.epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Log Loss: {epoch_log_loss:.4f}\n")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(train_dir / "model", exist_ok=True)
    os.makedirs(train_dir / "classifier", exist_ok=True)

    model.save(os.path.join(train_dir, "model"))
    model_fd = FlyteDirectory(os.path.join(train_dir, "model"))

    torch.save(
        classifier.state_dict(),
        os.path.join(train_dir, "classifier", 'classifier_head.pt'))
    classifier_fd = FlyteDirectory(os.path.join(train_dir, "classifier"))

    retVal = TrainingResults(
        config=train_cfg,
        average_loss=avg_loss,
        train_logloss=epoch_log_loss,
        model=model_fd,
        classifier=classifier_fd,
        classifier_dim=model.get_sentence_embedding_dimension()
    )

    return retVal


@task(
    container_image=image,
    enable_deck=True,
)
def model_evaluation(
    results: TrainingResults, total_data: TotalDataset)\
        -> TrainingResults:
    df_val = total_data.val_data.open(pd.DataFrame).all()
    model = sentence_transformers.SentenceTransformer(results.model.path)
    classifier = ClassifierHead(results.classifier_dim)
    classifier.load_state_dict(
        torch.load(
            os.path.join(results.classifier.path, "classifier_head.pt")))
    return evaluate_model(df_val, model, classifier)


@workflow
def finetuning_wf(nrows: int = 1000) -> TrainingResults:
    data = download_dataset()
    total_data = process_dataset(data=data, nrows=nrows)
    model = get_model(model_name="sentence-transformers/all-mpnet-base-v2")

    train_cfg = TrainingConfig(
        epochs=1,
        batch_size=16,
        learning_rate=2e-5
    )
    train_results = train_model(
        model_folder=model,
        data=total_data,
        train_cfg=train_cfg)

    eval_results = model_evaluation(
        results=train_results,
        total_data=total_data
    )

    return train_results


if __name__ == "__main__":
    finetuning_wf()
    pass
