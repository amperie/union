import torch
from sklearn.metrics import log_loss, confusion_matrix, classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass


def evaluate_model(val_df, model, classifier, batch_size=32, threshold=0.5):
    """
    Evaluate the model on validation data and return various metrics

    Args:
        val_df: DataFrame with 'prompt' and 'target' columns
        model: Loaded fine-tuned SentenceTransformer model
        classifier: Loaded classifier head
        batch_size: Batch size for validation
        threshold: Classification threshold for binary predictions

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    model.eval()
    classifier.eval()

    device = next(classifier.parameters()).device
    all_predictions = []
    true_labels = val_df['target'].values

    # Get predictions in batches
    print("Running validation predictions...")
    for i in tqdm(range(0, len(val_df), batch_size)):
        batch_texts = val_df['prompt'].iloc[i:i + batch_size].tolist()
        with torch.no_grad():
            # Get embeddings
            embeddings = model.encode(batch_texts, convert_to_tensor=True)
            embeddings = embeddings.to(device)
            # Get predictions
            outputs = classifier(embeddings)
            predictions = outputs.squeeze().cpu().numpy()
            all_predictions.extend(
                predictions if predictions.ndim > 0 else [predictions])

    # Convert to numpy array
    all_predictions = np.array(all_predictions)

    # Calculate metrics
    val_log_loss = log_loss(true_labels, all_predictions)
    binary_predictions = (all_predictions >= threshold).astype(int)
    conf_matrix = confusion_matrix(true_labels, binary_predictions)
    class_report = classification_report(
        true_labels, binary_predictions, output_dict=True)

    # Create a results dictionary
    results = {
        'log_loss': val_log_loss,
        'accuracy': class_report['accuracy'],
        'precision': class_report['1']['precision'],
        'recall': class_report['1']['recall'],
        'f1_score': class_report['1']['f1-score'],
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'true_labels': true_labels
    }

    # Print results
    print("\nValidation Results:")
    print(f"Log Loss: {val_log_loss:.4f}")
    print(f"Accuracy: {class_report['accuracy']:.4f}")
    print(f"Precision: {class_report['1']['precision']:.4f}")
    print(f"Recall: {class_report['1']['recall']:.4f}")
    print(f"F1 Score: {class_report['1']['f1-score']:.4f}")

    return results


def plot_results(results):
    """
    Create visualizations of the validation results
    """
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot confusion matrix
    sns.heatmap(results['confusion_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Blues',
                ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    # Plot prediction distribution
    sns.histplot(data=pd.DataFrame({
        'Predictions': results['predictions'],
        'Labels': results['true_labels']
    }), x='Predictions', hue='Labels', bins=30, ax=ax2)
    ax2.set_title('Prediction Distribution')
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Count')

    plt.tight_layout()
    plt.show()

# results = evaluate_model(df_val, model, classifier)

# Plot results
# plot_results(results)
