import torch
import plotly.graph_objects as go

# Code based on plot from https://gist.github.com/gpleiss/0b17bc4bd118b49050056cfcd5446c71

def read_data(path):
    # Load tensors from the file
    data = torch.load(path)
    y_preds = data['y_preds']
    y_true = data['y_true']
    return y_preds, y_true



def save_ece_plot(y_probs_1, y_true_1, y_probs_2, y_true_2, name_1, name_2, model_name, number, color1, color2, n_bins=10):
    """
    y_probs - a torch tensor (size n x num_classes) with the y_probs from the final linear layer
    - NOT the softmaxes
    y_true - a torch tensor (size n) with the labels
    """

    # Get confidences and predictions
    confidences_1, predictions_1 = y_probs_1.max(1)
    accuracies_1 = predictions_1.eq(y_true_1)

    confidences_2, predictions_2 = y_probs_2.max(1)
    accuracies_2 = predictions_2.eq(y_true_2)

    # Define bins
    bins = torch.linspace(0, 1, n_bins + 1, device=y_probs_1.device)
    width = 1.0 / n_bins
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Torch equivalent of bin centers
    bin_indices_1 = [
        (confidences_1 >= bin_lower) & (confidences_1 < bin_upper)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]

    bin_indices_2 = [
        (confidences_2 >= bin_lower) & (confidences_2 < bin_upper)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]

    # Calculate bin_corrects and bin_scores
    bin_corrects_m1 = torch.tensor([
        accuracies_1[bin_index].float().mean() if bin_index.any() else 0.0
        for bin_index in bin_indices_1
    ], device=y_probs_1.device)

    # Calculate bin_corrects and bin_scores
    bin_corrects_m2 = torch.tensor([
        accuracies_2[bin_index].float().mean() if bin_index.any() else 0.0
        for bin_index in bin_indices_2
    ], device=y_probs_2.device)

    # Convert to lists for Plotly (Plotly works with lists or tensors but prefers Python-native lists)
    bin_centers_list = bin_centers.tolist()
    bin_corrects_m1_list = bin_corrects_m1.tolist()
    bin_corrects_m2_list = bin_corrects_m2.tolist()

    # Plot reliability diagram with Plotly
    fig = go.Figure()

    # Add bars for accuracies
    fig.add_trace(go.Bar(
        x=bin_centers_list,
        y=bin_corrects_m1_list,
        name="Accuracy " + name_1,
        marker=dict(color=color1),
        width=width,
    ))

    # Add bars for gaps
    fig.add_trace(go.Bar(
        x=bin_centers_list,
        y=bin_corrects_m2_list,
        name="Accuracy " + name_2,
        marker=dict(color=color2)#"rgba(255, 182, 193, 0.6)"),  # Light pink with transparency
    ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='gray')
    ))

    # Update layout
    fig.update_layout(
        title=f"Reliability Diagram: {model_name}",
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        barmode='overlay',
        legend=dict(x=0.02, y=0.98),
    )

    # Save and show plot
    fig.write_image(f"ECE_Plot_{model_name}_{number}.png")


if __name__ == "__main__":
    # Create evaluation_results directory if it doesn't exist
    for num_models_1 in range(10):
        model_name_1 = "evaluate_wrn_vtst_m=10"
        name_2 = "WRN"
        name_1 = "V-TST"
        #num_models_1 = 1
        path1 = f"ECE_Plot_{model_name_1}_{num_models_1}_In-Distribution.pt"
        y_probs_1, y_true_1 = read_data(path1)
        model_name_2 = "evaluate_wrn_cifar10"
        num_models_2 = 0
        path2 = f"ECE_Plot_{model_name_2}_{num_models_2}_In-Distribution.pt"
        y_probs_2, y_true_2 = read_data(path2)
        plot_name =  "CE Loss" + " In-Distribution"
        save_ece_plot(y_probs_1, y_true_1, y_probs_2, y_true_2, name_1, name_2, plot_name, num_models_1, "blue", "rgba(255, 182, 193, 0.6)")

        
        path1 = f"ECE_Plot_{model_name_1}_{num_models_1}_SHIFT.pt"
        y_probs_1, y_true_1 = read_data(path1)
        path2 = f"ECE_Plot_{model_name_2}_{num_models_2}_SHIFT.pt"
        y_probs_2, y_true_2 = read_data(path2)
        plot_name = "CE Loss " + "SHIFT"
        save_ece_plot(y_probs_1, y_true_1, y_probs_2, y_true_2, name_1, name_2, plot_name, num_models_1, "red", "rgba(255, 182, 193, 0.6)")
        