

import pickle
import matplotlib.pyplot as plt

# Load and process training data
# training_file_path = '.log-balanced-normalization-corrected/train-heads/hack/resizepad=224/randaugment/normalized/end-to-end-trainer/lr=0.005/label_smoothing=0.00/training.pkl'
# training_file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/EWC_102/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.0001/label_smoothing=0.00/training.pkl'
training_file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-102-ClassBalancing/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.001/label_smoothing=0.00/training.pkl'
# training_file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/Oracle_100budget_EWC/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.0001/label_smoothing=0.00/training.pkl'
with open(training_file_path, 'rb') as file:
    training_data = pickle.load(file)

sorted_loss_curve = dict(sorted(training_data['training_loss_curve'].items()))
sorted_accuracy_curve = dict(sorted(training_data['training_accuracy_curve'].items()))

# Load and process validation data
# validation_file_path = '.log-balanced-normalization-corrected/train-heads/hack/resizepad=224/randaugment/normalized/end-to-end-trainer/lr=0.005/label_smoothing=0.00/validation.pkl'
# validation_file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/EWC_102/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.0001/label_smoothing=0.00/validation.pkl'
validation_file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-102-ClassBalancing/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.001/label_smoothing=0.00/validation.pkl'
# validation_file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/Oracle_100budget_EWC/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.0001/label_smoothing=0.00/validation.pkl'

with open(validation_file_path, 'rb') as file:
    validation_data = pickle.load(file)

x = list(validation_data.keys())
y = list(validation_data.values())

# Create a figure with 3 subplots
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 3, 1)
plt.plot(list(sorted_loss_curve.keys()), list(sorted_loss_curve.values()), marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot training accuracy
plt.subplot(1, 3, 2)
plt.plot(list(sorted_accuracy_curve.keys()), list(sorted_accuracy_curve.values()), marker='o', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot validation data
plt.subplot(1, 3, 3)
plt.plot(x, y, marker='o', color='blue')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('combined_plotsNewEnd.png', dpi=300)
