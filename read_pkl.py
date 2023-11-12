import pickle
import matplotlib.pyplot as plt
# Replace 'yourfile.pkl' with the path to the actual .pkl file you want to read
file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Nov2023_SingleTraining/resultsJustFeedback/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.001/label_smoothing=0.00//training.pkl'
# file_path = '/nfs/hpc/share/sail_on3/TestsForPaper/Nov2023/resultsGANAug/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.001/label_smoothing=0.00/validation.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Use pickle to load the file
    data = pickle.load(file)


# import ipdb; ipdb.set_trace()
# Assuming the dictionaries contain numeric indices and float values, let's plot them.
loss_curve = data['training_loss_curve']
accuracy_curve = data['training_accuracy_curve']

# Sort the dictionaries by key to ensure the plots are in the correct order
sorted_loss_curve = dict(sorted(loss_curve.items()))
sorted_accuracy_curve = dict(sorted(accuracy_curve.items()))

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(list(sorted_loss_curve.keys()), list(sorted_loss_curve.values()), marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot the accuracy curve
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(list(sorted_accuracy_curve.keys()), list(sorted_accuracy_curve.values()), marker='o', color='orange')
plt.title('Training Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()

# Save the figures
plt.savefig('/nfs/hpc/share/sail_on3/TestsForPaper/Nov2023_SingleTraining/resultsJustFeedback/logs/resizepad=224/none/normalized/end-to-end-trainer/lr=0.001/label_smoothing=0.00/training_curves.png', dpi=300)  # Save as a PNG file with 300 dpi
# If you want to save them separately, you can create and save individual plots

# Separate figures
# Save loss curve
plt.figure()
plt.plot(list(sorted_loss_curve.keys()), list(sorted_loss_curve.values()), marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_curve.png', dpi=300)  # Save as a PNG file with 300 dpi

# Save accuracy curve
plt.figure()
plt.plot(list(sorted_accuracy_curve.keys()), list(sorted_accuracy_curve.values()), marker='o', color='orange')
plt.title('Training Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy_curve.png', dpi=300) 