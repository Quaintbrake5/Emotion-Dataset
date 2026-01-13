import matplotlib.pyplot as plt

# Replace these lists with your actual values from training logs
epochs = list(range(1, 42))

train_acc = [0.3163,0.3516,0.3797,0.4186,0.4672,0.5078,0.5351,0.5610,0.5717,
             0.5887,0.6069,0.6157,0.6294,0.6391,0.6564,0.6669,0.6786,
             0.7207,0.7279,0.7386,0.7431,0.7456,0.7465,0.7509,0.7619,0.7599,
             0.7641,0.7719,0.7725,0.7773,0.7769,0.7780,0.7867,0.7873,0.7933,0.8003,
             0.7969,0.8034,0.8043,0.8079,0.8025]

val_acc = [0.1662,0.3375,0.2691,0.2485,0.3007,0.2963,0.4382,0.1897,0.3949,
           0.2029,0.3684,0.5566,0.1838,0.4875,0.3191,0.4566,0.3074,
           0.6654,0.6787,0.6963,0.6875,0.6618,0.6728,0.6831,0.7103,0.6919,
           0.7169,0.7221,0.7265,0.6846,0.7228,0.6559,0.7235,0.7338,0.6846,0.7221,
           0.7382,0.7316,0.7301,0.7294,0.7301]

train_loss = [1.6158,1.5476,1.5003,1.4383,1.3491,1.2834,1.2458,1.1910,1.1506,
              1.1218,1.0867,1.0587,1.0334,1.0021,0.9757,0.9554,0.9260,
              0.8288,0.7938,0.7795,0.7577,0.7555,0.7414,0.7309,0.7050,0.7072,
              0.7061,0.6854,0.6833,0.6675,0.6694,0.6543,0.6402,0.6373,0.6243,0.6161,
              0.6140,0.6016,0.5973,0.5981,0.6045]

val_loss = [6.3035,1.5897,1.8855,2.3427,1.6845,3.1334,1.3796,3.8913,1.5613,
            4.0533,1.9410,1.2652,6.4824,1.3822,4.0630,1.5221,7.0390,
            0.9738,0.9766,0.9134,0.9548,1.0192,0.9914,0.9690,0.8915,0.9361,
            0.9016,0.8594,0.8670,0.9728,0.8342,1.0666,0.8675,0.8773,1.0301,0.8963,
            0.8517,0.8525,0.8485,0.8595,0.8649]

# Plot
plt.figure(figsize=(14,6))

# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Val Accuracy', marker='o')
plt.title('Train vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.axvspan(1,17, color='red', alpha=0.1, label='Chaotic Phase')
plt.axvspan(18,36, color='yellow', alpha=0.1, label='Stabilization Phase')
plt.axvspan(37,41, color='green', alpha=0.1, label='Fine-Tuning Phase')

# Loss
plt.subplot(1,2,2)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.title('Train vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.axvspan(1,17, color='red', alpha=0.1)
plt.axvspan(18,36, color='yellow', alpha=0.1)
plt.axvspan(37,41, color='green', alpha=0.1)

plt.tight_layout()
plt.show()
plt.savefig('training_validation_plots.png')