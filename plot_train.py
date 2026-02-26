import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = r"C:\Users\bmahsa\Downloads\In Vivo Image_Wholemount (CART Samples)\training_valid_under_1\metrics.csv"

epochs = []
global_steps = []
train_loss_avgs = []
train_sup_avgs = []
train_cons_avgs = []
lams = []
valid_frac_avgs = []

with open(csv_filename, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader, None)

    for row in reader:
        if not row or len(row) < 7:
            continue

        epochs.append(int(row[0]))
        global_steps.append(int(row[1]))
        train_loss_avgs.append(float(row[2]))
        train_sup_avgs.append(float(row[3]))
        train_cons_avgs.append(float(row[4]))
        # lams.append(float(row[5]))
        valid_frac_avgs.append(float(row[6]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# -------- Left: Loss curves --------
ax[0].plot(epochs, train_loss_avgs, label="Total")
ax[0].plot(epochs, train_sup_avgs, label="Supervised")
ax[0].plot(epochs, train_cons_avgs, label="Consistency")

ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

n = len(epochs)
step = max(1, n // 10)
ax[0].set_xticks(epochs[::step])


# -------- Right: Valid fraction --------
ax[1].plot(epochs, valid_frac_avgs, label="Valid Fraction")

ax[1].set_title("Valid Fraction per Epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Valid Fraction")
ax[1].legend()

ax[1].set_xticks(epochs[::step])


fig.tight_layout()
plt.show()