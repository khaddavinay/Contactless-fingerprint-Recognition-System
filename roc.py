import numpy as np
import matplotlib.pyplot as plt




def get_roc_curve(frr, far):
    plt.title('ROC curve')
    x_axis = np.array(far) # False Positive (Accept / Match) Rate
    y_axis = np.array(frr) # False Reject Rate = 1 - TPR
    plt.plot(x_axis, y_axis) # x, y
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('False Reject Rate (FRR)')
    plt.xlabel('False Accept Rate (FAR)')
    plt.grid(True)
    plt.show()