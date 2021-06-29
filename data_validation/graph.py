
resultOld = validatorOld.evaluate_validator()
print("New", resultNew['precision'], resultNew['recall'], resultNew['f1'])
print("Old", resultOld['precision'], resultOld['recall'], resultOld['f1'])

import matplotlib.pyplot as plt
import numpy as np


labels = ['Precision', 'Recall', 'F1']
New_validator = [resultNew['precision'], resultNew['recall'], resultNew['f1']]
Old_validator = [resultOld['precision'], resultOld['recall'], resultOld['f1']]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, New_validator, width, label='New')
rects2 = ax.bar(x + width/2, Old_validator, width, label='Old')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores comparision between new and old')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


fig.tight_layout()

plt.show()