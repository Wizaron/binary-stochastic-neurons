import glob, os
import numpy as np

logs = glob.glob(os.path.join('logs', '*.log'))
logs = np.sort(logs)

print '{:40}:{:6}:{}'.format('Model', 'Epoch', 'Accuracy')
for log_file in logs:
    log = np.loadtxt(log_file, dtype='str', delimiter=',')[1:]
    best = log[np.argmax(log[:, 3])]
    best_epoch, best_acc = best[[0, 3]]
    log_name = os.path.splitext(os.path.basename(log_file))[0]
    out = '{:40}:{:6}:{}'.format(log_name, best_epoch, best_acc)
    print out
