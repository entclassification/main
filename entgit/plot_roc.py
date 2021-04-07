import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import time

def get_std_auc(ts):
	aucs = []
	for t in ts:
		l = [x / 101 for x in range(101)]
		a = auc(l, t)
		aucs.append(a)


	return round(np.std(aucs), 3)


std_aucs = []
acs = []

tprs = np.load('tprs.npy', allow_pickle=True)
random.seed(0)

tpr = tprs[0]
base_fpr = np.linspace(0, 1, 101)



t = np.array(tpr)
std_auc = get_std_auc(t)
std_aucs.append(std_auc)
mean_t = t.mean(axis=0)

mean_auc = auc([x / 101 for x in range(101)], mean_t)
acs.append(mean_auc)
std = t.std(axis=0)

tpr_upper = np.minimum(mean_t + std, 1)
tpr_lower = mean_t - std

h, = plt.plot(base_fpr, mean_t, 'b', color='#cedd20')
plt.fill_between(base_fpr, tpr_lower, tpr_upper, color='#f2f6c7', alpha=0.3)

handles = [h]
fellows = [(0.020408163265306145, 0.9405940594059405),(0.010204081632653073, 0.9603960396039604), (0.015625, 0.9142857142857143) ]
residents = [(0.025125628140703515, 0.9591836734693877), (0.0, 0.9), (0.010582010582010581, 0.8981481481481481)]
handles.append(plt.scatter([x[0] for x in fellows], [x[1] for x in fellows], marker='o'))
handles.append(plt.scatter([x[0] for x in residents], [x[1] for x in residents], marker='o'))

plt.legend(handles, ["Normal ","Fellow", "Resident"], loc='lower right')


plt.plot([0, 1], [0, 1],'k--')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.savefig('figures/rocs/0.png')
plt.show()
plt.close()

tpr = tprs[1]
base_fpr = np.linspace(0, 1, 101)



t = np.array(tpr)
std_auc = get_std_auc(t)
std_aucs.append(std_auc)
mean_t = t.mean(axis=0)

mean_auc = auc([x / 101 for x in range(101)], mean_t)
acs.append(mean_auc)
std = t.std(axis=0)

tpr_upper = np.minimum(mean_t + std, 1)
tpr_lower = mean_t - std


h, = plt.plot(base_fpr, mean_t, 'b', label="AUC=" + str(mean_auc), color='#0000ff')
plt.legend(loc='lower right')
plt.fill_between(base_fpr, tpr_lower, tpr_upper, color='#d1d1eb', alpha=0.3)

handles = [h]
fellows = [(0.26717557251908397 , 0.8571428571428571 ),(0.11940298507462688 , 0.7916666666666666), (0.22123893805309736 , 0.704225352112676) ]
residents = [(0.20960698689956336 , 0.7647058823529411 ), (0.21739130434782605 , 0.746268656716418), (0.2661290322580645 , 0.6938775510204082)]
handles.append(plt.scatter([x[0] for x in fellows], [x[1] for x in fellows], marker='o'))
handles.append(plt.scatter([x[0] for x in residents], [x[1] for x in residents], marker='o'))

plt.legend(handles, ["IP", "Fellow", "Resident"], loc='lower right')

plt.plot([0, 1], [0, 1],'k--')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.savefig('figures/rocs/1.png')
plt.show()
plt.close()

tpr = tprs[2]

base_fpr = np.linspace(0, 1, 101)



t = np.array(tpr)
std_auc = get_std_auc(t)
std_aucs.append(std_auc)
mean_t = t.mean(axis=0)

mean_auc = auc([x / 101 for x in range(101)], mean_t)
acs.append(mean_auc)
std = t.std(axis=0)

tpr_upper = np.minimum(mean_t + std, 1)
tpr_lower = mean_t - std

h, = plt.plot(base_fpr, mean_t, 'b', color='#b92b23')
plt.legend(loc='lower right')
plt.fill_between(base_fpr, tpr_lower, tpr_upper, color='#ecd2d1', alpha=0.3)

handles = [h]
fellows = [(0.05882352941176472 , 0.5590062111801242 ),(0.11675126903553301 , 0.75), (0.15340909090909094 , 0.5867768595041323 ) ]
residents = [(0.09638554216867468 , 0.6259541984732825  ), (0.14689265536723162 , 0.6 ), (0.14649681528662417 , 0.5357142857142857)]
handles.append(plt.scatter([x[0] for x in fellows], [x[1] for x in fellows], marker='o'))
handles.append(plt.scatter([x[0] for x in residents], [x[1] for x in residents], marker='o'))

plt.legend(handles, ["NP", "Fellow", "Resident"], loc='lower right')

plt.plot([0, 1], [0, 1],'k--')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.savefig('figures/rocs/2.png')
plt.show()
plt.close()

