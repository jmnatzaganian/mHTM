import numpy as np
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from mHTM.datasets.loader import load_mnist, MNISTCV

def main():
	seed = 123456789
	np.random.seed(seed)
	ntrain, ntest = 800, 200
	(tr_x, tr_y), (te_x, te_y) = load_mnist()
	x, y = np.vstack((tr_x, te_x)), np.hstack((tr_y, te_y))
	cv = MNISTCV(tr_y, te_y, ntrain, ntest, 1, seed)

	for tr, te in cv:
		clf = OneVsRestClassifier(LinearSVC(random_state=seed), -1)
		clf.fit(x[tr], y[tr])
		print clf.score(x[te], y[te])
		
		clf = LinearSVC(random_state=seed)
		clf.fit(x[tr], y[tr])
		print clf.score(x[te], y[te])
		
		clf = OneVsOneClassifier(LinearSVC(random_state=seed), -1)
		clf.fit(x[tr], y[tr])
		print clf.score(x[te], y[te])

if __name__ == '__main__':
	main()
