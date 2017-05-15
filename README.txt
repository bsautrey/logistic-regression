Written by Ben Autrey: https://github.com/bsautrey

---Overview---

Implement logistic regression from Andrew Ng's CS229 course: http://cs229.stanford.edu/notes/cs229-notes1.pdf. Batch gradient ascent is used to learn the parameters, i.e. maximize the likelihood.

alpha - The learning rate.
dampen - Factor by which alpha is dampened on each iteration. Default is no dampening, i.e. dampen = 1.0
tol - The stopping criteria
theta - The parameters to be learned.

---Requirements---

* numpy: https://docs.scipy.org/doc/numpy/user/install.html
* matplotlib: https://matplotlib.org/users/installing.html

---Example---

1) Change dir to where logistic_regression.py is.

2) Run this in a python terminal:

from logistic_regression import LogisticRegression
lr = LogisticRegression()
lr.generate_example()

OR

See the function generate_example() in LWR.py.