<p align="center">
<img src="static/img/animation.gif" alt="Randomized Truncation" width="400"/>
</p>

<a href="https://github.com/angusjlowe/rtrunc">rtrunc</a> is a small package to compute randomized truncations of
pure quantum states, optimized with respect to the following distance measures
(as of the current version): (i) trace distance and (ii) robustness.

<p align="center">
<img src="static/img/density_matrix_plots.png" alt="density matrix plots" width="600"/>
</p>

## Installation

First:

```console
git clone https://github.com/angusjlowe/rtrunc.git
```
Then within the cloned directory:

```console
pip install .
```

Or, if you anticipate making changes to the code while using it,
run

```console
pip install -e .
```

Check the (minimal) tests pass by running

```console
python -m pytest tests/
```

If this works, you should be good to go!

## Example usage

Most of the example code in this section is contained in examples folder. The code
to produce the grayscale density matrix plots above is contained in
examples/density_matrix_plots.py.

### Computing the trace distance
First, we'll import the necessary packages and create a random quantum
state which we'd like to approximate. Here, n is the dimension and k is the
number of non-zero entries in the approximation. We only need the
trace distance module within the rtrunc package for this example, so we
just call the module itself rtrunc.

```python
import numpy as np
from rtrunc import td_optimizer as rtrunc

n, k = 150, 50 

# normal, random
v = np.random.normal(0,1,n)

# normalize
v = v/np.linalg.norm(v)
v = -np.sort(-np.abs(v))
```

Next, we can make a new trace distance optimizer object. The optimal
fidelity is immediately available upon instantiation.

```python
# instantiate trace distance optimization
tdo = rtrunc.TDOptimizer(k, v)
fid = tdo.fid
print("Optimal pure trace distance is {:.3f}".format(np.sqrt(1-fid**2)))
```
In this case, we get

```console
Optimal pure trace distance is 0.370
```

To compute the trace distance and unit vector m corresponding to
the optimal measurement, we need to run:

```python
m,td = tdo.getOptimalTDMeas()
print("Optimal mixed trace distance is {:.3f}".format(td))
```
which gives us

```console
Optimal mixed trace distance is 0.268
```
which is more accurate, as expected.


### Sampling

Continuing with the above example, we can now sample pure states
from the ensemble corresponding to the optimal density matrix \sigma^*.
Let's say we are interested in the observable
serving as the optimal measurement for discriminating v from its
deterministic truncation. Call this m_det.

```python
# store deterministic truncation
vtrunc = np.concatenate((v[:k],np.zeros(n-k)))
vtrunc = vtrunc/np.linalg.norm(vtrunc)

# compute worst-case measurement
(evals, evecs) = np.linalg.eig(np.outer(v,v)-np.outer(vtrunc, vtrunc))
max_index = np.argmax(evals)
m_det = evecs[:,max_index]
m_det = m_det/np.linalg.norm(m_det)

# store expectation values
true_expec = np.abs(np.dot(m_det, v))**2
det_trunc_expec = np.abs(np.dot(m_det, vtrunc))**2
```

We can sample sparse pure states using randomized truncation to estimate
this observable, and then store the data.

```python
# sample to compute expec. val.
n_samples = 1000
expec_samples = []
rob_expec_samples = []
for j in range(n_samples):
    if (j+1) % 20 == 0:
        print("Sample {}".format(j+1))
    phi = tdo.sampleOptimalTDState()
    expec = np.abs(np.dot(phi, m_det))**2
    expec_samples.append(expec)
    phi = ro.sampleOptimalRobState()
    expec = np.abs(np.dot(phi, m_det))**2
    rob_expec_samples.append(expec)

# store data
means = np.array(list(map(lambda x: np.mean(expec_samples[:x]), [*range(1,n_samples+1)])))
stds = np.array(list(map(lambda x: np.std(expec_samples[:x], ddof=1)/np.sqrt(x), [*range(2,n_samples+1)])))
stds = np.concatenate(([0], stds))
xs = np.arange(n_samples)+1
ys = np.abs(means - true_expec)

rob_means = np.array(list(map(lambda x: np.mean(rob_expec_samples[:x]), [*range(1,n_samples+1)])))
rob_stds = np.array(list(map(lambda x: np.std(rob_expec_samples[:x], ddof=1)/np.sqrt(x), [*range(2,n_samples+1)])))
rob_stds = np.concatenate(([0], rob_stds))
rob_ys = np.abs(rob_means - true_expec)
```
We can do then plot the difference in expectation compared to that from the
deterministic truncation. 

```python
# plot error in estimate from rtrunc against no. of samples
plt.plot(xs, ys, '-', label='rtrunc (trace distance)', color='blue')
plt.fill_between(xs, ys-stds, ys+stds, color='blue', alpha=0.2)

plt.plot(xs, rob_ys, '-', label='rtrunc (robustness)', color='orange')
plt.fill_between(xs, rob_ys-rob_stds, rob_ys+rob_stds, color='orange', alpha=0.2)

# plot error in estimate from deterministic trunc.
plt.plot(xs, np.ones(n_samples)*np.abs(det_trunc_expec - true_expec), '--', color='black', label='closed-form dtrunc. expec. diff.')

# show plot
plt.xlabel('no. of samples')
plt.legend()
title1 = "Estimating worst-case observable for $|v_{1:k}\\rangle$."
title2 = " n={}, k={}.".format(n, k)
plt.title(title1 + title2)
plt.show()
```

This gives:

![trunc_estimate_img](static/img/sampling_plot_1.png)


### Getting the optimal state

The method getOptimalTDState() returns the density matrix corresponding
to the optimal randomized truncation. This method will be costly for large
dimensions since we are generating an nxn matrix, but is useful for checking
the output at smaller dimensions. With n=20 and k=12, we can use it to
visualize the optimal approximation. This leads to the image at the
beginning of this README.

### Making that snazzy animation in this README file

Use the code in src/random_subset_text_gif.py.  You will need to obtain the file DejaVuSans.ttf, which you can get [here](https://dejavu-fonts.github.io/).

 The command that produced this animation is:

```python random_subset_text_gif.py --text "Randomized\nTruncation" --font_size 48 --width 400 --height 200  --fps 6 --seconds 6 --transparent --draw_spaces --line_height 40 --fg "#000000" --out "animation.gif"```

![Always Be Choosing randomly](static/img/abc.png)
