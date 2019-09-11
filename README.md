# Probabilistic Models of Cognition, in Pyro


## Overview

Work-in-progress project to translate [Probabilistic Models of Cognition](https://probmods.org) ([github](https://github.com/probmods/probmods2)) from [WebPPL](http://webppl.org/) to [Pyro](http://pyro.ai/). This collection of notebooks might help audiences without much background in probabilistic programming start using Pyro, building basic knowledge of these powerful tools so readers can begin to apply them to more complex domains. 

A second Pyro project more akin to [DIPPL](http://dippl.org/) examining Pyro's implementation with something like [minipyro](https://github.com/pyro-ppl/pyro/blob/dev/examples/minipyro.py) seems a logical next step after this - though that book would diverge more from its WebPPL counterpart.





## Requirements

```
python >= 3.6

pyro-ppl >= 0.4.0
jupyter
matplotlib
seaborn
tqdm >= 4.31.1
```




## PyroWebPPL

We introduce a simple, minimal implementation of WebPPL operators using Pyro as a backend for key PPL functionality. Code for this implementation can be found in the `pyro_webppl` directory for this project.

The PyroWebPPL examples can serve as a stepping stone to learning Pyro. All readers are encouraged to explore `pyro_webppl/webppl.py` and `pyro_webppl/infer.py` to better understand Pyro. Adventurous readers may add breakpoints or print statements to the source code to examine pyro variables used (e.g. traces in MH inference) in the relatively trivial examples present in this book.





## Contributing

Contributions are welcome! See chapters and TODOs below for ideas.



# Progress


## Chapters
- **Done**: 1, 2, 3, 4, 5  (physics sims missing in 2, 3 - see below)
- **Draft Only**: 6, 7, 8, 9, 10
- **Not Started**: 12, 13, 14, 15, 16


## Example

Nothing here yet.



## PyroWebPPL TODO

### `pyro_webppl/infer.py`
- [x] MH
- [ ] MH drift kernels
- [ ] Sequential Monte Carlo (SMC)
- [ ] `'optimize'` (with SGD, Adam, ...)
- [ ] Use `HashingMarginal` and `viz_marginals` for functions that don't return `torch.Tensor` - needed for Ch. 10
- [ ] reflectance/illumination example in ch 5 is slow

### `pyro_webppl/webppl.py`
- [ ] `marginalize`  (ch 6)
- [ ] `posterior_predictive.MAP()`  (ch 6)
- [ ] `correlation`  (ch 6)
- [ ] `viz_marginals`  (ch 4)
- [ ] `viz_density`  (ch 5)
- [x] `viz_heatmap` (ch 5)
- [ ] `viz_scatter` (ch 5)  - maybe unnecessary

- [ ] Physics example (ch 2)
- [ ] Drawing examples (ch 7)


## Javascript <-> Python

The physics examples are a bit tricky to implement natively in Python. I spent some time exploring `pymunk` and `pygame` but had trouble getting examples to run in jupyter notebooks, essential for this project.

Notebooks natively support `%%javascript` cells and you can pass data between python and js ([example](https://www.stefaanlippens.net/jupyter-custom-d3-visualization.html)), so a particularly convenient solution would be to load the [javascript libraries used in probmods](https://github.com/probmods/probmods2/tree/master/assets/js) directly and use these in the appropriate chapters.


