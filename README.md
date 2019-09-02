# Probabilistic Models of Cognition, in Pyro


## Overview

This is a work-in-progress project to translate [Probabilistic Models of Cognition](https://probmods.org) ([github](https://github.com/probmods/probmods2)) from [WebPPL](http://webppl.org/) to [Pyro](http://pyro.ai/).

I started this project after reading [this post of the pyro.ai forum](https://forum.pyro.ai/t/how-to-begin-learning-probabilistic-programming/519) and feeling similarly overwhelmed by the existing Pyro examples and documentation. This book might help audiences without a background in probabilistic programming start using Pyro, building basic knowledge of these powerful tools so readers can begin to apply them to more complex domains.

A second Pyro project more akin to [DIPPL](http://dippl.org/) examining Pyro's implementation with something like [minipyro](https://github.com/pyro-ppl/pyro/blob/dev/examples/minipyro.py) seems a logical next step after Probabilistic Models of Cognition - though this book would diverge more from its WebPPL counterpart.





## Requirements

```
python >= 3.6

pyro-ppl >= 0.3.3
jupyter
matplotlib
seaborn
tqdm >= 4.31.1
```


## Contributing

Contributions are welcome! See TODO list below for suggestions.



## Chapters
- **Done**: 1, 2 (missing physics sim), 3, 4
- **In Progress**: 5
- **Draft Only**: 6, 7, 8, 9, 10
- **Not Started**: 12, 13, 14, 15, 16






## PyroWebPPL

For this book, we introduce a basic, minimal implementation of WebPPL operators using Pyro as a backend for key PPL functionality. Code for this implementation can be found in the `src` directory for this project.

The PyroWebPPL examples in this book can serve as a stepping stone to learning Pyro. Readers are encouraged to explore `src/webppl.py` and `src/infer.py` to better understand Pyro. Adventurous readers may add breakpoints or print statements to the source code to examine pyro variables used in the relatively simple examples present in this book.




## PyroWebPPL TODO
- [x] MH
- [ ] MH drift kernels
- [ ] Sequential Monte Carlo (SMC)
- [ ] `'optimize'` (with SGD, Adam, ...)
- [ ] Use HashingMarginal and viz.marginals for functions that don't return `torch.Tensor` - necessary for Ch. 10
- [ ] `marginalize`  (ch 6)
- [ ] `posterior_predictive.MAP()`  (ch 6)
- [ ] `correlation`  (ch 6)

- [ ] images for ch 4, 5
- [ ] Embed probmods physics javascript
    - [ ] Physics example (ch 2)
    - [ ] Drawing examples (ch 7)
- [ ] `viz_marginals`  (ch 4)
- [x] `viz_heatmap` (ch 5)
- [ ] `viz_scatter` (ch 5)  - may not be necessary
