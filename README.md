# Base Classes for Models
Base Class and utilities for machine learning with Pytorch models.

This repository is inspired by the design that Dr. Christopher Potts used in his 
extremely strong course on natural language understanding (I highly recommend it!),
cs224u, offered as a <a href='https://online.stanford.edu/courses/xcs224u-natural-language-understanding'>professional program in xcs224u</a>. His 
design for a base class for torch models was the initial inspiration here. Notably
it conforms with the sci-kit learn project, and so all the parameter search 
utilities there are usable with models properly subclasses from the base. I heavily
leveraged Dr. Potts' code for inspiration, however I've made extensive changes to
suit my own preferences. His original liscense is the Apache 2.0 liscense for this
code, and <a href='https://github.com/cgpotts/cs224u'>can be viewed at his course repo here</a>.