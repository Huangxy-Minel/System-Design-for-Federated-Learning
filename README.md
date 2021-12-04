## System Design for Federated Learning
Paper list of federated learning - About system design. Currently, it mainly focus on distributed computing framework design and communication efficiency for Federated Learning (Cross-Silo & Cross-devices). 

Here is Chinese version: [Neth-Lab](https://neth-lab.netlify.app/project/) with a more detailed classification.

**Last update: December, 4th, 2021.**

---

## Catlog

**For some paper, we have provided some Chinese blogs.**

#### [1 Federated Learning Foundation](#1)
- [1.1 Blogs](#1.1)
- [1.2 Survey](#1.2)

#### [2 Research Areas](#2)
- [2.1 Survey](#2.1)
- [2.2 Distributed computing framework](#2.2)
- [2.3 Communication efficiency](#2.3)

#### [3 Opensource Projects](#3)

---

<h2 id="1">1 Federated Learning Foundation</h2>

<h4 id="1.1">1.1 Blogs</h3>

- [Understand the types of federated learning](https://blog.openmined.org/federated-learning-types/). Sep 2020: 
A brief introduction to the terminology and classification of federal learning

<h4 id="1.2">1.2 Survey</h3>

- [Federated Machine Learning: Concept and Applications](https://dl.acm.org/doi/abs/10.1145/3298981). TITS. Qiang Yang. 2019: Chinese blog: [Overview of Federated Learning](https://neth-lab.netlify.app/publication/21-3-2-overview-of-federated-learning/)

---

<h2 id="2">2 Research Areas</h2>

<h4 id="2.1">2.1 Survey</h3>

<h4 id="2.2">2.2 Distributed computing framework</h3>

This section will collect paper from both **Distributed framework for other computation (e.g. DNN and batch-based computation)** and **Distributed System for FL**

<h4 id="2.2.1">2.2.1 System for FL</h4>

- [Towards federated learning at scale: System design](https://mlsys.org/Conferences/2019/doc/2019/193.pdf). 2019. MLSys: A framework scaling for horizontal FL. Chinese blogs: [Survey of Distributed Framework in Federated Learning](https://neth-lab.netlify.app/publication/21-11-26-survey-of-distributed-framework-in-federated-learning/)
- [FedML: A Research Library and Benchmark for Federated Machine Learning](https://arxiv.org/abs/2007.13518). 2020. arXiv: A library and system architecture for FL.

<h4 id="2.2.2">2.2.2 System for traditional computation</h4>

Since currently there is a few research paper about distributed framework for FL, here we provides some important paper about Distributed computation.

- [GAIA: A System for Interactive Analysis on Distributed Graphs Using a High-Level Language](https://www.usenix.org/system/files/nsdi21-qian.pdf). 2021. NSDI: A memory management system for interactive graph computation. Chineses blog: [Survey of Distributed Framework in Federated Learning](https://neth-lab.netlify.app/publication/21-11-26-survey-of-distributed-framework-in-federated-learning/)

<h4 id="2.3">2.3 Communication efficiency</h3>

---

<h2 id="3">3 Opensource Projects</h2>

- [FATE](https://github.com/search?q=federated+learning): Industrial framework for FL. From WeBank. Chinese blog: [Architecture of FATE](https://neth-lab.netlify.app/publication/21-3-12-architecture-of-fate/)
- [PySyft](https://github.com/OpenMined/PySyft)
- [Tensorflow Federated](https://github.com/tensorflow/federated)
- [PyTorch Implementation](https://github.com/shaoxiongji/federated-learning): An implementation based on PyTorch. From [shaoxiongji](https://github.com/shaoxiongji)