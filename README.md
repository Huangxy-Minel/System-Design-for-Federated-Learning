## System Design for Federated Learning
A paper list of federated learning - About system design. Currently, it mainly focuses on **distributed computing framework design, communication efficiency and asynchronous computing** for **Federated Learning** (Cross-Silo & Cross-devices). 

Here is Chinese version: [Neth-Lab](https://neth-lab.netlify.app/project/) with a more detailed classification.

**Last update: December, 21th, 2021.**

---

## Catlog

**For some paper, we have provided some Chinese blogs.**

#### [1 Federated Learning Foundation](#1)
- [1.1 Blogs](#1.1)
- [1.2 Survey](#1.2)
- [1.3 Algorithms](#1.3)

#### [2 Research Areas](#2)
- [2.1 Survey](#2.1)
- [2.2 Distributed computing framework](#2.2)
- [2.3 Communication efficiency](#2.3)
- [2.4 Asynchronous computing](#2.4)

#### [3 Opensource Projects](#3)

---

<h2 id="1">1 Federated Learning Foundation</h2>

<h4 id="1.1">1.1 Blogs</h4>

- [Understand the types of federated learning](https://blog.openmined.org/federated-learning-types/). Sep 2020
    - A brief introduction to the terminology and classification of federal learning

<h4 id="1.2">1.2 Survey</h4>

- [Federated Machine Learning: Concept and Applications](https://dl.acm.org/doi/abs/10.1145/3298981). TITS. Qiang Yang. 2019
    - Chinese blog: [Overview of Federated Learning. Section 1](https://neth-lab.netlify.app/publication/21-3-2-overview-of-federated-learning/)

<h4 id="1.3">1.3 Algorithms</h4>

- [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://dl.acm.org/doi/abs/10.1145/3133956.3133982). 2017. CCS
    - Horizontal logistic regression algorithm
- [Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption](https://arxiv.org/abs/1711.10677). 2017. arXiv
    - Vertical logistic regression algorithm.
    - Chinese blog: [Machine Learning & Federated Learning. Section 5](https://neth-lab.netlify.app/publication/21-09-01-machine-learning-and-federated-learning/#section2)
- [SecureBoost: A Lossless Federated Learning Framework](https://ieeexplore.ieee.org/abstract/document/9440789). 2021. IEEE Intelligent Systems
    - Vertical secure boosting algorithm.
    - Chinese blog: [Machine Learning & Federated Learning. Section 6](https://neth-lab.netlify.app/publication/21-09-01-machine-learning-and-federated-learning/#section2)

---

<h2 id="2">2 Research Areas</h2>

<h4 id="2.1">2.1 Survey</h4>

- [A Survey on Distributed Machine Learning](https://dl.acm.org/doi/abs/10.1145/3377454). 2020. ACM Computing Surveys
    - About system challenges for distributed machine learning
    - Chinese blog: [Survey of Distributed Framework in Federated Learning. Section 1](https://neth-lab.netlify.app/publication/21-11-26-survey-of-distributed-framework-in-federated-learning/)

<h4 id="2.2">2.2 Distributed computing framework</h4>

This section will collect paper from both **Distributed framework for other computation (e.g. DNN and batch-based computation)** and **Distributed System for FL**

<h4 id="2.2.1">2.2.1 Framework for FL</h4>

- [Towards federated learning at scale: System design](https://mlsys.org/Conferences/2019/doc/2019/193.pdf). 2019. MLSys
    - A framework for scaling horizontal FL. 
    - Chinese blogs: [Survey of Distributed Framework in Federated Learning. Section 3](https://neth-lab.netlify.app/publication/21-11-26-survey-of-distributed-framework-in-federated-learning/)
- [FedML: A Research Library and Benchmark for Federated Machine Learning](https://arxiv.org/abs/2007.13518). 2020. arXiv
    - A library and system architecture for FL.

<h4 id="2.2.2">2.2.2 Framework for Machine Learning</h4>

Since currently there is a few research paper about distributed framework for FL, here we provide related work focus on Machine Learning Framework for reference.

- [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/conference/osdi21/presentation/wang). 2021. OSDI
- [P3: Distributed Deep Graph Learning at Scale](https://www.usenix.org/conference/osdi21/presentation/gandhi). 2021. OSDI
- [Scaling Large Production Clusters with Partitioned Synchronization](https://www.usenix.org/conference/atc21/presentation/feng-yihui). 2021. ATC
    - A distributed resource scheduler architecture. Use partition synchronization method to reduce the impact of contention on high-quality resources and staleness of local states, which causes high scheduling latency.
    - Chinese blog: [Survey of Distributed Framework in Federated Learning. Section 4](https://neth-lab.netlify.app/publication/21-11-26-survey-of-distributed-framework-in-federated-learning/)
- [Shard Manager: A Generic Shard Management Framework for Geo-distributed Applications](https://dl.acm.org/doi/10.1145/3477132.3483546). 2021. SOSP
- [Rabia: Simplifying State-Machine Replication Through Randomization](https://dl.acm.org/doi/10.1145/3477132.3483582). 2021. SOSP
- [Zico: Efficient GPU Memory Sharing for Concurrent DNN Training](https://www.usenix.org/conference/atc21/presentation/lim). 2021. ATC
- [Advanced synchronization techniques for task-based runtime systems](https://dl.acm.org/doi/10.1145/3437801.3441601). 2021. PPoPP
- [Are dynamic memory managers on GPUs slow?: a survey and benchmarks](https://dl.acm.org/doi/10.1145/3437801.3441612). 2021. PPoPP
- [DAPPLE: a pipelined data parallel approach for training large models](https://dl.acm.org/doi/10.1145/3437801.3441593). 2021. PPoPP
- [Sentinel: Efficient Tensor Migration and Allocation on Heterogeneous Memory Systems for Deep Learning](https://ieeexplore.ieee.org/abstract/document/9407112). 2021. HPCA
- [GAIA: A System for Interactive Analysis on Distributed Graphs Using a High-Level Language](https://www.usenix.org/system/files/nsdi21-qian.pdf). 2021. NSDI 
    - A memory management system for interactive graph computation, at distributed infrastructure layer. 
    - Chineses blog: [Survey of Distributed Framework in Federated Learning. Section 2](https://neth-lab.netlify.app/publication/21-11-26-survey-of-distributed-framework-in-federated-learning/)
- [Ownership: A Distributed Futures System for Fine-Grained Tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf). 2021. NSDI
- [HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism](https://www.usenix.org/conference/atc20/presentation/park). 2020. ATC

<h4 id="2.2.3">2.2.3 Key Papers</h4>

This section includes key paper in distributed framework for machine learning.

- [MapReduce: simplified data processing on large clusters](https://dl.acm.org/doi/abs/10.1145/1327452.1327492). 2004. OSDI
    - Chinese blog: [Summary of MapReduce](https://neth-lab.netlify.app/publication/21-1-4-summary-of-mapreduce/)
- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia). 2012. OSDI
    - Chinese blog: [Summary of Apache Spark](https://neth-lab.netlify.app/publication/21-3-19-summary-of-apache-spark/)
- [Large scale distributed deep networks](https://proceedings.neurips.cc/paper/2012/file/6aca97005c68f1206823815f66102863-Paper.pdf). 2012. NeurIPS
- [Scaling distributed machine learning with the parameter server](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu). 2014. OSDI
    - Chinese blog: [Summary of Parameter Server](https://neth-lab.netlify.app/publication/21-10-04-summary-of-parameter-server/)
- [Spark sql: Relational data processing in spark](https://dl.acm.org/doi/abs/10.1145/2723372.2742797). 2015. SIGMOD
- [Tensorflow: A system for large-scale machine learning](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi). 2016. OSDI
    - Chinese blog: [Summary of TensorFlow](https://neth-lab.netlify.app/publication/21-10-04-summary-of-tensorflow/)
- [Ray: A Distributed Framework for Emerging AI Applications](https://www.usenix.org/conference/osdi18/presentation/moritz). 2018. OSDI
    - Chinese blog: [Summary of Ray](https://neth-lab.netlify.app/publication/21-10-24-summary-of-ray/)
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html). 2019. NeurIPS
- [PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/abs/10.1145/3341301.3359646). 2019. SOSP

---

<h4 id="2.3">2.3 Computation & Communication Efficiency</h4>

<h4 id="2.3.1">2.3.1 Eficiency for FL</h4>

- [Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/conference/osdi21/presentation/lai). 2021. OSDI
- [Cheetah: Optimizing and Accelerating Homomorphic Encryption for Private Inference](https://ieeexplore.ieee.org/abstract/document/9407118). 2021. HPCA
- [Communication-Efficient Federated Learning with Adaptive Parameter Freezing](https://ieeexplore.ieee.org/abstract/document/9546506/). 2021. ICDCS
- [BatchCrypt: Efficient Homomorphic Encryption for Cross-Silo Federated Learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang). 2020. ATC
    - Use quantization method to compress encrypted data size, which reduces the costs of communication and computation.
    - Chinese blog: [Survey of Algorithm-based Optimization for Federated Learning. Section 3](https://neth-lab.netlify.app/publication/21-11-23-survey-of-communication-in-federated-learning/)
- [CMFL: Mitigating Communication Overhead for Federated Learning](https://ieeexplore.ieee.org/abstract/document/8885054). 2019. ICDCS
    - Reduce communication costs by reducing times of communication between edge devices and center server.
    - Chinese blog: [Survey of Algorithm-based Optimization for Federated Learning. Section 2](https://neth-lab.netlify.app/publication/21-11-23-survey-of-communication-in-federated-learning/)

<h4 id="2.3.2">2.3.2 Efficiency for Machine Learning</h4>

This section will introduce some researches focus on tradition Machine Learning, which is related to Federated Learning.

- [Octo: INT8 Training with Loss-aware Compensation and Backward Quantization for Tiny On-device Learning](https://www.usenix.org/conference/atc21/presentation/zhou-qihua). 2021. ATC
    - A INT8 quantization model, which is used in tiny on-device learning
    - Chinese blog: [Survey of Algorithm-based Optimization for Federated Learning. Section 4](https://neth-lab.netlify.app/publication/21-11-23-survey-of-communication-in-federated-learning/)
- [Hoplite: efficient and fault-tolerant collective communication for task-based distributed systems](https://dl.acm.org/doi/abs/10.1145/3452296.3472897). 2021. SIGCOMM
    - Introduce collective communication to task-based runtime distributed frameworks (e.g., Ray, Dask, Hydro)
    - Chinese blog: [Summary of Hoplite](https://neth-lab.netlify.app/publication/21-12-15-summary-of-hoplite/)
- [Gradient Compression Supercharged High-Performance Data Parallel DNN Training](https://dl.acm.org/doi/10.1145/3477132.3483553). 2021. SOSP
- [A novel memory-efficient deep learning training framework via error-bounded lossy compression](https://dl.acm.org/doi/10.1145/3437801.3441597). 2021. PPoPP
- [waveSZ: a hardware-algorithm co-design of efficient lossy compression for scientific data](https://dl.acm.org/doi/abs/10.1145/3332466.3374525). 2020. PPoPP
- [A generic communication scheduler for distributed DNN training acceleration](https://dl.acm.org/doi/10.1145/3341301.3359642). 2019. SOSP. Chinese blog: [Summary of A generic communication scheduler for distributed DNN training acceleration](https://neth-lab.netlify.app/publication/20-12-21-a-generic-communication-scheduler-for-distributed-dnn-training-acceleration/)
- [Gaia: Geo-distributed machine learning approaching lan speeds](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/hsieh). 2017. NSDI

---

<h4 id="2.4">2.4 Asynchronous Computing</h4>

<h4 id="2.4.1">2.4.1 Asynchronous for FL</h4>

- [VF2Boost: Very Fast Vertical Federated Gradient Boosting for Cross-Enterprise Learning](https://dl.acm.org/doi/abs/10.1145/3448016.3457241)
- [Secure bilevel asynchronous vertical federated learning with backward updating](https://arxiv.org/abs/2103.00958). 2021. arXiv

<h4 id="2.4.2">2.4.2 Asynchronous for Machine Learning</h4>

- [Asynchrony versus bulk-synchrony for a generalized N-body problem from genomics](https://dl.acm.org/doi/10.1145/3437801.3441580). 2021. PPoPP

---
<h2 id="3">3 Opensource Projects</h2>

- [FATE](https://github.com/search?q=federated+learning): Industrial framework for FL. From WeBank. Chinese blog: [Architecture of FATE](https://neth-lab.netlify.app/publication/21-3-12-architecture-of-fate/)
- [PySyft](https://github.com/OpenMined/PySyft)
- [Tensorflow Federated](https://github.com/tensorflow/federated)
- [PyTorch Implementation](https://github.com/shaoxiongji/federated-learning): An implementation based on PyTorch. From [shaoxiongji](https://github.com/shaoxiongji)