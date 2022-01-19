## System Design for Federated Learning
A paper list of federated learning - About system design. Currently, it mainly focuses on **distributed computing framework, communication & computation efficiency** for **Federated Learning** (Cross-Silo & Cross-devices). 

**Chinese blogs**: [Neth-Lab](https://neth-lab.netlify.app/publication/21-12-31-survey-of-system-design-for-distributed-ml-and-fl/), includes study notes, tutorials and development documents.

**Last update: Janary, 19th, 2022.**

---

## Catlog

**For some paper, we have provided some Chinese blogs.**

#### [1 Federated Learning Foundation](#1)
- [1.1 Blogs](#1.1)
- [1.2 Survey](#1.2)
- [1.3 Algorithms](#1.3)

#### [2 Research Areas](#2)
- [2.1 Survey](#2.1)
- [2.2 Optimization in algorithm perspective](#2.2)
- [2.3 Optimization in framework perspective](#2.3)
- [2.4 Optimization in communication perspective](#2.4)
- [2.5 Optimization for Memory](#2.5)

#### [3 Opensource Projects](#3)

#### [4 Researchers](#4)

---

<h2 id="1">1 Federated Learning Foundation</h2>

<h3 id="1.1">1.1 Blogs</h3>

- [Understand the types of federated learning](https://blog.openmined.org/federated-learning-types/). Sep 2020
    - A brief introduction to the terminology and classification of federal learning

<h3 id="1.2">1.2 Survey</h3>

- [Federated Machine Learning: Concept and Applications](https://dl.acm.org/doi/abs/10.1145/3298981). TITS. Qiang Yang. 2019
    - Chinese blog: [Overview of Federated Learning. Section 1](https://neth-lab.netlify.app/publication/21-3-2-overview-of-federated-learning/)

<h3 id="1.3">1.3 Algorithms</h3>

- [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://dl.acm.org/doi/abs/10.1145/3133956.3133982). 2017. CCS
    - Horizontal logistic regression algorithm

- [Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption](https://arxiv.org/abs/1711.10677). 2017. arXiv
    - Vertical logistic regression algorithm.
    - Chinese blog: [Machine Learning & Federated Learning. Section 5](https://neth-lab.netlify.app/publication/21-09-01-machine-learning-and-federated-learning/#section5)

- [SecureBoost: A Lossless Federated Learning Framework](https://ieeexplore.ieee.org/abstract/document/9440789). 2021. IEEE Intelligent Systems
    - Vertical secure boosting algorithm.
    - Chinese blog: [Machine Learning & Federated Learning. Section 6](https://neth-lab.netlify.app/publication/21-09-01-machine-learning-and-federated-learning/#section6)

---



<h2 id="2">2 Research Areas</h2>



<h3 id="2.1">2.1 Survey</h3>

- [A Comprehensive Survey of Privacy-preserving Federated Learning: A Taxonomy, Review, and Future Directions](https://dl.acm.org/doi/abs/10.1145/3460427). 2021. ACM Cmputing Surveys

- [A Quantitative Survey of Communication Optimizations in Distributed Deep Learning](https://ieeexplore.ieee.org/abstract/document/9275615). 2021. IEEE Network

- [A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://ieeexplore.ieee.org/abstract/document/9599369). 2021. TKDE
    - About system challenges for Fedrated Learning
    - Chinese blog: [Survey of System Design for Distributed ML & FL. Section 2.2](https://neth-lab.netlify.app/publication/21-12-31-survey-of-system-design-for-distributed-ml-and-fl/#section2)

- [System Optimization in Synchronous Federated Training: A Survey](https://arxiv.org/abs/2109.03999). 2021. arXiv
    - Focus on time-to-accuracy optimization for FL system
    - Chinese blog: [Survey of System Design for Distributed ML & FL. Section 2.3](https://neth-lab.netlify.app/publication/21-12-31-survey-of-system-design-for-distributed-ml-and-fl/#section2)

- [A Survey on Distributed Machine Learning](https://dl.acm.org/doi/abs/10.1145/3377454). 2020. ACM Computing Surveys
    - About system challenges for distributed machine learning
    - Chinese blog: [Survey of System Design for Distributed ML & FL. Section 2.1](https://neth-lab.netlify.app/publication/21-12-31-survey-of-system-design-for-distributed-ml-and-fl/)

- [Communication-Efficient Distributed Deep Learning: A Comprehensive Survey](https://arxiv.org/abs/2003.06307). 2020. arXiv

---


<h3 id="2.2">2.2 Optimization in algorithm perspective</h3>

<h4 id="2.2.1">2.2.1 Optimization for FL</h4>

- [Secure bilevel asynchronous vertical federated learning with backward updating](https://ojs.aaai.org/index.php/AAAI/article/view/17301). 2021. AAAI

- [VF2Boost: Very Fast Vertical Federated Gradient Boosting for Cross-Enterprise Learning](https://dl.acm.org/doi/abs/10.1145/3448016.3457241). 2021. SIGMOD

- [FedML: A Research Library and Benchmark for Federated Machine Learning](https://arxiv.org/abs/2007.13518). 2020. arXiv
    - A library and system architecture for FL.

- [FDML: A Collaborative Machine Learning Framework for Distributed Features](https://dl.acm.org/doi/abs/10.1145/3292500.3330765). 2019. KDD

<h4 id="2.2.2">2.2.2 Optimization for ML</h4>

</h4>

---




<h3 id="2.3">2.3 Optimization in framework perspective</h3>

This section will collect paper from both **Distributed framework for other computation (e.g. DNN and batch-based computation)** and **Distributed System for FL**

<h4 id="2.3.1">2.3.1 Optimization for FL</h4>

<h5 id="2.3.1.1">2.3.1.1 Topology</h5>

- [Sphinx: Enabling Privacy-Preserving Online Learning over the Cloud](https://www.ieee-security.org/TC/SP2022/program-papers.html). 2022. S&P

- [Throughput-Optimal Topology Design for Cross-Silo Federated Learning](https://arxiv.org/abs/2010.12229). 2020. arXiv

- [Towards Federated Learning at Scale: System Design](https://mlsys.org/Conferences/2019/doc/2019/193.pdf). 2019. MLSys
    - A framework for scaling horizontal FL. 
    - Chinese blogs: [Survey of Distributed Framework in Federated Learning. Section 3](https://neth-lab.netlify.app/publication/21-11-26-survey-of-framework-based-optimization-for-federated-learning/#section3)

<h5 id="2.3.1.2">2.3.1.2 Scheduler</h5>

- [Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/conference/osdi21/presentation/lai). 2021. OSDI

- [TiFL: A tier-based federated learning system](https://dl.acm.org/doi/abs/10.1145/3369583.3392686). 2020. HPDC

<h4 id="2.3.2">2.3.2 Optimization for Machine Learning</h4>

Since currently there is a few research paper about distributed framework for FL, here we provide related work focus on Machine Learning Framework for reference.

<h5 id="2.3.2.1">2.3.2.1 Topology</h5>

- [Gradient Compression Supercharged High-Performance Data Parallel DNN Training](https://dl.acm.org/doi/10.1145/3477132.3483553). 2021. SOSP

- [DAPPLE: a pipelined data parallel approach for training large models](https://dl.acm.org/doi/10.1145/3437801.3441593). 2021. PPoPP

- [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/conference/osdi21/presentation/wang). 2021. OSDI 

- [P3: Distributed Deep Graph Learning at Scale](https://www.usenix.org/conference/osdi21/presentation/gandhi). 2021. OSDI

- [HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism](https://www.usenix.org/conference/atc20/presentation/park). 2020. ATC

- [PipeDream: generalized pipeline parallelism for DNN training](https://dl.acm.org/doi/abs/10.1145/3341301.3359646). 2019. SOSP

- [Ray: A Distributed Framework for Emerging AI Applications](https://www.usenix.org/conference/osdi18/presentation/moritz). 2018. OSDI
    - Actor-based framework & parallelism methods for reinforce learning.
    - Chinese blog: [Summary of Ray](https://neth-lab.netlify.app/publication/21-10-24-summary-of-ray/)

- [Tensorflow: A system for large-scale machine learning](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi). 2016. OSDI
    - Chinese blog: [Summary of TensorFlow](https://neth-lab.netlify.app/publication/21-10-04-summary-of-tensorflow/)

- [Spark sql: Relational data processing in spark](https://dl.acm.org/doi/abs/10.1145/2723372.2742797). 2015. SIGMOD

- [Scaling distributed machine learning with the parameter server](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu). 2014. OSDI
    - Chinese blog: [Summary of Parameter Server](https://neth-lab.netlify.app/publication/21-10-04-summary-of-parameter-server/)

- [Large scale distributed deep networks](https://proceedings.neurips.cc/paper/2012/file/6aca97005c68f1206823815f66102863-Paper.pdf). 2012. NeurIPS

- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia). 2012. OSDI
    - Chinese blog: [Summary of Apache Spark](https://neth-lab.netlify.app/publication/21-3-19-summary-of-apache-spark/)

- [MapReduce: simplified data processing on large clusters](https://dl.acm.org/doi/abs/10.1145/1327452.1327492). 2004. OSDI
    - Chinese blog: [Summary of MapReduce](https://neth-lab.netlify.app/publication/21-1-4-summary-of-mapreduce/)

<h5 id="2.3.2.2">2.3.2.2 Scheduler</h5>

- [CrystalPerf: Learning to Characterize the Performance of Dataflow Computation through Code Analysis](https://www.usenix.org/conference/atc21/presentation/tian). 2021. ATC

- [Scaling Large Production Clusters with Partitioned Synchronization](https://www.usenix.org/conference/atc21/presentation/feng-yihui). 2021. ATC
    - A distributed resource scheduler architecture. Use partition synchronization method to reduce the impact of contention on high-quality resources and staleness of local states, which causes high scheduling latency.
    - Chinese blog: [Survey of Framework-based Optimization for Federated Learning. Section 4](https://neth-lab.netlify.app/publication/21-11-26-survey-of-framework-based-optimization-for-federated-learning/#section4)

- [Shard Manager: A Generic Shard Management Framework for Geo-distributed Applications](https://dl.acm.org/doi/10.1145/3477132.3483546). 2021. SOSP

- [Advanced synchronization techniques for task-based runtime systems](https://dl.acm.org/doi/10.1145/3437801.3441601). 2021. PPoPP

- [Ownership: A Distributed Futures System for Fine-Grained Tasks](https://www.usenix.org/system/files/nsdi21-wang.pdf). 2021. NSDI

---




<h3 id="2.4">2.4 Optimization in communication perspective</h3>

<h4 id="2.4.1">2.4.1 Optimization for FL</h4>

- [Cheetah: Optimizing and Accelerating Homomorphic Encryption for Private Inference](https://ieeexplore.ieee.org/abstract/document/9407118). 2021. HPCA

- [Communication-Efficient Federated Learning with Adaptive Parameter Freezing](https://ieeexplore.ieee.org/abstract/document/9546506/). 2021. ICDCS

- [BatchCrypt: Efficient Homomorphic Encryption for Cross-Silo Federated Learning](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang). 2020. ATC
    - Use quantization method to compress encrypted data size, which reduces the costs of communication and computation.
    - Chinese blog: [Summary of BatchCrypt](https://neth-lab.netlify.app/publication/22-01-12-summary-of-batchcrypt/)

- [Communication-Efficient Federated Deep Learning With Layerwise Asynchronous Model Update and Temporally Weighted Aggregation](https://ieeexplore.ieee.org/abstract/document/8945292). 2020. TNNLS

- [CMFL: Mitigating Communication Overhead for Federated Learning](https://ieeexplore.ieee.org/abstract/document/8885054). 2019. ICDCS
    - Reduce communication costs by reducing times of communication between edge devices and center server. Similar as Gaia, it introduces relevance between local updates and global updates to determine whether transfer the local updates to center server.
    - Chinese blog: [Summary of CMFL](https://neth-lab.netlify.app/publication/22-01-18-summary-of-cmfl/)


<h4 id="2.4.2">2.4.2 Optimization for Machine Learning</h4>

This section will introduce some researches focus on tradition Machine Learning, which is related to Federated Learning.

- [Octo: INT8 Training with Loss-aware Compensation and Backward Quantization for Tiny On-device Learning](https://www.usenix.org/conference/atc21/presentation/zhou-qihua). 2021. ATC
    - A INT8 quantization model, which is used in tiny on-device learning
    - Chinese blog: [Survey of Communication-based Optimization for Federated Learning. Section 4](https://neth-lab.netlify.app/publication/22-01-02-survey-of-communication-based-optimization-for-federated-learning/#section4)

- [Hoplite: efficient and fault-tolerant collective communication for task-based distributed systems](https://dl.acm.org/doi/abs/10.1145/3452296.3472897). 2021. SIGCOMM
    - Introduce collective communication to task-based runtime distributed frameworks (e.g., Ray, Dask, Hydro)
    - Chinese blog: [Summary of Hoplite](https://neth-lab.netlify.app/publication/21-12-15-summary-of-hoplite/)

- [FetchSGD: Communication-Efficient Federated Learning with Sketching](https://arxiv.org/abs/2007.07682). 2020. ICML

- [waveSZ: a hardware-algorithm co-design of efficient lossy compression for scientific data](https://dl.acm.org/doi/abs/10.1145/3332466.3374525). 2020. PPoPP

- [Communication-efficient distributed sgd with sketching](https://arxiv.org/abs/1903.04488). 2019. NIPS
    - Use sketch method to choose top-k gradient elements so that workers just need transfer top-k updates, which reduces communication cost.
    - Chinese blog: [Summary of Sketching. Section 2](https://neth-lab.netlify.app/publication/22-01-18-summary-of-sketching/#section2)

- [A generic communication scheduler for distributed DNN training acceleration](https://dl.acm.org/doi/10.1145/3341301.3359642). 2019. SOSP.   
    - Chinese blog: [Summary of A generic communication scheduler for distributed DNN training acceleration](https://neth-lab.netlify.app/publication/20-12-21-a-generic-communication-scheduler-for-distributed-dnn-training-acceleration/)

- [Sketchml: Accelerating distributed machine learning with data sketches](https://dl.acm.org/doi/abs/10.1145/3183713.3196894). 2018. SIGMOD

- [Gradient Sparsification for Communication-Efficient Distributed Optimization](https://proceedings.neurips.cc/paper/2018/file/3328bdf9a4b9504b9398284244fe97c2-Paper.pdf). 2018. NIPS

- [Horovod: fast and easy distributed deep learning in TensorFlow](https://arxiv.org/abs/1802.05799). 2018. arXiv
    - Chinese blog: [Summary of Hoplite. Section 3](https://neth-lab.netlify.app/publication/21-12-15-summary-of-hoplite/#section3)

- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html). 2017. NIPS

- [Gaia: Geo-distributed machine learning approaching lan speeds](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/hsieh). 2017. NSDI
    - Use significant function to determine the importance of updates. If smaller than threshold, do not transfer so that mitigate the overhead of WAN bandwidth. Introduce a new parallelism method called ASP, which is proved can guarantee convergence requirement.
    - Chinese blog: <a href="https://neth-lab.netlify.app/publication/22-01-16-summary-of-gaia/">Summary of Gaia</a>

---

<h4 id="2.5">2.5 Optimization for Memory</h4>

<h4 id="2.5.1">2.5.1 Optimization for FL</h4>



<h4 id="2.5.2">2.5.2 Optimization for Machine Learning</h4>

- [GAIA: A System for Interactive Analysis on Distributed Graphs Using a High-Level Language](https://www.usenix.org/system/files/nsdi21-qian.pdf). 2021. NSDI 
    - A memory management system for interactive graph computation, at distributed infrastructure layer. 
    - Chineses blog: [Survey of Framework-based Optimization for Federated Learning. Section 2](https://neth-lab.netlify.app/publication/21-11-26-survey-of-framework-based-optimization-for-federated-learning/#section2)

- [A novel memory-efficient deep learning training framework via error-bounded lossy compression](https://dl.acm.org/doi/10.1145/3437801.3441597). 2021. PPoPP

- [Zico: Efficient GPU Memory Sharing for Concurrent DNN Training](https://www.usenix.org/conference/atc21/presentation/lim). 2021. ATC

- [Are dynamic memory managers on GPUs slow?: a survey and benchmarks](https://dl.acm.org/doi/10.1145/3437801.3441612). 2021. PPoPP

- [Sentinel: Efficient Tensor Migration and Allocation on Heterogeneous Memory Systems for Deep Learning](https://ieeexplore.ieee.org/abstract/document/9407112). 2021. HPCA

---
<h2 id="3">3 Opensource Projects</h2>

- [FATE](https://github.com/search?q=federated+learning): Industrial framework for FL. From WeBank. Chinese blog: [Architecture of FATE](https://neth-lab.netlify.app/publication/21-3-12-architecture-of-fate/)

- [PySyft](https://github.com/OpenMined/PySyft)

- [Tensorflow Federated](https://github.com/tensorflow/federated)

- [PyTorch Implementation](https://github.com/shaoxiongji/federated-learning): An implementation based on PyTorch. From [shaoxiongji](https://github.com/shaoxiongji)

<h2 id="4">4 Researchers</h2>

- [Mosharaf Chowdhury, University of Michigan](https://scholar.google.com.hk/citations?user=Dzh5C9EAAAAJ&hl=zh-CN&oi=sra)

- [Ion Stoica, Professor of Computer Science, UC Berkeley](https://scholar.google.com.hk/citations?hl=zh-CN&user=vN-is70AAAAJ)

- [Matei Zaharia, Stanford DAWN Lab and Databricks](https://scholar.google.com.hk/citations?hl=zh-CN&user=I1EvjZsAAAAJ)

- [Wei Wang, HKUST](https://scholar.google.com/citations?user=FeJrzPMAAAAJ)
