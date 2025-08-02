<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Portfolio Projects: Machine Learning with Differential Geometry and Topological Data Analysis

Based on my comprehensive research, I'll provide you with a detailed guide for developing compelling portfolio projects that combine differential geometry and topological data analysis (TDA) with machine learning, focusing on visual storytelling and demonstrable value.

## Current State of the Field

The intersection of differential geometry, TDA, and machine learning is experiencing significant growth, with **Geomstats** [^1][^2] emerging as the leading Python framework for geometric statistics and machine learning on manifolds. Current SOTA work at ICLR 2025 includes projects on learning differential equations on manifolds [^3], geometric deep learning models [^4], and applications like RNA structure design using geometric deep learning [^5].

## High-Impact Portfolio Project Categories

### 1. **Geometric Machine Learning Applications**

**Project: Medical Image Analysis with Riemannian Geometry**

- Apply geometric learning to medical imaging data (brain MRI, cardiac imaging)
- Use SPD matrices from diffusion tensor imaging for disease classification [^1]
- Implement Fréchet means and geodesic regression on manifolds [^6]
- **Visual story**: Interactive 3D brain visualizations showing geometric differences between healthy and diseased tissue

**Project: Financial Market Analysis on Hyperbolic Spaces**

- Model market correlations using hyperbolic geometry [^1]
- Implement clustering algorithms on hyperbolic manifolds
- Compare Euclidean vs. hyperbolic representations of market data
- **Visual story**: Interactive network visualizations showing market relationships in hyperbolic space


### 2. **Topological Data Analysis for Real-World Problems**

**Project: Drug Discovery with Persistent Homology**

- Use **Giotto-TDA** [^7] or **scikit-TDA** [^8] for molecular analysis
- Apply TDA to protein folding landscapes and drug-target interactions
- Implement persistence diagrams for chemical compound classification
- **Visual story**: 3D molecular visualizations with topological features highlighted

**Project: Social Network Analysis with TDA**

- Analyze community structure evolution using persistent homology [^9]
- Apply Mapper algorithm from **tda-mapper** [^10] to social media data
- Track topological changes in network structure over time
- **Visual story**: Interactive network evolution with topological birth-death cycles


### 3. **Cutting-Edge Geometric Deep Learning**

**Project: Point Cloud Classification with Geometric Neural Networks**

- Implement E(3)-equivariant networks for 3D object recognition [^4]
- Use **PyTorch Geometric** [^11] for graph neural networks on point clouds
- Compare performance with traditional CNN approaches
- **Visual story**: 3D interactive point cloud viewer with real-time classification

**Project: Time Series Forecasting on Manifolds**

- Model temporal data as curves on Riemannian manifolds [^3]
- Implement geometric recurrent neural networks
- Apply to financial time series or climate data
- **Visual story**: Interactive manifold visualizations showing temporal evolution


## Technical Implementation Strategy

### Core Libraries and Frameworks

**Primary Stack:**

- **Geomstats** [^1]: Core geometric computations and manifold operations
- **PyTorch Geometric** [^11]: Graph neural networks and geometric deep learning
- **Giotto-TDA** [^7]: Comprehensive TDA toolkit with scikit-learn compatibility
- **scikit-TDA** [^8]: Additional TDA algorithms and visualizations

**Visualization and Interaction:**

- **Plotly/Dash**: Interactive 3D visualizations and web applications
- **Three.js**: Advanced 3D web graphics for manifold visualizations
- **Gradio** [^12]: Rapid ML model demo deployment
- **Streamlit**: Data science web applications


### Visual Storytelling Components

**Essential Visualization Elements** [^13][^14]:

1. **Interactive 3D manifold representations** showing geometric transformations
2. **Real-time parameter adjustment** demonstrating algorithmic sensitivity
3. **Before/after comparisons** highlighting geometric vs. Euclidean approaches
4. **Persistence diagrams** with interactive filtering and exploration
5. **Geodesic path visualizations** on curved manifolds

**Storytelling Structure:**

1. **Problem Context**: Real-world motivation and challenges
2. **Geometric Insight**: Why differential geometry/TDA provides unique value
3. **Algorithm Implementation**: Technical approach with visual explanations
4. **Results Comparison**: Quantitative improvements over traditional methods
5. **Interactive Demonstration**: Live model with user-controllable parameters

## Rust Optimization Opportunities

The research reveals limited but growing Rust adoption for geometric algorithms [^15][^16]:

**Current Rust Libraries:**

- **diffgeom** [^15]: Basic differential geometry primitives
- **kurbo** [^17]: 2D geometric algorithms and curves
- **kornia-rs** [^18]: Computer vision library with 3D operations

**Optimization Potential:**

- Implement core TDA algorithms (persistent homology computation) in Rust
- Create Rust bindings for Geomstats' computational kernels
- Develop high-performance geometric neural network layers
- **Demonstration Value**: Profile Python vs. Rust implementations showing 2-10x speedups


## Project Portfolio Architecture

### Tier 1: Foundation Projects (2-3 months each)

1. **Geomstats Tutorial Extension**: Advanced examples with novel applications
2. **TDA Visualization Suite**: Interactive persistence diagram explorer
3. **Geometric Data Augmentation**: Novel augmentation techniques for ML

### Tier 2: Applied Research Projects (3-6 months each)

1. **Medical/Biological Application**: Disease classification using geometric features
2. **Financial/Economic Modeling**: Market analysis with hyperbolic geometry
3. **Computer Vision Innovation**: 3D object recognition with equivariant networks

### Tier 3: Novel Algorithmic Contributions (6+ months)

1. **New Geometric Learning Algorithm**: Original method with theoretical analysis
2. **Rust Implementation**: High-performance core library development
3. **Multi-modal Geometric Fusion**: Combining different geometric representations

## Competitive Advantages and Differentiation

**Against Traditional ML Portfolios:**

- Demonstrates mathematical sophistication and theoretical depth
- Shows ability to work with cutting-edge, rapidly evolving field
- Provides unique geometric insights often missed by standard approaches

**Technical Differentiators:**

- Real-time interactive demonstrations of geometric concepts [^13]
- Quantitative comparisons showing when geometric methods excel
- Integration of multiple geometric frameworks (differential geometry + TDA)
- Performance optimization through Rust implementations

**Presentation Excellence:**

- Mathematical rigor with accessible visual explanations
- Interactive web-based demonstrations deployable on any device
- Clear value propositions with measurable improvements
- Open-source contributions to established libraries

This comprehensive approach positions you at the forefront of geometric machine learning while creating portfolio pieces that demonstrate both theoretical understanding and practical implementation skills. The combination of visual storytelling, interactive demonstrations, and performance optimization creates compelling evidence of expertise in this high-value, rapidly growing field.

<div style="text-align: center">⁂</div>

[^1]: https://geomstats.github.io/getting_started/examples.html

[^2]: https://github.com/geomstats/geomstats

[^3]: https://openreview.net/forum?id=OwpLQrpdwE

[^4]: https://iclr.cc/virtual/2025/poster/30973

[^5]: https://openreview.net/forum?id=lvw3UgeVxS

[^6]: https://geomstats.github.io/tutorials/index.html

[^7]: https://www.jmlr.org/papers/volume22/20-325/20-325.pdf

[^8]: https://docs.scikit-tda.org

[^9]: https://link.springer.com/article/10.1007/s10462-024-10710-9

[^10]: https://tda-mapper.readthedocs.io

[^11]: https://github.com/pyg-team/pytorch_geometric

[^12]: https://gradio.app

[^13]: https://portf0l.io/blog/article/how-to-use-data-storytelling-in-portfolios

[^14]: https://www.thoughtspot.com/data-trends/best-practices/data-storytelling

[^15]: https://docs.rs/differential-geometry/latest/diffgeom/

[^16]: https://www.stephendiehl.com/posts/tensor_canonicalization_rust/

[^17]: https://docs.rs/kurbo/

[^18]: https://arxiv.org/html/2505.12425v1

[^19]: https://github.com/dotchen/torchgeometry

[^20]: https://www.linkedin.com/pulse/introduction-differential-geometry-patrick-nicolas-g8qtc

[^21]: https://geomstats.github.io/geomstats.github.io/index.html

[^22]: https://pytorch-geometric.readthedocs.io/en/2.6.1/get_started/introduction.html

[^23]: https://math.stackexchange.com/questions/584551/applications-of-differential-geometry-in-artificial-intelligence

[^24]: https://orbit.dtu.dk/files/350384757/Thesis_Alison_Pouplin-2.pdf

[^25]: https://www.youtube.com/watch?v=Ju-Wsd84uG0

[^26]: https://www.geeksforgeeks.org/data-science/introduction-to-pytorch-geometric/

[^27]: https://orbit.dtu.dk/en/publications/differential-geometric-approaches-to-machine-learning

[^28]: https://abess.readthedocs.io/en/latest/auto_gallery/5-scikit-learn-connection/plot_2_geomstats.html

[^29]: https://wandb.ai/int_pb/intro_to_pyg/reports/Introduction-to-PyTorch-Geometric-and-Weights-Biases--VmlldzozOTU1Njkz

[^30]: https://mathoverflow.net/questions/350228/how-useful-is-differential-geometry-and-topology-to-deep-learning

[^31]: https://pytorch-geometric.readthedocs.io

[^32]: https://www.reddit.com/r/MLQuestions/comments/sampvs/which_subject_would_be_more_useful_to_learn_for/

[^33]: https://gi.ece.ucsb.edu/geomstats-geometry-machine-learning

[^34]: https://colab.research.google.com/drive/1d0jLDwgNBtjBVQOFe8lO_1WrqTVeVZx9?usp=sharing

[^35]: https://www.hilarispublisher.com/open-access/topological-data-analysis-in-machine-learning-new-approaches-and-applications.pdf

[^36]: https://www.unibo.it/en/study/course-units-transferable-skills-moocs/course-unit-catalogue/course-unit/2024/473155

[^37]: https://teaspoontda.github.io/teaspoon/

[^38]: https://www.symphonyai.com/wp-content/uploads/2020/07/SAI-Topological-Data-Analysis-and-Machine-Learning-Better-Together-vf.pdf

[^39]: https://iclr.cc/virtual/2025/poster/29790

[^40]: https://www.netreveal.ai/wp-content/uploads/2021/03/TDA-and-Machine-Learning-Better-Together.pdf

[^41]: https://iclr.cc/virtual/2024/workshop/20581

[^42]: https://gudhi.inria.fr

[^43]: https://www.numberanalytics.com/blog/practical-guide-to-tda-in-machine-learning

[^44]: https://openreview.net/forum?id=AP0ndQloqR

[^45]: https://github.com/scikit-tda/scikit-tda

[^46]: https://www.tandfonline.com/doi/full/10.1080/23746149.2023.2202331

[^47]: https://iclr.cc/virtual/2025/papers.html

[^48]: https://github.com/FatemehTarashi/awesome-tda

[^49]: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.667963/full

[^50]: https://reintech.io/blog/optimizing-rust-code-for-performance-guide

[^51]: https://www.reddit.com/r/rust/comments/ja8nwi/looking_for_2d_geometry_library_in_rust/

[^52]: https://internals.rust-lang.org/t/rust-and-numeric-computation/20425

[^53]: https://github.com/bgalvao/nodevo

[^54]: https://geomstats.github.io/api/geomstats.numerics.html

[^55]: https://edu.anarcho-copy.org/Programming Languages/Rust/practical-machine-learning-rust-applications.pdf

[^56]: https://ems.press/content/serial-article-files/46872

[^57]: https://news.ycombinator.com/item?id=25325426

[^58]: https://www.reddit.com/r/rust/comments/13eij5q/is_anyone_doing_machine_learning_in_rust/

[^59]: https://www.reddit.com/r/rust/comments/1f87siw/an_optimization_thats_impossible_in_rust/

[^60]: https://internals.rust-lang.org/t/roadmap-2017-request-needs-of-hpc/4276

[^61]: https://deepcausality.com/blog/views-on-rust-ml/

[^62]: https://cran.r-project.org/package=rgeomstats

[^63]: https://lib.rs/algorithms

[^64]: https://github.com/deepcausality-rs/deep_causality

[^65]: https://openreview.net/forum?id=d6Kk7moQH3

[^66]: https://www.nature.com/articles/s41598-024-74045-9

[^67]: https://github.com/MelihGulum/Comprehensive-Data-Science-AI-Project-Portfolio

[^68]: https://www.azorobotics.com/News.aspx?newsID=15399

[^69]: https://github.com/dimitreOliveira/MachineLearning

[^70]: https://iclr.cc/virtual/2025/35631

[^71]: https://www.linkedin.com/pulse/reviews-papers-geometric-learning-2024-patrick-nicolas-5yzfc

[^72]: https://github.com/Yashodatta15/Project_Portfolio

[^73]: https://cse.engin.umich.edu/stories/eleven-papers-by-cse-researchers-at-iclr-2025

[^74]: https://www.sciencedirect.com/science/article/abs/pii/S0010482524002956

[^75]: https://www.projectpro.io/article/machine-learning-projects-on-github/465

[^76]: http://arxiv.org/list/math.DG/2024-09?skip=0\&show=250

[^77]: https://github.com/Vatsalparsaniya/Machine-Learning-Portfolio

[^78]: https://arxiv.org/abs/2305.14749

[^79]: https://orbit.dtu.dk/en/projects/deep-learning-theory-and-differential-geometry

[^80]: https://github.com/ToobaJamal/datascience-portfolio

[^81]: https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12210

[^82]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10863674/

[^83]: https://github.com/MilesCranmer/awesome-ml-demos

[^84]: https://saiwa.ai/landing/machine-learning-online-demo/

[^85]: https://www.juiceanalytics.com/writing/20-best-data-storytelling-examples

[^86]: https://www.pnas.org/doi/10.1073/pnas.2100473118

[^87]: https://github.com/trekhleb/machine-learning-experiments

[^88]: https://shorthand.com/the-craft/examples-of-powerful-data-storytelling/index.html

[^89]: https://agorism.dev/book/math/diff-geom/Visual Differential Geometry and Forms: A Mathematical Drama in Five Acts by Tristan Needham.pdf

[^90]: https://developmentalsystems.org/Interactive_DeepRL_Demo/

[^91]: https://careerfoundry.com/en/blog/data-analytics/data-analytics-portfolio-examples/

[^92]: https://www.math.colostate.edu/~clayton/research/papers/VDGF.pdf

[^93]: https://www.interaction-design.org/literature/article/design-with-intent-craft-your-portfolio-with-visual-storytelling-tools

[^94]: https://www2.compute.dtu.dk/~sohau/weekendwithbernie/Differential_geometry_for_generative_modeling.pdf

[^95]: https://magenta.withgoogle.com/demos/web/

[^96]: https://www.visualcinnamon.com/portfolio/

[^97]: https://www.jstor.org/stable/j.ctv1cmsmx5

