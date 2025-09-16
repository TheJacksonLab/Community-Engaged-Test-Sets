# Community-Engaged-Test-Sets
We introduce a generalizable mode of scientific outreach that couples a published study to a community-engaged test set, enabling post-publication evaluation by the broader ML community. This approach is demonstrated using a prior study on AI-guided discovery of photostable light-harvesting small molecules (<a href="https://doi.org/10.1038/s41586-024-07892-1">Closed-loop transfer enables artificial intelligence to yield chemical knowledge</a>, <a href="https://github.com/TheJacksonLab/ClosedLoopTransfer/tree/main">GitHub Repository</a>). After publishing an experimental dataset and in-house ML models, we leveraged automated block chemistry to synthesize nine additional light-harvesting molecules to serve as a blinded community test set. We then hosted an open Kaggle competition (<a href="https://www.kaggle.com/competitions/molecular-machine-learning">Molecular Data Machine Learning</a>) where we challenged the world community to outperform our best in-house predictive photostability model. 


<img width="500" height="524" alt="Screenshot 2025-07-25 at 1,13,21 PM-Picsart-AiImageEnhancer" src="https://github.com/user-attachments/assets/5eb3f7ed-a5b4-4b98-88e9-431e31e65049" />



# Dataset
Our data was collected in house using automated modular block synthesis and automated characterization. We had a training set of 42 light-harvesting small molecules and a test set of 7 molecules. For each molecule in the dataset, we provided 144 chemical features calculated from RDKit and TDDFT as well as the corresponding SMILES string.

<a href="https://github.com/Jasonwu617/Community-Engaged-Test-Sets/blob/main/train.csv">train.csv</a>

<a href="https://github.com/Jasonwu617/Community-Engaged-Test-Sets/blob/main/test.csv">test.csv</a>

<a href="https://github.com/Jasonwu617/Community-Engaged-Test-Sets/blob/main/solution.csv">solution.csv</a>



# Authors
Jason L. Wu, David M. Friday, Changhyun Hwang, Seungjoo Yi, Tiara C. Torres-Flores, Martin D. Burke, Ying Diao, Charles M. Schroeder, Nicholas E. Jackson

# Funding Acknowledgements
This work was supported by the Molecule Maker Lab Institute, an AI Research Institutes program supported by the US National Science Foundation under grant no. 2019897 and grant no. 2505932.





