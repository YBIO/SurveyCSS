<img src="illustration/AIRVICLAB.png" width="200px">

# <div align='center'> SurveyCSS (Continuously updating) </div>

## NEWS
### 24-08-18 :smiley: Our paper has been accepted in IEEE TPAMI

### 24-07-23 :blush: Our latest verison of paper has been released in ArXiv!

## Our Paper (Updated at 24-07-23)
A Survey on Continual Semantic Segmentation: Theory, Challenge, Method and Application
[[paper]](https://browse.arxiv.org/abs/2310.14277) | [[blog]](https://ybio.github.io/2024/06/01/blog_SurveyCSS/)

## Abstract
Continual learning, also known as incremental learning or life-long learning, stands at the forefront of deep learning and AI systems. It breaks through the obstacle of one-way training on close sets and enables continuous adaptive learning on open-set conditions. In the recent decade, continual learning has been explored and applied in multiple fields especially in computer vision covering classification, detection and segmentation tasks. Continual semantic segmentation (CSS), of which the dense prediction peculiarity makes it a challenging, intricate and burgeoning task. In this paper, we present a review of CSS, committing to building a comprehensive survey on problem formulations, primary challenges, universal datasets, neoteric theories and multifarious applications. Concretely, we begin by elucidating the problem definitions and primary challenges. Based on an in-depth investigation of relevant approaches, we sort out and categorize current CSS models into two main branches including **data-replay** and **data-free** sets. In each branch, the corresponding approaches are similarity-based clustered and thoroughly analyzed, following qualitative comparison and quantitative reproductions on relevant datasets. Besides, we also introduce four CSS specialities with diverse application scenarios and development tendencies. Furthermore, we develop a benchmark for CSS encompassing representative references, evaluation results and reproductions. We hope this survey can serve as a reference-worthy and stimulating contribution to the advancement of the life-long learning field, while also providing valuable perspectives for related fields. 

  ![](https://img.shields.io/badge/task%20incre.-gray) 
  ![](https://img.shields.io/badge/domain%20incre.-blue) 
  ![](https://img.shields.io/badge/class%20incre.-yellow) 
  ![](https://img.shields.io/badge/modality%20incre.-purple)

  
![task_legend](illustration/task_legend.png)
![method_category](illustration/category.png)



## <div align='center'> Data-free Approaches </div>
### 2025

- SVSRD: Spatial Visual and Statistical Relation Distillation for Class-Incremental Semantic Segmentation [TMM 2025] [[paper]](https://ieeexplore.ieee.org/abstract/document/10891485) ![](https://img.shields.io/badge/class%20incre.-yellow)

- CLIMB-3D: Continual Learning for Imbalanced 3D Instance Segmentation [ArXiv 2025] [[paper]](https://arxiv.org/pdf/2502.17429) [[code]](https://github.com/vgthengane/CLIMB3D) ![](https://img.shields.io/badge/class%20incre.-yellow)

- IPSeg: Image Posterior Mitigates Semantic Drift in Class-Incremental Segmentation [ArXiv 2025] [[paper]](https://arxiv.org/pdf/2502.04870) [[code]](https://github.com/YanFangCS/IPSeg) ![](https://img.shields.io/badge/class%20incre.-yellow)

- Domain-Incremental Semantic Segmentation for Traffic Scenes. [TITS 2025] [[paper]](https://ieeexplore.ieee.org/abstract/document/10843143) ![](https://img.shields.io/badge/domain%20incre.-blue)

- Domain-Incremental Semantic Segmentation for Autonomous Driving under Adverse Driving Conditions [ArXiv 2025] [[paper]](https://arxiv.org/pdf/2501.05246) ![](https://img.shields.io/badge/domain%20incre.-blue)


- Incremental few-shot instance segmentation without fine-tuning on novel classes [CVIU 2025] [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1077314225000463) ![](https://img.shields.io/badge/class%20incre.-yellow)

- Modality-Incremental Learning with Disjoint Relevance Mapping Networks for Image-based Semantic Segmentation [WACV 2025] [[paper]](https://arxiv.org/abs/2411.17610)  ![](https://img.shields.io/badge/modality%20incre.-purple)


### 2024
- Strike a Balance in Continual Panoptic Segmentation [ECCV 2024] [[paper]](https://arxiv.org/pdf/2407.16354) [[code]](https://github.com/jinpeng0528/BalConpas) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Continual Panoptic Perception: Towards Multi-modal Incremental Interpretation of Remote Sensing Images [ACMMM 2024] [[paper]](https://arxiv.org/abs/2407.14242) [[code]](https://github.com/YBIO/CPP) ![](https://img.shields.io/badge/class%20incre.-yellow) ![](https://img.shields.io/badge/modality%20incre.-purple)

- A Surprisingly Simple Approach to Generalized Few-Shot Semantic Segmentation [NeruIPS 2024] [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f75a57e9c71e8369da0150ea769d5a2-Abstract-Conference.html) ![](https://img.shields.io/badge/class%20incre.-yellow)

- SegACIL: Solving the Stability-Plasticity Dilemma in Class-Incremental Semantic Segmentation [ArXiv 2024] [[paper]](https://arxiv.org/abs/2412.10834) [[code]](https://github.com/qwrawq/SegACIL) ![](https://img.shields.io/badge/class%20incre.-yellow)
- SAM-IF: Leveraging SAM for Incremental Few-Shot Instance Segmentation [Arxiv 2024] [[paper]](https://arxiv.org/pdf/2412.11034) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation [ECCV 2024] [[paper]](https://arxiv.org/pdf/2407.14142) [[code]](https://github.com/zhengyuan-xie/ECCV24_NeST) ![](https://img.shields.io/badge/class%20incre.-yellow) 
- Low-Rank Continual Pyramid Vision Transformer: Incrementally Segment Whole-Body Organs in CT with Light-Weighted Adaptation [MICCAI 2024] [[paper]](https://arxiv.org/pdf/2410.04689) ![](https://img.shields.io/badge/domain%20incre.-blue) 
- Federated Cross-Incremental Self-Supervised Learning for Medical Image Segmentation [TNNLS 2024] [[paper]](https://ieeexplore.ieee.org/abstract/document/10715722)  ![](https://img.shields.io/badge/domain%20incre.-blue) 
- CLMS: Bridging Domain Gaps in Medical Imaging Segmentation with Source-Free Continual Learning for Robust Knowledge Transfer and Adaptation [Medical Image Analysis 2024] [[paper]](https://www.sciencedirect.com/science/article/pii/S1361841524003293)  ![](https://img.shields.io/badge/domain%20incre.-blue) 
- AWF: Adaptive Weight Fusion for Enhanced Class Incremental Semantic Segmentation [ArXiv 2024] [[paper]](https://arxiv.org/pdf/2409.08516) ![](https://img.shields.io/badge/class%20incre.-yellow)
- CIT: Rethinking Class-incremental Semantic Segmentation with a Class Independent Transformation [ArXiv 2024] [[paper]](https://arxiv.org/pdf/2411.02715) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Domain-Incremental Learning for Remote Sensing Semantic Segmentation With Multifeature Constraints in Graph Space [TGRS 2024] [[paper]](https://ieeexplore.ieee.org/abstract/document/10723740) ![](https://img.shields.io/badge/domain%20incre.-blue)
- kNN-CLIP: Retrieval Enables Training-Free Segmentation on Continually Expanding Large Vocabularies [TMLR 2024] [[paper]](https://arxiv.org/pdf/2404.09447)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- Background Adaptation with Residual Modeling for Exemplar-Free Class-Incremental Semantic Segmentation. [ECCV 2024] [[paper]](https://arxiv.org/abs/2407.09838) [[code]](https://andyzaq.github.io/barmsite/) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Mitigating Background Shift in Class-Incremental Semantic Segmentation [ECCV 2024] [[paper]](https://arxiv.org/pdf/2407.11859) [[code]](https://github.com/RoadoneP/ECCV2024_MBS) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation [ECCV 2024] [[paper]](https://arxiv.org/pdf/2407.14142) [[code]](https://github.com/zhengyuan-xie/ECCV24_NeST) ![](https://img.shields.io/badge/class%20incre.-yellow)
- L2A: Learning Affinity from Attention for Weakly Supervised Continual Semantic Segmentation [TCSVT 2024] [[paper]](https://ieeexplore.ieee.org/document/10683729) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Prompt-Guided Semantic-aware Distillation for
Weakly Supervised Incremental Semantic
Segmentation. [TCSVT 2024] [[paper]](https://ieeexplore.ieee.org/abstract/document/10553257) [[code]](https://github.com/Magic-Nova77/PGSD) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Cs2K: Class-specific and Class-shared Knowledge Guidance for Incremental Semantic Segmentation. [ArXiv 2024] [[paper]](https://arxiv.org/pdf/2407.09047) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Incremental Nuclei Segmentation from Histopathological Images via Future-class Awareness and Compatibility-inspired Distillation [CVPR 2024] [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Incremental_Nuclei_Segmentation_from_Histopathological_Images_via_Future-class_Awareness_and_CVPR_2024_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- CoMasTRe: Continual Segmentation with Disentangled Objectness Learning and Class Recognition [CVPR 2024] [[paper]](https://arxiv.org/pdf/2403.03477) ![](https://img.shields.io/badge/class%20incre.-yellow)
- ECLIPSE: Efficient Continual Learning in Panoptic Segmentation with Visual Prompt Tuning [CVPR 2024] [[paper]](https://arxiv.org/pdf/2403.20126) ![](https://img.shields.io/badge/class%20incre.-yellow)
- SimCS:  Simulation for Domain Incremental Online Continual Segmentation [AAAI 2024] [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28952)  ![](https://img.shields.io/badge/domain%20incre.-blue)
- SegViT v2: [SegViTv2: Exploring Efficient and Continual Semantic Segmentation with Plain Vision Transformers] [IJCV 2024] [[paper]](https://arxiv.org/abs/2306.06289) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Tendency-driven Mutual Exclusivity for Weakly Supervised Incremental Semantic Segmentation [Arxiv 2024] [[paper]](https://arxiv.org/pdf/2404.11981.pdf) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Layer-Specific Knowledge Distillation for Class Incremental Semantic Segmentation [TIP 2024] [[paper]](https://ieeexplore.ieee.org/abstract/document/10462931) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Privacy-Preserving Synthetic Continual Semantic Segmentation for Robotic Surgery [TMI 2024] [[paper]](https://arxiv.org/pdf/2402.05860.pdf) ![](https://img.shields.io/badge/class%20incre.-yellow)
- ConSept: Continual Semantic Segmentation via Adapter-based Vision Transformer [ArXiv 2024] [[paper]](https://arxiv.org/pdf/2402.16674.pdf) ![](https://img.shields.io/badge/class%20incre.-yellow)
- A Domain Adaptive Semantic Segmentation Method Using Contrastive Learning and Data Augmentation [Neural Processing Letters 2024] [[paper]](https://link.springer.com/article/10.1007/s11063-024-11529-9) ![](https://img.shields.io/badge/domain%20incre.-blue)
- Boosting knowledge diversity, accuracy, and stability via tri-enhanced distillation for domain continual medical image segmentation [Medical Image Analysis 2024] [[paper]](https://www.sciencedirect.com/science/article/pii/S1361841524000379)  ![](https://img.shields.io/badge/domain%20incre.-blue)
- Cross-Domain Few-Shot Incremental Learning for Point-Cloud Recognition. [WACV2024] [[paper]](https://openaccess.thecvf.com/content/WACV2024/html/Tan_Cross-Domain_Few-Shot_Incremental_Learning_for_Point-Cloud_Recognition_WACV_2024_paper.html) ![](https://img.shields.io/badge/domain%20incre.-blue)
- Towards Domain-Aware Knowledge Distillation for Continual Model Generalization. [WACV 2024] [[paper]](https://openaccess.thecvf.com/content/WACV2024/html/Reddy_Towards_Domain-Aware_Knowledge_Distillation_for_Continual_Model_Generalization_WACV_2024_paper.html) ![](https://img.shields.io/badge/domain%20incre.-blue)
- MDINet: Multi-Domain Incremental Network for Change Detection. [TGRS 2024] [[paper]](https://ieeexplore.ieee.org/abstract/document/10379022) ![](https://img.shields.io/badge/domain%20incre.-blue)


### 2023
- Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments [NeurIPS 2023] [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/ce3cf998b7f59271e80ce03fb74a7115-Paper-Conference.pdf) ![](https://img.shields.io/badge/class%20incre.-yellow)
- CoDEPS: Online Continual Learning for Depth Estimation and Panoptic Segmentation. [ArXiv 2023] [[paper]](https://arxiv.org/pdf/2303.10147.pdf) ![](https://img.shields.io/badge/domain%20incre.-blue)
- FALCON: Fairness Learning via Contrastive Attention Approach to Continual Semantic Scene Understanding in Open World. [ArXiv 2023] [[paper]](https://arxiv.org/pdf/2311.15965.pdf) ![](https://img.shields.io/badge/class%20incre.-yellow)
- L2MNet: Enhancing Continual Semantic Segmentation with Mask Matching [PRCV 2023] [[paper]](https://link.springer.com/chapter/10.1007/978-981-99-8549-4_11)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- SDCL: Subspace Distillation for Continual Learning [[Neural Networks 2024]] [[paper]](https://www.sciencedirect.com/science/article/pii/S0893608023004057) [[code]](https://github.com/csiro-robotics/SDCL) ![](https://img.shields.io/badge/task%20incre.-gray) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Inherit With Distillation and Evolve With Contrast: Exploring Class Incremental Semantic Segmentation Without Exemplar Memory [TPAMI 2023] [[paper]](https://ieeexplore.ieee.org/abstract/document/10120962) ![](https://img.shields.io/badge/class%20incre.-yellow)
- CoinSeg: Contrast Inter- and Intra- Class Representations for Incremental Segmentation [ICCV 2023] [[paper]](https://arxiv.org/pdf/2310.06368v1.pdf) [[code]](https://github.com/zkzhang98/CoinSeg) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Class-incremental Continual Learning for Instance Segmentation with Image-level Weak Supervision. [ICCV 2023] [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Hsieh_Class-incremental_Continual_Learning_for_Instance_Segmentation_with_Image-level_Weak_Supervision_ICCV_2023_paper.html) [[code]](https://github.com/AI-Application-and-Integration-Lab/CL4WSIS)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- Effects of architectures on continual semantic segmentation. [ArXiv 2023] [[paper]](https://arxiv.org/abs/2302.10718)
- LGKD: [Label-Guided Knowledge Distillation for Continual Semantic Segmentation on 2D Images and 3D Point Clouds] [ICCV 2023] [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Label-Guided_Knowledge_Distillation_for_Continual_Semantic_Segmentation_on_2D_Images_ICCV_2023_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Preparing the Future for Continual Semantic Segmentation. [ICCV 2023] [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_Preparing_the_Future_for_Continual_Semantic_Segmentation_ICCV_2023_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- GSC: [Gradient-Semantic Compensation for Incremental Semantic Segmentation] [TMM 2023] [[paper]](https://arxiv.org/abs/2307.10822)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- MiCro: [MiCro: Modeling Cross-Image Semantic Relationship Dependencies for Class-Incremental Semantic Segmentation in Remote Sensing Images.] [TGRS 2023] [[paper]](https://ieeexplore.ieee.org/abstract/document/10188852)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- CL-PCSS: [Continual Learning for LiDAR Semantic Segmentation: Class-Incremental and Coarse-to-Fine strategies on Sparse Data] [CVPR 2023] [[paper]](https://openaccess.thecvf.com/content/CVPR2023W/CLVision/html/Camuffo_Continual_Learning_for_LiDAR_Semantic_Segmentation_Class-Incremental_and_Coarse-To-Fine_Strategies_CVPRW_2023_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- AWT: [Attribution-aware Weight Transfer: A Warm-Start Initialization for Class-Incremental Semantic Segmentation] [WACV 2023] [[paper]](https://openaccess.thecvf.com/content/WACV2023/htmlGoswami_Attribution-Aware_Weight_Transfer_A_Warm-Start_Initialization_for_Class-Incremental_Semantic_Segmentation_WACV_2023_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- EWF: [Endpoints Weight Fusion for Class Incremental Semantic Segmentation] [CVPR 2023] [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_Endpoints_Weight_Fusion_for_Class_Incremental_Semantic_Segmentation_CVPR_2023_paper.pdf) [[code]](https://github.com/schuy1er/EWF_official) ![](https://img.shields.io/badge/class%20incre.-yellow)
- SATS: [SATS: Self-attention transfer for continual semantic segmentation] [PR 2023] [[paper]](https://browse.arxiv.org/pdf/2203.07667.pdf) [[code]](https://github.com/QIU023/SATS_Continual_Semantic_Seg)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- Incrementer: [Incrementer: Transformer for Class-Incremental Semantic Segmentation with Knowledge Distillation Focusing on Old Class][CVPR 2023][[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Shang_Incrementer_Transformer_for_Class-Incremental_Semantic_Segmentation_With_Knowledge_Distillation_Focusing_CVPR_2023_paper.html)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- S3R: [S3R: Shape and Semantics-Based Selective Regularization for Explainable Continual Segmentation Across Multiple Sites] [TMI 2023] [[paper]](https://ieeexplore.ieee.org/abstract/document/10078916) ![](https://img.shields.io/badge/domain%20incre.-blue)
- ContinualPMF:[Continual Road-Scene Semantic Segmentation via Feature-Aligned Symmetric Multi-Modal Network] [ArXiv 2023] [[paper]](https://arxiv.org/abs/2308.04702) ![](https://img.shields.io/badge/modality%20incre.-purple)
- CoMFormer: [CoMFormer: Continual Learning in Semantic and Panoptic Segmentation] [CVPR 2023] [[paper]](https://arxiv.org/abs/2211.13999)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- DILRS: [DILRS: Domain-Incremental Learning for Semantic Segmentation in Multi-Source Remote Sensing Data] [Remote Sensing 2023] [[paper]](https://www.mdpi.com/2072-4292/15/10/2541) ![](https://img.shields.io/badge/domain%20incre.-blue)
- RaSP: [RaSP: Relation-aware Semantic Prior for Weakly Supervised Incremental Segmentation]  [ArXiv 2023] [[paper]](https://arxiv.org/abs/2305.19879) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Efficient Multi-Grained Knowledge Reuse for Class Incremental Segmentation. [ArXiv 2023] [[paper]](https://arxiv.org/pdf/2306.02027.pdf) ![](https://img.shields.io/badge/class%20incre.-yellow)
- CISDQ: Continual Learning for Image Segmentation with Dynamic Query. [TCSVT 2023] [[paper]](https://ieeexplore.ieee.org/abstract/document/10335718) ![](https://img.shields.io/badge/class%20incre.-yellow)
- 
### 2022
- Multi-Head Distillation for Continual Unsupervised Domain Adaptation in Semantic Segmentation. [CVPRW 2022] [[paper]](https://openaccess.thecvf.com/content/CVPR2022W/CLVision/papers/Saporta_Multi-Head_Distillation_for_Continual_Unsupervised_Domain_Adaptation_in_Semantic_Segmentation_CVPRW_2022_paper.pdf) ![](https://img.shields.io/badge/domain%20incre.-blue)
- CONDA: Continual Unsupervised Domain Adaptation Learning in Visual Perception for Self-Driving Cars. [ArXiv 2022] [[paper]](https://arxiv.org/pdf/2212.00621.pdf) ![](https://img.shields.io/badge/domain%20incre.-blue)
- iFS-RCNN: An Incremental Few-shot Instance Segmenter [CVPR 2022] [[paper]](https://arxiv.org/pdf/2205.15562.pdf) ![](https://img.shields.io/badge/class%20incre.-yellow)
- WILSON: [Incremental Learning in Semantic Segmentation From Image Labels] [CVPR 2022] [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Cermelli_Incremental_Learning_in_Semantic_Segmentation_From_Image_Labels_CVPR_2022_paper.html) [[code]](https://github.com/fcdl94/WILSON) ![](https://img.shields.io/badge/class%20incre.-yellow)
- RBC: [RBC: Rectifying the Biased Context in Continual Semantic Segmentation] [ECCV 2022] [[paper]](https://browse.arxiv.org/pdf/2203.08404.pdf) [[code]](https://github.com/dcdcvgroup/RBC) ![](https://img.shields.io/badge/class%20incre.-yellow)
- CBNA: [Continual BatchNorm Adaptation (CBNA) for Semantic Segmentation] [TITS 2022] [[paper]](https://ieeexplore.ieee.org/abstract/document/9843858) [[code]](https://github.com/ifnspaml/CBNA)  ![](https://img.shields.io/badge/domain%20incre.-blue)
- ACD: [A contrastive distillation approach for incremental semantic segmentation in aerial images] [ICIAP 2022] [[paper]](https://browse.arxiv.org/pdf/2112.03814.pdf) [[code]](https://github.com/edornd/contrastive-distillation) ![](https://img.shields.io/badge/class%20incre.-yellow)
- TANet: [Class-Incremental Learning Network for Small Objects Enhancing of Semantic Segmentation in Aerial Imagery] [TGRS 2022] [[paper]](https://ieeexplore.ieee.org/abstract/document/9594782) ![](https://img.shields.io/badge/class%20incre.-yellow)
- ST-CISS: [Self-Training for Class-Incremental Semantic Segmentation] [TNNLS 2022] [[paper]](https://ieeexplore.ieee.org/abstract/document/9737321) ![](https://img.shields.io/badge/class%20incre.-yellow)
- RCIL: [Representation Compensation Networks for Continual Semantic Segmentation] [CVPR 2022] [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Representation_Compensation_Networks_for_Continual_Semantic_Segmentation_CVPR_2022_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- CAF：[Continual attentive fusion for incremental learning in semantic segmentation] [TMM 2022] [[paper]](https://ieeexplore.ieee.org/abstract/document/9757872) ![](https://img.shields.io/badge/class%20incre.-yellow)
- UCD: [Uncertainty-aware Contrastive Distillation for Incremental Semantic Segmentation] [TPAMI 2022] [[paper]](https://ieeexplore.ieee.org/abstract/document/9745778) ![](https://img.shields.io/badge/class%20incre.-yellow)
- REMINDER: [Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation] [CVPR 2022] [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Phan_Class_Similarity_Weighted_Knowledge_Distillation_for_Continual_Semantic_Segmentation_CVPR_2022_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- DFD-LM: [Class-Incremental Learning for Semantic Segmentation in Aerial Imagery via Distillation in All Aspects] [TGRS 2022] [[paper]](https://ieeexplore.ieee.org/abstract/document/9648310) ![](https://img.shields.io/badge/class%20incre.-yellow)


### 2021
- SDR: [Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations] [CVPR 2021] [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Michieli_Continual_Semantic_Segmentation_via_Repulsion-Attraction_of_Sparse_and_Disentangled_Latent_CVPR_2021_paper.html?ref=https://githubhelp.com) ![](https://img.shields.io/badge/class%20incre.-yellow)
- PLOP: [PLOP: Learning without Forgetting for Continual Semantic Segmentation] [CVPR 2021] [[paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Douillard_PLOP_Learning_Without_Forgetting_for_Continual_Semantic_Segmentation_CVPR_2021_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)


### 2020
- MiB: [Modeling the Background for Incremental Learning in Semantic Segmentation] [CVPR 2020] [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Cermelli_Modeling_the_Background_for_Incremental_Learning_in_Semantic_Segmentation_CVPR_2020_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)

### 2019
- ILT: [Incremental Learning Techniques for Semantic Segmentation] [ICCVW 2019] [[paper]](https://openaccess.thecvf.com/content_ICCVW_2019/html/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)

## <div align='center'> Data-replay Approaches </div>
### 2025
- Replay Without Saving: Prototype Derivation and Distribution Rebalance for Class-Incremental Semantic Segmentation [TPAMI 2025] [[paper]](https://ieeexplore.ieee.org/abstract/document/10904177) ![](https://img.shields.io/badge/class%20incre.-yellow)

- Deconfound Semantic Shift and Incompleteness in Incremental Few-shot Semantic Segmentation [AAAI 2025] [[paper]](https://hhudelta.github.io/publication/aaai2025/AAAI2025.pdf)  ![](https://img.shields.io/badge/class%20incre.-yellow)

### 2024
- Learning At a Glance: Towards Interpretable Data-Limited Continual Semantic Segmentation Via Semantic-Invariance Modelling. [TPAMI 2024] [[paper]](https://ieeexplore.ieee.org/document/10520832) [[code]](https://github.com/YBIO/LAG) ![](https://img.shields.io/badge/class%20incre.-yellow) ![](https://img.shields.io/badge/domain%20incre.-blue)   ![](https://img.shields.io/badge/modality%20incre.-purple) 
- Task Consistent Prototype Learning for Incremental Few-Shot Semantic Segmentation [ICPR 2024] [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-78347-0_10) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Recall-Based Knowledge Distillation for Data Distribution Based Catastrophic Forgetting in Semantic Segmentation [ICPR 2024] [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-78347-0_7) ![](https://img.shields.io/badge/domain%20incre.-blue)
- Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation. [ECCV 2024] [[paper]](https://arxiv.org/pdf/2407.13363) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Learning with Style: Continual Semantic Segmentation Across Tasks and Domains. [TPAMI 2024] [[paper]]  ![](https://img.shields.io/badge/class%20incre.-yellow) ![](https://img.shields.io/badge/domain%20incre.-blue)
- Comprehensive Generative Replay for Task-Incremental Segmentation with Concurrent Appearance and Semantic Forgetting. [MICCAI24] [[paper]](https://arxiv.org/pdf/2406.19796) [[code]](https://github.com/jingyzhang/CGR) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Incremental Model Enhancement via Memory-based Contrastive Learning. [IJCV 2024] [[paper]](https://link.springer.com/article/10.1007/s11263-024-02138-z) ![](https://img.shields.io/badge/task%20incre.-gray) 
- Comprehensive Generative Replay for Task-Incremental Segmentation with Concurrent Appearance and Semantic Forgetting [ArXiv 2024] [[paper]](https://arxiv.org/pdf/2406.19796) ![](https://img.shields.io/badge/task%20incre.-gray) 
- A Dual Enrichment Synergistic Strategy to Handle Data Heterogeneity for Domain Incremental Cardiac Segmentation [TMI 2024] [[paper]](https://ieeexplore.ieee.org/abstract/document/10433413) ![](https://img.shields.io/badge/domain%20incre.-blue)
- BACS: Background Aware Continual Semantic Segmentation. [ArXiv 2024] [[paper]](https://arxiv.org/pdf/2404.13148) ![](https://img.shields.io/badge/class%20incre.-yellow)
- MiSSNet: Memory-inspired Semantic Segmentation Augmentation Network for Class-Incremental Learning in Remote Sensing Images [TGRS 2024] [[paper]](https://ieeexplore.ieee.org/abstract/document/10418153) ![](https://img.shields.io/badge/class%20incre.-yellow)
- ConSept: Continual Semantic Segmentation via Adapter-based Vision Transformer [ArXiv 2024] [[paper]](https://arxiv.org/abs/2402.16674) ![](https://img.shields.io/badge/class%20incre.-yellow)
- TIKP: Text-to-Image Knowledge Preservation for Continual Semantic Segmentation [AAAI 2024] [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29598) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Continual atlas-based segmentation of prostate MRI. [WACV 2024] [[paper]](https://openaccess.thecvf.com/content/WACV2024/html/Ranem_Continual_Atlas-Based_Segmentation_of_Prostate_MRI_WACV_2024_paper.html) [[code]](https://github.com/MECLabTUDA/Atlas-Replay) ![](https://img.shields.io/badge/task%20incre.-gray)
### 2023
- Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation [NeurIPS 2023] [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/708e0d691a22212e1e373dc8779cbe53-Abstract-Conference.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Replaying Styles for Continual Semantic Segmentation Across Domains [ACPR 2023] [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-47637-2_23) ![](https://img.shields.io/badge/domain%20incre.-blue)
- Continual Semantic Segmentation via Scalable Contrastive Clustering and Background Diversity [ICDM 2023] [[paper]](https://ieeexplore.ieee.org/abstract/document/10415751) ![](https://img.shields.io/badge/class%20incre.-yellow)
- FairCL: [Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments] [NeurIPS 2023] [[paper]](https://arxiv.org/abs/2305.15700) ![](https://img.shields.io/badge/class%20incre.-yellow)
- DiffusePast: [DiffusePast: Diffusion-based Generative Replay for Class Incremental Semantic Segmentation] [ArXiv 2023] [[paper]](https://arxiv.org/pdf/2308.01127) ![](https://img.shields.io/badge/class%20incre.-yellow)
- FMWILSS: [Foundation Model Drives Weakly Incremental Learning for Semantic Segmentation] [CVPR 2023] [[paper]](https://arxiv.org/abs/2302.14250) ![](https://img.shields.io/badge/class%20incre.-yellow)
- AMSS: [Continual Semantic Segmentation With Automatic Memory Sample Selection] [CVPR 2023] [[paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Continual_Semantic_Segmentation_With_Automatic_Memory_Sample_Selection_CVPR_2023_paper.html) ![](https://img.shields.io/badge/class%20incre.-yellow)
- RECALL+: [Adversarial Web-based Replay for Continual Learning in Semantic Segmentation] [ArXiv 2023] [[paper]](https://browse.arxiv.org/pdf/2309.10479) ![](https://img.shields.io/badge/class%20incre.-yellow)
- EndoCSS: [Rethinking exemplars for continual semantic segmentation in endoscopy scenes: Entropy-based mini-batch pseudo-replay] [CBM 2023] [[paper]](https://browse.arxiv.org/pdf/2308.14100) ![](https://img.shields.io/badge/class%20incre.-yellow)
- Domain-Incremental Cardiac Image Segmentation With Style-Oriented Replay and Domain-Sensitive Feature Whitening [TMI 2023] [[paper]](https://ieeexplore.ieee.org/document/9908146) ![](https://img.shields.io/badge/domain%20incre.-blue)
- GarDA: [Generative appearance replay for continual unsupervised domain adaptation] [Medical Image Analysis 2023] [[paper]](https://arxiv.org/abs/2301.01211) ![](https://img.shields.io/badge/domain%20incre.-blue)

### 2022
- ProCA: [Prototype-guided continual adaptation for class-incremental unsupervised domain adaptation] [ECCV 2022] [[paper]](https://browse.arxiv.org/pdf/2207.10856.pdf) [[code]](https://github.com/Hongbin98/ProCA)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- SPPA: [Continual Semantic Segmentation via Structure Preserving and Projected Feature Alignment] [ECCV 2022] [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890341.pdf) [[code]](https://github.com/AsteRiRi/SPPA)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- ALIFE: [Alife: Adaptive logit regularizer and feature replay for incremental semantic segmentation] [NeurIPS 2022] [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d516fc09b53e9a7fade4fbad703e686-Abstract-Conference.html) [[code]](https://github.com/cvlab-yonsei/ALIFE)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- DKD: [Decomposed knowledge distillation for class-incremental semantic segmentation] [NeurIPS 2022] [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/439bf902de1807088d8b731ca20b0777-Abstract-Conference.html) [[code]](https://github.com/cvlab-yonsei/DKD) ![](https://img.shields.io/badge/class%20incre.-yellow) 
- MicroSeg: [Mining Unseen Classes via Regional Objectness: A Simple Baseline for Incremental Segmentation] [NeurIPS 2022] [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/99b419554537c66bf27e5eb7a74c7de4-Paper-Conference.pdf) [[code]](https://github.com/zkzhang98/MicroSeg) ![](https://img.shields.io/badge/class%20incre.-yellow) 
- CCDA: [Continual coarse-to-fine domain adaptation in semantic segmentation] [IVC 2022] [[paper]](https://doi.org/10.1016/j.imavis.2022.104426) ![](https://img.shields.io/badge/domain%20incre.-blue)
- Continual Adaptation of Semantic Segmentation Using Complementary 2D-3D Data Representations [LRA 2022] [[paper]](https://ieeexplore.ieee.org/abstract/document/9874976) ![](https://img.shields.io/badge/domain%20incre.-blue)
- Improving replay-based continual semantic segmentation with smart data selection [ITSC 2022] [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9922284) ![](https://img.shields.io/badge/domain%20incre.-blue)
- SIL-LAND: Segmentation Incremental Learning in Aerial Imagery via LAbel Number Distribution Consistency [TGRS 2022] [[paper]](https://ieeexplore.ieee.org/document/9858901) ![](https://img.shields.io/badge/class%20incre.-yellow) 

### 2021
- SSUL: [SSUL: Semantic Segmentation with Unknown Label for Exemplar-based Class-Incremental Learning] [NeurIPS 2021] [[paper]](https://proceedings.neurips.cc/paper/2021/hash/5a9542c773018268fc6271f7afeea969-Abstract.html) [[code]](https://github.com/clovaai/SSUL)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- RECALL: [RECALL: Replay-based Continual Learning in Semantic Segmentation] [ICCV 2021] [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Maracani_RECALL_Replay-Based_Continual_Learning_in_Semantic_Segmentation_ICCV_2021_paper.pdf) [[code]](https://github.com/LTTM/RECALL)  ![](https://img.shields.io/badge/class%20incre.-yellow)
- FSS: [Prototype-based Incremental Few-Shot Semantic Segmentation] [BMVC 2021] [[paper]](https://arxiv.org/abs/2012.01415) [[code]](https://github.com/fcdl94/FSS)  



## <div align='center'> Few-shot CSS </div>
- SRAA: [Advancing Incremental Few-shot Semantic Segmentation via Semantic-guided Relation Alignment and Adaptation] [MMM 2024] [[paper]](https://arxiv.org/abs/2305.10868)
- FSCILSS: [Few-Shot Class-Incremental Semantic Segmentation via Pseudo-Labeling and Knowledge Distillation] [ISPDS 2023] [[paper]](https://ieeexplore.ieee.org/abstract/document/10235731)
- GAPS: [GAPS: Few-Shot Incremental Semantic Segmentation via Guided Copy-Paste Synthesis] [CVPRW 2023] [[paper]](https://motion.cs.illinois.edu/papers/CVPRWorkshop2023-Qiu-FewShotSegmentation.pdf)
- EHNet: [Incremental Few-Shot Semantic Segmentation via Embedding Adaptive-Update and Hyper-class Representation] [ACM MM 2022] [[paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3548218)
- PIFS: [Prototype-based Incremental Few-Shot Semantic Segmentation] [BMVC 2021] [[paper]](https://arxiv.org/abs/2012.01415)

## <div align='center'> Specific Applications </div>
- Towards Realistic Incremental Scenario in Class Incremental Semantic Segmentation [ArXiv 2024] [[paper]](https://arxiv.org/pdf/2405.09858)
- UnCLe SAM: Unleashing SAM’s Potential for Continual Prostate MRI Segmentation [MIDL 2024] [[paper]](https://openreview.net/forum?id=jRtUQ2VnNi)
- Cross-Domain Few-Shot Incremental Learning for Point-Cloud Recognition [WACV 2024] [[paper]](https://openaccess.thecvf.com/content/WACV2024/papers/Tan_Cross-Domain_Few-Shot_Incremental_Learning_for_Point-Cloud_Recognition_WACV_2024_paper.pdf)
- Attacks on Continual Semantic Segmentation by Perturbing Incremental Samples. [AAAI 2024] [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28509)
- Federated Incremental Semantic Segmentation. [CVPR 2023] [[paper]](https://arxiv.org/abs/2304.04620) [[code]](https://github.com/JiahuaDong/FISS)
- Principles of Forgetting in Domain-Incremental Semantic Segmentation in Adverse Weather Conditions [CVPR 2023] [[paper]](https://arxiv.org/abs/2303.14115)
- Continual segment: Towards a single, unified and accessible continual segmentation model of 143 whole-body organs in ct scans. [ICCV 2023] [[paper]](https://arxiv.org/abs/2302.00162)
- Continual Nuclei Segmentation via Prototype-wise Relation Distillation and Contrastive Learning. [TMI 2023] [[paper]](https://ieeexplore.ieee.org/abstract/document/10227350)
- Continual learning for abdominal multi-organ and tumor segmentation. [MICCAI 2023] [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_4)

## <div align='center'> Reproductions </div>
### Class-incre. CSS Performance 
The evaluation of anti-forgetting ability and robustness of CSS methods.
![Performance](illustration/performance.png)
### Class-incre. CSS Visualization
Here are some qualitative visualizations of recent CSS methods.
![task_legend](illustration/visualization.png)

### Interpretability Analysis
Feature-based visualization is an explicit way to visualize the semantic clusters during CL steps. For example, TSNE and CAM-series are potential manners to assist the result analysis.

### Qualitative Analysis
![qualitative_analysis](illustration/analysis.png)

### Cite this repo
If this project is helpful, please consider citing it as
```
@ARTICLE{SurveyCSS,
  author={Yuan, Bo and Zhao, Danpei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Survey on Continual Semantic Segmentation: Theory, Challenge, Method and Application}, 
  year={2024},
  volume={},
  number={},
  pages={1-20},
  doi={10.1109/TPAMI.2024.3446949}}
```
## Related Paper
- Brain-inspired Continual Semantic Segmentation [[paper]](https://ieeexplore.ieee.org/abstract/document/10520832/) | [[blog]](https://ybio.github.io/2024/06/01/blog_LAG/)
- Inherit With Distillation and Evolve With Contrast [[paper]](https://arxiv.org/pdf/2309.15413)
- Continual Panoptic Perception [[paper]](https://arxiv.org/abs/2407.14242)

## TODO
&#9745;  Method characteristics summary

&#9745; Performance comparison

&#9745; Roadmap update

