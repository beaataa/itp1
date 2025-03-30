# ITP1 - EV Battery Component Detection with Deep Learning

**ITP Team 9**  
**Objective**: Automate the detection and segmentation of EV battery components using deep learning for safer, faster disassembly in recycling workflows.

---

### Overview

This project focuses on detecting and segmenting key EV battery components (e.g., bolts, busbars, cables) using semantic and instance segmentation models like **YOLOv8** and **DeepLabV3+**.  
Our pipeline is designed to support integration with **Mech-Mind industrial cameras** and is optimized for future 3D localization (for ITP2).

---

### Solution Process Flow
1. Data Collection
2. Data Annotation / Labeling
3. Data Preprocessing & Augmentation
4. Data Preparation
5. Model Training
6. Performance Evaluation
7. Results & Insights

---

## 🧪 Model Performance

### YOLOv8
- **mAP@0.5**: 0.589
- Best Classes: Bolt (0.758), Busbar (0.755)
- Challenging Classes: Nut (0.417), Cable (0.429)
- Inference Time: ~5s (Tesla P100)

### DeepLabV3+
- **Mean IoU**: 81.97%
- Best Class: Plastic Film (90.8% IoU)
- Lowest: Plastic Cover (64.8% IoU)

---

Team Members:
- Fun Kai Jun
- Heng Yu Xin
- Ng Wei Herng
- Yeo Ya Xuan Beata

Supervisors:
Ang Jia Yuan, Lou Xin, Miao Xiao Xiao
