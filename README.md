<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

<div align="center">
  
## 🌿 EcoMedVision: CNN-Based Medical vs Recyclable Waste Classification

</div>

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

**EcoMedVision** is an AI-driven sustainability project that uses **Convolutional Neural Networks (CNNs)** to automatically classify waste generated in healthcare environments into **medical** and **recyclable** categories.
Improper segregation of hospital waste can lead to serious **health hazards** and **environmental pollution** — mixing infectious items like syringes or gloves with recyclable plastics increases contamination and landfill load.
This project aims to make waste management in healthcare **smarter, safer, and more sustainable** — entirely through a software solution, without requiring any extra hardware.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### 🧠 What It Does

* Takes an **image of waste** (captured from datasets or uploaded by the user).
* Predicts whether the item belongs to:

  * 🩺 **Medical Waste** (e.g., syringes, gloves, masks, biohazard bags)
  * ♻️ **Recyclable Waste** (e.g., plastic bottles, paper cups, cardboard)
* Uses a deep learning model built with **CNN / Transfer Learning (MobileNetV2 or ResNet50)**.
* Provides a simple **web interface** (Flask or Streamlit) for image upload and instant classification results.
* Promotes **sustainability** by reducing human error in waste segregation.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### ⚙️ Tech Stack

* **Language:** Python 🐍
* **Frameworks:** TensorFlow / Keras, OpenCV, NumPy, Pandas, Matplotlib
* **Model Architecture:** Custom CNN or Transfer Learning (MobileNetV2 / ResNet50 / EfficientNet)
* **Frontend:** Streamlit / Flask (upload → classify → visualize)
* **Environment:** Google Colab / Kaggle Notebook (GPU-ready, no hardware required)

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### 📦 Datasets Used

| Dataset                                  | Description                                                           | Link                                                                                                                   |
| ---------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 🗑️ **TrashNet**                         | Classic dataset of recyclable materials (plastic, paper, metal, etc.) | [Kaggle: TrashNet Dataset](https://www.kaggle.com/datasets/garythung/trashnet)                                         |
| 🧴 **Pharmaceutical & Biomedical Waste** | Dataset of medical waste items (gloves, masks, syringes)              | [Kaggle: Biomedical Waste Dataset](https://www.kaggle.com/datasets/engineeringubu/pharmaceutical-and-biomedical-waste) |
| ♻️ **Waste Classification Data**         | Large dataset with recyclable vs non-recyclable labels                | [Kaggle: Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)                |

Images from these datasets are merged and relabeled into two classes — **Medical Waste** and **Recyclable Waste** — for binary classification.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### 🧩 Model Workflow

1. **Data Preprocessing:**
   Image resizing, normalization, and augmentation (rotation, zoom, flips).
2. **Model Training:**
   CNN / transfer-learning architecture trained with categorical cross-entropy loss.
3. **Evaluation:**
   Model accuracy, confusion matrix, precision, recall, and F1-score.
4. **Visualization:**
   Grad-CAM heatmaps show what the model focused on for transparency.
5. **Deployment:**
   Streamlit/Flask app for real-time image classification.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### 🌍 Impact

* Encourages **sustainable waste segregation** in healthcare.
* Reduces the **risk of biohazard contamination**.
* Minimizes **manual labor and misclassification errors**.
* Offers a **cost-free, software-only solution** — easy to integrate with hospital systems.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### ❓ Why EcoMedVision? 🌱

- Hospitals and clinics produce mixed streams of waste where infectious and recyclable items are often combined, increasing contamination risk and disposal costs.
- EcoMedVision helps reduce human error and improves waste segregation accuracy using a lightweight, software-only approach that can be integrated into existing workflows.
- By automating classification, we protect staff, reduce environmental harm, and lower disposal and recycling costs.

### 🌟 Vision

To be the go-to AI assistant for safer, greener healthcare waste management — enabling hospitals worldwide to reduce contamination, increase recycling, and improve public health through intelligent imaging and classification. 🌍💚

### 🎯 Mission

1. Build an accurate, interpretable, and easy-to-deploy image classification system that separates medical waste from recyclables.
2. Provide clear visual explanations (Grad-CAM) so staff trust model decisions and can audit predictions.
3. Make the solution accessible (open-source demos and Streamlit/Flask apps) so even low-resource facilities can adopt the tool.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### 💡 Future Enhancements

* Multi-class classification (e.g., *Infectious*, *Non-infectious*, *Recyclable*, *Organic*).
* Integration with real-time CCTV feeds for automatic detection.
* Confidence-based alerts (flag low-confidence predictions for human review).
* Lightweight model quantization for edge deployment.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

### 🏁 Project Status

🚧 **In progress** — Data preprocessing and baseline CNN model training under development.
Next milestone: Streamlit app deployment & Grad-CAM visualization integration.

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

