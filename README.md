# FYP
Project Description:
In this final year project, we developed a smart agricultural solution aimed at detecting and visualizing disease-affected regions in potato plants using drone-captured imagery. The primary focus of the system is the identification of Late Blight, a major disease that severely impacts potato crop yields.

To achieve this, a YOLOv8-Seg (segmentation variant of YOLOv8) deep learning model was trained on annotated drone images of potato leaves. Unlike traditional classification or object detection models, YOLOv8-Seg is capable of performing instance segmentation, which allowed us to precisely highlight the damaged regions on the leaves pixel by pixel.

Key steps in the project included:

Data Collection: Aerial images of potato crops were collected using a drone equipped with a high-resolution RGB camera.

Preprocessing and Annotation: Images were preprocessed (resized, oriented, and cleaned), and regions affected by Late Blight were annotated to train the model effectively.

Model Training: YOLOv8-Seg was trained on the labeled dataset to learn the visual patterns of disease-affected areas.

Detection and Visualization: After training, the model was able to accurately detect and segment damaged leaf areas in real-time. The segmented outputs visually highlight the diseased parts, helping farmers take timely action.

This system supports precision agriculture by enabling scalable, fast, and non-invasive disease monitoring across large fields using drone technology and deep learning
