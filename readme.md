# OCT Scan Image Classifier

This repository contains a Streamlit application for classifying OCT scan images using a deep learning model trained on a large dataset.

## Model Details

- **Model Architecture**: ResNet50
- **Accuracy**: 99.59%
- **Training Data**: 84,000 OCT scan images

Watch the demo video of the application on YouTube:

[![Watch the video](https://img.youtube.com/vi/VVL_f9Gqpt0/0.jpg)](https://www.youtube.com/watch?v=VVL_f9Gqpt0)

## Requirements

Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/ramchandra06032004/OCT-images-classifier.git
    cd OCT-images-classifier
    ```

2. Create a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

4. Install the required packages individually:

    - Pillow:

        ```sh
        pip install Pillow
        ```

    - NumPy:

        ```sh
        pip install numpy
        ```

    - Pandas:

        ```sh
        pip install pandas
        ```

    - TensorFlow:

        ```sh
        pip install tensorflow
        ```

    - Streamlit:

        ```sh
        pip install streamlit
        ```

    - Plotly:

        ```sh
        pip install plotly
        ```

    - Scikit-learn:

        ```sh
        pip install scikit-learn
        ```


    

## Running the Application

Before running the application, download the trained model from Google Drive (link provided below) and place it in the `ModelTraining` folder.

[Download Trained Model](https://drive.google.com/drive/folders/1FnZPPkulSNw1NfOtFo_uPk8-MDXilno7?usp=sharing)


1. Navigate to the `streamlit_app` directory:

    ```sh
    cd streamlit_app
    ```

2. Run the Streamlit app:

    ```sh
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` to view the app.

## Acknowledgements

- [Streamlit](https://www.streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pillow](https://python-pillow.org/)
- [NumPy](https://numpy.org/)

## Highlights

- **High Accuracy**: Our model achieves an impressive 99.59% accuracy in classifying OCT scan images.
- **Robust Architecture**: Utilizes the powerful ResNet50 architecture, ensuring efficient and accurate image classification.
- **Extensive Training Data**: Trained on a comprehensive dataset of 84,000 OCT scan images, enhancing the model's reliability and performance.
