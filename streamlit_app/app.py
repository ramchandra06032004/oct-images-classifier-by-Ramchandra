from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model as keras_load_model

import streamlit as st
import plotly.express as px
import pickle 
import os


def load_model():
    # Load the trained model
    model_path = "../ModelTraining/best_model.h5"
    if os.path.exists(model_path):
        model = keras_load_model(model_path)
    else:
        model = None
    return model

model = load_model()

if model is None:
    raise FileNotFoundError("The model file was not found. Please check the model path and ensure the model is trained and saved correctly.")



# Function to load the history from a .pkl file
def load_history(path):
    with open(path, 'rb') as file:
        history = pickle.load(file)
    return history

# Load the history
history_path = "../ModelTraining/Raw_training_history.pkl"
history = load_history(history_path)

df=pd.DataFrame(history)
df=df.rename_axis('epochs').reset_index()
# Set the title and the sidebar menu options
st.sidebar.title("OCT Scanned Images Classifier")
user_menu = st.sidebar.radio(
    'Select an option',
    ("AI Model", "Educational Resources", "About AI Model")
)

# Image preprocessing
def preprocess_image(img):
    # Resize the image to 224x224
    img = img.resize((224, 224))
    # Convert the image to RGB if it is grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Convert the image to a numpy array
    img_array = keras_image.img_to_array(img)
    # Normalize the image
    img_array = tf.cast(img_array / 255., tf.float32)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
#ai model 
if user_menu == "AI Model":
    st.title("AI Based OCT Scanned Images Classifier")
    st.write("Our Optical Coherence Tomography (OCT) Image Classifier leverages the power of Convolutional Neural Networks (CNNs) to accurately classify OCT scans. This cutting-edge tool is designed to assist medical professionals in diagnosing retinal diseases with high precision.")
    st.header("How it Works")
    st.write("Upload Image: Simply upload the OCT scan image.") 
    st.write("Analyze: Our CNN model processes the image to detect and classify retinal conditions.") 
    st.write("Results: View detailed results, including the predicted class and confidence level.")

    # Input Section
    st.header("Upload OCT Scan Image")
    uploaded_file = st.file_uploader("Choose an OCT scan image (JPG format)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Process the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded OCT scan image", use_column_width=True)
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img)
        
        # Process predictions (depending on your model's output)
        # For example, if your model outputs class probabilities:
        predicted_class = np.argmax(predictions, axis=1)
        if predicted_class==0:
            #CNV (Choroidal Neovascularization)
            st.write("Predicted Class: CNV (Choroidal Neovascularization)")
            st.write("surety Level: {:.2f}%".format(predictions[0][0]*100))
        elif predicted_class==1:
            #DME (Diabetic Macular Edema):
            st.write("Predicted Class: DME (Diabetic Macular Edema)")
            st.write("surety Level: {:.2f}%".format(predictions[0][1]*100))
        elif predicted_class==2:
            #DRUSEN
            st.write("Predicted Class: DRUSEN")
            st.write("surety Level: {:.2f}%".format(predictions[0][2]*100))
        else:
            #NORMAL
            st.write("Predicted Class: NORMAL")
            st.write("surety Level: {:.2f}%".format(predictions[0][3]*100))

#Educational Resources
if user_menu == "Educational Resources":
    st.title("Educational Resources")
    st.title("OCT Scan (Optical Coherence Tomography)")
    st.header("What is an OCT Scan?")
    
    st.write("Optical Coherence Tomography (OCT) is a non-invasive imaging test that uses light waves to take cross-sectional pictures of the retina, the light-sensitive tissue lining the back of the eye. It provides detailed images of the retina's layers, helping in the diagnosis and management of various eye conditions.")
    
    st.header("Common Retinal Diseases Detected by OCT")
    st.write("1. CNV (Choroidal Neovascularization)")
    st.write("2. DME (Diabetic Macular Edema)")
    st.write("3. DRUSEN")
    #cnv
    st.header("information of CNV (Choroidal Neovascularization)")
    st.write("Formation of New Blood Vessels: CNV involves the abnormal growth of new blood vessels in the choroid layer beneath the retina.")
    st.write("Penetration: These new blood vessels can penetrate the retinal pigment epithelium and grow into the retina.")
    st.write("Associated Conditions: CNV is most commonly associated with age-related macular degeneration (AMD), but can also occur in other conditions such as myopic degeneration and ocular histoplasmosis.")
    st.write("Leakage: The new vessels are often fragile and prone to leaking blood and fluid.")
    st.write("Symptoms: Symptoms of CNV may include blurred or distorted central vision, straight lines appearing wavy, and dark or empty areas in the center of vision")

    #dme
    st.header("information of DME (Diabetic Macular Edema)")
    st.write("Diabetic Complication: DME is a common complication of diabetic retinopathy, which occurs in patients with diabetes.")
    st.write("Fluid Accumulation: Fluid leaks from damaged blood vessels into the macula, causing swelling.")
    st.write("Macula: The macula is responsible for sharp, detailed central vision.")
    st.write("Blood Vessel Damage: High blood sugar levels damage retinal blood vessels over time.")
    st.write("Symptoms: Symptoms of DME may include blurred or distorted central vision, difficulty reading, and color perception changes.")

    #drusen
    st.header("information of DRUSEN")
    st.write("Yellow Deposits: Drusen are yellow deposits that form beneath the retina.")
    st.write("Age-related: Commonly found in individuals over the age of 60.")
    st.write("Types: There are two types: hard drusen (small and well-defined) and soft drusen (larger and less distinct).")
    st.write("Accumulation: Drusen accumulation is part of the aging process but can signify early stages of AMD.")
    st.write("Symptoms: Drusen may not cause symptoms in the early stages, but larger or more numerous drusen can lead to vision changes.")

    #advantages of OCT
    st.header("Advantages of OCT")
    st.write("Non-Invasive: OCT is a non-invasive imaging technique that does not require contact with the eye.")
    st.write("High Resolution: It provides high-resolution cross-sectional images of the retina.")
    st.write("Early Detection: OCT can detect retinal changes at an early stage, allowing for timely intervention.")
    st.write("Monitoring: It is useful for monitoring disease progression and treatment response over time.")
    st.write("Objective Data: OCT provides objective data for diagnosis and treatment planning.")
    st.write("Patient Comfort: Patients find OCT scans comfortable and quick, with no discomfort or side effects.")
    st.write("Cross-Sectional Views: OCT can image multiple layers of tissue by providing cross-sectional views, which helps in assessing the depth and structure of abnormalities.")
    st.write("Research Applications: OCT is widely used in research to study retinal diseases and develop new treatment strategies.")

if user_menu == "About AI Model":
    st.title("About AI Model")
    st.header("Model Architecture")
    st.write("We replicated the ResNet-50 architecture and successfully trained it on a dataset of 85,000 images. This deep learning model demonstrated high accuracy in classifying complex patterns within the data, showcasing its robustness and effectiveness.")
    st.image("modelArchirecture.png", caption="Last Layers of ResNet-50 Architecture",width=500)

    st.header("Model Training Details")
    st.write("Training Duration: The model requires approximately 5 hours for training on a GPU, utilizing a dataset of 83,484 images.")
    st.write("Optimizer Used: Adam")
    st.write("Loss Function: Categorical Crossentropy")
    st.write("Number of Epochs: 17")
    st.write("Accuracy Achieved: 99.59% (More than human accuracy)")
    #accuracy image
    st.image("accuracy.png", caption="Highst accuracy we reach",width=800)

    #plotting the training history


    accuracy=df["accuracy"]*100
    val_accuracy=df["val_accuracy"]*100

    fig = px.line(df, x='epochs', y=accuracy, title='Epochs v/s Training Accuracy & Validation Accuracy in percent' )

    # Add a line plot for loss using add_trace
    fig.add_trace(
        px.line(df, x='epochs', y=val_accuracy).data[0]
    )

    # Show the plot
    fig.update_layout(
        width=800,  # Width of the figure in pixels
        height=600 ,
    )
    st.plotly_chart(fig)

    loss=df["loss"]*100
    val_loss=df["val_loss"]*100
    fig = px.line(df, x='epochs', y=loss, title='Epochs v/s  Training Loss & Validation Loss in percent' )

    # Add a line plot for loss using add_trace
    fig.add_trace(
        px.line(df, x='epochs', y=val_loss).data[0]
    )

    # Show the plot
    fig.update_layout(
        width=800,  # Width of the figure in pixels
        height=600 ,
    )
    st.plotly_chart(fig)

    st.write("download images for testing the model")
    link = 'https://drive.google.com/drive/folders/15PvB_QC5_zv7hdg_SuJflEP6uHK8m9g4?usp=drive_link'
    st.markdown(f'''
    <a href="{link}" style="text-decoration: none;">
        <button style="background-color: red; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer; text-align: center;">
            Download
        </button>
    </a>
    ''', unsafe_allow_html=True)
    st.write("For More Images for testing the model, you can download the dataset from the link below")
    st.write("https://www.kaggle.com/datasets/paultimothymooney/kermany2018")
    

# Footer section
footer = """
<style>
footer {
    visibility: hidden;
}
.main-footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    text-align: left;
    padding: 5px;
    font-size: 14px;
    color: white;
}
</style>
<div class="main-footer">
    <p>Developed by Neural Navigators | &copy; 2024 All Rights Reserved</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)