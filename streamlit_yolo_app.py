import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
from io import BytesIO
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

# Initialize the YOLOv8 model
model = YOLO("models/yolov8s.pt")

st.set_page_config(
    page_title="Object Detection App",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("Object Detection App")

    st.sidebar.markdown(
        """
        Welcome to the Object Detection App! This app allows you to perform object detection on images 
        using YOLOv8 model. Choose one of the options below to get started.
        """
    )

    choice = st.sidebar.radio("Select an option", ("Upload an image", "Use webcam", "Provide image URL"))

    if choice == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            results = model(source=img)
            annotated_img = results[0].plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            col1.image(img, caption="Uploaded Image", use_column_width=True)
            col2.image(annotated_img_rgb, caption="Predicted Image", use_column_width=True)

    elif choice == "Use webcam":
        client_settings = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

        class ObjectDetector(VideoTransformerBase):
            def transform(self, frame):
                img = Image.fromarray(frame.to_ndarray())
                results = model(source=img)
                annotated_img = results[0].plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, caption="Predicted Image", use_column_width=True)

        webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            client_settings=client_settings,
            video_transformer_factory=ObjectDetector,
        )

    elif choice == "Provide image URL":
        image_url = st.text_input("Enter the image URL:")

        if image_url != "":
            try:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                results = model(source=img)
                annotated_img = results[0].plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                col1.image(img, caption="Downloaded Image", use_column_width=True)
                col2.image(annotated_img_rgb, caption="Predicted Image", use_column_width=True)

            except:
                st.error("Error: Invalid image URL or unable to download the image.")

if __name__ == '__main__':
    main()
