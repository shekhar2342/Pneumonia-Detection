from helper.libraries import *
from helper import functions

# Mayank Parihar

# App Title
st.title("Pneumodetector APP")

# Introduction text
st.markdown(unsafe_allow_html=True, body="<p>Welcome to Pneumodetector APP.</p>")

st.markdown("First, let's load an X-Ray Chest image.")

# Loading model

# Img uploader
img = st.file_uploader(label="Load X-Ray Chest image")

if img is not None:
    # Preprocessing Image
    p_img = functions.preprocess_image(img)

    # Loading model
    loading_msg = st.empty()
    loading_msg.text("Predicting...")
    model = functions.input_model()

    # Predicting result
    prob, prediction = functions.predict(model, p_img)

    loading_msg.text('')

    if prediction:
        st.markdown(unsafe_allow_html=True,
                    body="<span style='color:red; font-size: 50px'><strong><h4>Pneumonia! :slightly_frowning_face:</h4></strong></span>")
    else:
        st.markdown(unsafe_allow_html=True,
                    body="<span style='color:green; font-size: 50px'><strong><h3>Healthy! :smile: </h3></strong></span>")

    st.text(f"*Probability of pneumonia is {round(prob * 100, 2)}%")
