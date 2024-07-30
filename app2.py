import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import replicate
import io
import base64
import time
import numpy as np
import requests

st.set_page_config(layout="wide")
st.title("Drawing-2-AI")

# Layout with two columns
col1, col2 = st.columns(2)

# Initialize session state variables
if 'canvas_image' not in st.session_state:
    st.session_state['canvas_image'] = None
if 'prev_canvas_image' not in st.session_state:
    st.session_state['prev_canvas_image'] = None
if 'run_api_call' not in st.session_state:
    st.session_state['run_api_call'] = False

# Drawing mode selection
tools = {
    "Freedraw": "freedraw",
    "Line": "line",
    "Rectangle": "rect",
    "Circle": "circle",
    "Polygon": "polygon",
}

with col1:
    # Sidebar selectbox using the dictionary keys
    st.subheader("Draw anything on the canvas below")
    selected_label = st.sidebar.selectbox("Drawing tool:", list(tools.keys()))

    # Get the corresponding tool from the dictionary
    drawing_mode = tools[selected_label]
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = '#000000'
    bg_color = '#FFFFFF'
    bg_image = None
    realtime_update = True
    point_display_radius = 1

    quality_mapping = {
        "Low (Fastest)": 5,
        "Medium (Fast)": 7,
        "High (Slow)": 12,
        "Ultra (Slowest)": 20
    }

    art_styles = {
        "Painting": "a detailed oil painting of a",
        "Photo": "a photo of a",
        "Cartoon": "a cartoon drawing of a"
    }

    quality = st.sidebar.radio("Image Quality", list(quality_mapping.keys()), index=1)
    quality_value = quality_mapping[quality]

    selected_art_style = st.sidebar.selectbox("Art Style:", list(art_styles.keys()))
    art_style_prompt = art_styles[selected_art_style]

    canvas_size = 512  # Setting both width and height to 512 to make the canvas square
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        width=canvas_size,
        height=canvas_size,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius,
        display_toolbar=True,
        key="full_app",
    )

def encode_image(img_bytes):
    return base64.b64encode(img_bytes).decode('utf-8')

def describe_image(img_bytes, api_key):
    # Encode the image
    base64_image = encode_image(img_bytes)

    # Set up the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Set up the payload
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what the object(s) in this drawing looks like in a concise manner."
                                "Only output the object(s), nothing else, no descriptors, just try to see what it is."
                                "Don't ever say it's a drawing. No punctuation. Concise and comma separated. Never say stick figure, always"
                                " assume a form, like a human, animal, or other object. Also add in composition details if applicable."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 20
    }

    # Make the request
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Return the response
    return response.json()['choices'][0]['message']['content']

def generate_ai_image(img_base64, description):
    rep_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    print(description)
    output = rep_client.run(
        "jagilley/controlnet-scribble:435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117",
        input={
            "eta": 0,
            "image": f"data:image/png;base64,{img_base64}",
            "scale": 9,
            "prompt": f"{description} {art_style_prompt}, masterpiece, perfection",
            "a_prompt": "best quality, extremely detailed",
            "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
            "ddim_steps": quality_value,
            "num_samples": "1",
            "image_resolution": "512"
        }
    )
    return output

# Check if canvas image has changed
def canvas_changed(canvas_image_data):
    if st.session_state.prev_canvas_image is None or not np.array_equal(canvas_image_data, st.session_state.prev_canvas_image):
        st.session_state.prev_canvas_image = canvas_image_data.copy()
        st.session_state.run_api_call = True
    else:
        st.session_state.run_api_call = False

if canvas_result.image_data is not None:
    canvas_image_data = canvas_result.image_data.astype("uint8")
    canvas_changed(canvas_image_data)

# Delay and perform the API call
if st.session_state.run_api_call:
    st.session_state.run_api_call = False
    # Wait for 0.5 seconds before sending the image to the model
    time.sleep(0.5)

    # Convert the canvas image data to a PNG image in memory
    img = Image.fromarray(st.session_state.prev_canvas_image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Encode image bytes to base64
    img_base64 = encode_image(img_bytes)

    # Get description of the image
    api_key = OPENAI_API_KEY  # Replace with your actual API key
    description = describe_image(img_bytes, api_key)
    # Display the description
    if description is not None:
        with col1:
            st.markdown(f"### {description}")

    output = generate_ai_image(img_base64, description)

    if output:
        generated_image_url = output[1]
        with col2:
            st.subheader("AI Image")
            st.image(generated_image_url)
    else:
        st.write("Error in generating image")
