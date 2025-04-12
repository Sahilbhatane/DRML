from openai import OpenAI
import streamlit as st
from PIL import Image
import base64
import time
import os
import uuid
import retry
import random
import google.generativeai as genai
from io import BytesIO

from prompt_utils import encode_image, get_custom_agent_system_message, get_custom_agent_user_message, get_direct_system_message, get_direct_text_augmented_user_message, get_user_augmented_message_with_qa_pairs
from utils import extract_html, extract_all_questions, remove_html_comments, extract_title, read_html, extract_text_from_html, rescale_image_loader, cleanup_response
from screenshot import take_screenshot_from_html, take_screenshot


model = 'gemini-1.5-pro'


@retry.retry(tries=3, delay=2)
def generate_questions(client, sketch, description, num_questions=5, focus="any"):
    # Get user message with the sketch encoded properly
    agent_user_message = get_custom_agent_user_message(sketch, description, num_questions, focus)
    system_message = st.session_state.agent_system_message
    
    # Get encoded version of the sketch
    encoded_sketch = encode_image(sketch)
    
    # Convert image to Gemini format
    image_parts = [{"mime_type": "image/jpeg", "data": encoded_sketch}]
    
    response = client.generate_content(
        contents=[
            {"role": "user", "parts": [{"text": system_message}]},
            {"role": "user", "parts": [{"text": agent_user_message[0]["text"]}, *image_parts]}
        ],
        generation_config={"temperature": 0.0, "max_output_tokens": 4096}
    )
    
    agent_resp = response.text.strip()
    questions = extract_all_questions(agent_resp)
    
    assert len(questions) >= num_questions, f"Expected {num_questions} questions, but got {len(questions)}"
    
    if len(questions) > num_questions:
        questions = questions[:num_questions]
    
    return questions

@retry.retry(tries=3, delay=2)
def generate_webpage(client, messages):
    # Convert OpenAI message format to Gemini format
    gemini_messages = []
    
    for message in messages:
        role = message["role"]
        if "content" in message and isinstance(message["content"], list):
            # Multi-part message with text and image
            parts = []
            for content_part in message["content"]:
                if content_part["type"] == "text":
                    parts.append({"text": content_part["text"]})
                elif content_part["type"] == "image_url":
                    image_url = content_part["image_url"]["url"]
                    if image_url.startswith("data:image/jpeg;base64,"):
                        image_data = image_url.split(",")[1]
                        parts.append({"mime_type": "image/jpeg", "data": image_data})
            gemini_messages.append({"role": "user" if role == "user" else "model", "parts": parts})
        else:
            # Text-only message
            content = message["content"]
            gemini_messages.append({"role": "user" if role == "user" else "model", "parts": [{"text": content}]})
    
    response = client.generate_content(
        contents=gemini_messages,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096}
    )
    
    agent_resp = response.text.strip()
    html_response = cleanup_response(extract_html(agent_resp))
    
    if html_response:
        return html_response, agent_resp, True
    else:
        return html_response, agent_resp, False


def get_response_message():
    return random.choice(
        [
            "Here is a revised implementation. Let me know what you think!",
            "I've made some adjustments according to your feedback. Would you like to make any further changes?",
            "I've updated the design based on your feedback. What do you think?",
        ]
    )


def run():
    # Hide Streamlit's default menu and deploy button
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        api_key = st.text_input("Google API Key", key="google_api_key", type="password", 
                               help="Enter your Google API key. Get one for free at https://makersuite.google.com/app/apikey")
        
        num_questions = st.number_input("How many questions should the agent ask?", min_value=1, max_value=5, value=1, help="Choose how many questions the agent should ask about the sketch design before generating a prototype.")

        question_focus = st.selectbox("What types of questions would you like the agent to focus on?", options=["any", "style", "layout", "text content", "design details"], help='Choose the type of questions for the agent to focus on, or select "any" to let the agent decide.')

        additional_feedback = st.checkbox("Additional customization after the agent has asked all questions?", value=True, help="Check this box if you would like to provide additional feedback or instructions to further customize the generated prototype after the agent has asked all questions.")

    # Main title and caption
    st.title("💬 DRML Drawable Agent")
    
    st.markdown("""
    <div style='font-size: 18px; margin-bottom: 20px;'>
        🚀 In this mode, the DRML Agent will first proactively ask questions regarding the design details of the sketch. You can control how many questions it asks, and steer the focus of these questions. Once the agent has acquired all the information it needs, it will attempt to generate a HTML prototype based on the sketch you provide. You may then optionally choose to provide additional instructions to further customize the prototype.
    </div>
    """, unsafe_allow_html=True)

    # Customize the second caption with different font size and spacing
    st.markdown("""
    <div style='font-size: 16px; color: gray; margin-top: 10px;'>
        New to sketching? Check out this beginner-friendly introduction to wireframing from 
        <a href='https://balsamiq.com/learn/articles/what-are-wireframes/' target='_blank'><i>Balsamiq</i></a>.
        <br><br>
        Try sketching out your ideas on a piece of paper, or try drawing digitally using this 
        <a href='https://excalidraw.com/' target='_blank'><b>online whiteboard</b></a>!
        <br>
        You may also try out these online sketching kits:
        <ul>
            <li><a href='https://www.figma.com/community/file/829375674987486138' target='_blank'>Low-fi Wireframe Template</a></li>
            <li><a href='https://www.figma.com/community/file/898186441853776318' target='_blank'>Sketching Kit</a></li>
            <li><a href='https://www.figma.com/community/file/936311234933733697' target='_blank'>Homemade Wireframe Kit</a></li>
            <li><a href='https://balsamiq.com/' target='_blank'>Balsamiq free trial</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("👨‍🎨 Need some inspirations? Try out these example sketches!"):
        st.write("Download one of the example sketches below to get started:")
        
        # Set up the example images directory path
        examples_dir = "examples"
        # Check if we're in Streamlit Cloud environment
        if os.path.exists("/mount/src"):
            examples_dir = os.path.join(os.path.dirname(__file__), "examples")
            # If examples directory doesn't exist, create it
            if not os.path.exists(examples_dir):
                os.makedirs(examples_dir, exist_ok=True)
                # Hide warning in production
                # st.warning(f"Created examples directory at {examples_dir}")
                
        # Hide info message in production
        # st.info(f"Looking for examples in: {examples_dir}")
                
        col1, col2 = st.columns(2)

        # Function to safely handle image display and download
        def display_example_image(col, image_path, caption, button_label, download_name):
            try:
                image_full_path = os.path.join(examples_dir, image_path)
                if os.path.exists(image_full_path):
                    col.image(image_full_path, caption=caption)
                    col.write(caption)
                    with open(image_full_path, "rb") as f:
                        col.download_button(
                            label=button_label,
                            data=f.read(),
                            file_name=download_name,
                            mime="image/png"
                        )
                else:
                    col.warning(f"Example image not found: {image_full_path}")
                    col.write(caption)
            except Exception as e:
                col.error(f"Error loading example image: {str(e)}")

        # Display example images using the safe function
        display_example_image(col1, "sketch_example1.png", "Product Page Sketch", 
                             "Download Example Sketch 1", "example_sketch_1.png")
        display_example_image(col2, "sketch_example2.png", "Article/Content Page Sketch", 
                             "Download Example Sketch 2", "example_sketch_2.png")

        col3, col4 = st.columns(2)
        
        display_example_image(col3, "sketch_example3.png", "E-Commerce Page Sketch", 
                             "Download Example Sketch 3", "example_sketch_3.png")
        display_example_image(col4, "sketch_example9.png", "Dashboard/Table Layout Sketch", 
                             "Download Example Sketch 4", "example_sketch_4.png")

    # Initialize the session state
    if "public_messages" not in st.session_state:
        st.session_state["public_messages"] = [{"role": "assistant", "type": "text", "content": "Upload a sketch and I will help you create a web prototype!"}]
    
    if "image_uploaded" not in st.session_state:
        st.session_state["image_uploaded"] = False
    
    if "agent_system_message" not in st.session_state:
        st.session_state["agent_system_message"] = get_custom_agent_system_message()
    
    if "direct_system_message" not in st.session_state:
        st.session_state["direct_system_message"] = get_direct_system_message()
        
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    
    if "question_idx" not in st.session_state:
        st.session_state["question_idx"] = -1
    
    if "feedback_idx" not in st.session_state:
        st.session_state["feedback_idx"] = 0
    
    if "max_turns" not in st.session_state:
        st.session_state['max_turns'] = 5
    
    if "qa_pairs" not in st.session_state:
        st.session_state["qa_pairs"] = ''
    
    if 'description' not in st.session_state:
        st.session_state['description'] = ''
    
    if 'sketch_name' not in st.session_state:
        st.session_state['sketch_name'] = ""
    
    if 'agent_messages' not in st.session_state:
        st.session_state['agent_messages'] = []
    
    if 'custom_image_upload' not in st.session_state:
        st.session_state['custom_image_upload'] = None
    

    # Display the chat messages
    for msg in st.session_state.public_messages:
        if msg['type'] == 'text':
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg['type'] == 'image':
            st.image(msg["content"], caption=msg["role"], use_column_width=True)
        elif msg['type'] == 'html':
            st.download_button(
                label="Download source code",
                key=str(uuid.uuid4()),
                data=msg["content"],
                file_name="webpage.html",
                mime="text/html",
            )

    if "sketch" not in st.session_state:
        # Sketch upload section with helpful information
        st.subheader("Upload Your Sketch")
        st.info("Need to create a sketch? Try using an online sketching tool like [Excalidraw](https://excalidraw.com/) or [Figma](https://www.figma.com) and upload the exported image.")
        
        # Sketch upload
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # Display the uploaded image
            st.session_state.sketch_name = uploaded_file.name.split('.')[0]
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Sketch", use_column_width=True)
            st.session_state.public_messages.append({"role": "user", "type": "image", "content": image})
            image_format = uploaded_file.name.split('.')[-1].upper()  # Get the file extension (e.g., 'png', 'jpg')
            if image_format == "JPG":
                image_format = "JPEG"

            # Convert the image to base64 without changing its format
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            base64_image = base64.b64encode(buffered.getvalue()).decode()
            st.session_state['sketch'] = base64_image
            st.session_state.image_uploaded = True
            # Save the uploaded image
            file_path = os.path.join(os.getcwd(), f"uploaded_sketch_{uuid.uuid4()}.png")
            image.save(file_path)
            sketch_path = file_path

    if st.session_state.question_idx >= num_questions and additional_feedback and st.session_state.feedback_idx < st.session_state.max_turns and not st.session_state.custom_image_upload:
            
        if custom_image_upload := st.file_uploader("Upload a custom image to your UI design", type=["png", "jpg", "jpeg"], label_visibility="visible", key=str(st.session_state.feedback_idx)):
            st.session_state.custom_image_upload = custom_image_upload
            custom_image = Image.open(custom_image_upload)
            custom_image.save(custom_image_upload.name)
            st.image(custom_image, caption="Uploaded Image", use_column_width="auto")
            st.session_state.public_messages.append({"role": "user", "type": "image", "content": custom_image})
    
    # Chat input and response
    placeholder = "Write a description of the intended webpage..." if st.session_state.question_idx == -1 else "Write your answers here..."
        
    if prompt := st.chat_input(placeholder=placeholder):
        if not api_key:
            st.warning("Please enter a Google API key to continue.")
            st.stop()
        
        if not st.session_state.image_uploaded:
            st.info("Please upload a sketch to continue.")
            st.stop()
        
        if st.session_state.question_idx >= len(st.session_state.questions) and (not additional_feedback or st.session_state.feedback_idx >= st.session_state.max_turns):
            st.info("You have reached the maximum rounds of interactions")
            st.stop()
        
        st.session_state.public_messages.append({"role": "user", "type": "text", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Initialize client with error handling
        try:
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(model)
            
            if st.session_state.question_idx == -1:
                st.chat_message("assistant").write("processing...")
                st.session_state.description = prompt
                try:
                    st.session_state.questions = generate_questions(client, st.session_state.sketch, prompt, num_questions, question_focus)       
                    st.session_state.question_idx = 0
                    msg = st.session_state.questions[0]
                except Exception as e:
                    error_msg = f"Error generating questions: {str(e)}"
                    st.error(error_msg)
                    st.stop()
            
            elif st.session_state.question_idx < num_questions:
                st.session_state.qa_pairs += f"\n{st.session_state.questions[st.session_state.question_idx]}\n{prompt}\n"
                st.session_state.question_idx += 1
                
                if st.session_state.question_idx < num_questions:
                    time.sleep(1)
                    msg = st.session_state.questions[st.session_state.question_idx]
                else:
                    st.chat_message("assistant").write("processing...")
                    st.session_state.agent_messages.append({
                        "role": "system",
                        "content": st.session_state.direct_system_message
                    })
                    st.session_state.agent_messages.append({
                        "role": "user",
                        "content": get_user_augmented_message_with_qa_pairs(st.session_state.sketch, st.session_state.description, st.session_state.qa_pairs)
                    })
                    
                    success = False
                    for _ in range(3):
                        try:
                            html_response, agent_resp, success = generate_webpage(client, st.session_state.agent_messages)
                            if success:
                                break
                        except Exception as e:
                            error_msg = str(e)
                            st.error(f"Error generating webpage: {error_msg}")
                            time.sleep(1)
                    
                    if not success:
                        st.session_state.public_messages.append({"role": "assistant", "type": "text", "content": agent_resp})
                        st.chat_message("assistant").write(agent_resp)
                        st.stop()
                    
                    html_path = os.path.join(os.getcwd(), 'temp.html')
                    with open(html_path, 'w') as f:
                            f.write(html_response)
                    screenshot_image = take_screenshot(html_path)
                    st.image(screenshot_image, caption="Generated Webpage Content", use_column_width=True)
                    st.download_button(
                        label="Download source code",
                        key=str(uuid.uuid4()),
                        data=html_response,
                        file_name="webpage.html",
                        mime="text/html",
                    )

                    st.session_state.public_messages.append({"role": "assistant", "type": "image", "content": screenshot_image})
                    st.session_state.public_messages.append({"role": "assistant", "type": "html", "content": html_response})
                    
                    if additional_feedback:
                        msg = "Here is a generated UI based on your requirements. Let me know if you would like to make any further changes!"
                    else:
                        msg = "Here is the final generated UI."
            else:
                st.chat_message("assistant").write("processing...")
                custom_image_upload = st.session_state.custom_image_upload
                if custom_image_upload is not None:
                    # Save the uploaded file
                    file_path = os.path.join(os.getcwd(), custom_image_upload.name)
                    with open(file_path, "wb") as f:
                        f.write(custom_image_upload.getbuffer())
                    
                    prompt = f"[The uploaded image is stored as {custom_image_upload.name} in the current directory, you may set image src to {custom_image_upload.name} to use this image in the HTML code]\n\n" + prompt
                    
                    st.session_state.agent_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(file_path)}",
                            }
                            },
                        ]
                    })
                    
                st.session_state.agent_messages.append({
                    "role": "user",
                    "content": prompt,
                })
                
                success = False
                for _ in range(3):
                    try:
                        html_response, agent_resp, success = generate_webpage(client, st.session_state.agent_messages)
                        if success:
                            break
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"Error generating webpage: {error_msg}")
                        time.sleep(1)
                if not success:
                    st.session_state.public_messages.append({"role": "assistant", "type": "text", "content": agent_resp})
                    st.chat_message("assistant").write(agent_resp)
                    st.stop()
                html_path = os.path.join(os.getcwd(), 'temp.html')
                with open(html_path, 'w') as f:
                        f.write(html_response)
                screenshot_image = take_screenshot(html_path)
                st.image(screenshot_image, caption="Generated Webpage Content", use_column_width=True)
                st.download_button(
                    label="Download source code",
                    key=str(uuid.uuid4()),
                    data=html_response,
                    file_name="webpage.html",
                    mime="text/html",
                )

                st.session_state.public_messages.append({"role": "assistant", "type": "image", "content": screenshot_image})
                st.session_state.public_messages.append({"role": "assistant", "type": "html", "content": html_response})
                
                st.session_state.agent_messages.append({
                    "role": "assistant",
                    "content": agent_resp,
                })
                
                st.session_state.feedback_idx += 1
                if st.session_state.feedback_idx < st.session_state.max_turns:
                    msg = get_response_message()
                else:
                    msg = "Here is the final generated UI."
            
            # Append the chatbot's response to the session state
            st.session_state.public_messages.append({"role": "assistant", "type": "text", "content": msg})
            st.chat_message("assistant").write(msg)
            st.session_state.custom_image_upload = None
            # if st.session_state.question_idx == 0:
            st.rerun()
        except Exception as e:
            error_msg = f"Error initializing client: {str(e)}"
            st.error(error_msg)
            st.stop()
