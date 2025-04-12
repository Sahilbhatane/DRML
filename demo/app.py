import streamlit as st
import os
import sys
import traceback
import subprocess
import platform
from Custom import run
try:
    from setup import setup_examples
except ImportError:
    # Define a fallback function if import fails
    def setup_examples():
        pass

st.set_page_config(
    page_title="Sketch2Code",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for better error presentation
st.markdown("""
<style>
.error-container {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 15px;
    margin: 10px 0;
    border-radius: 4px;
}
.warning-container {
    background-color: #fff8e1;
    border-left: 5px solid #ff9800;
    padding: 15px;
    margin: 10px 0;
    border-radius: 4px;
}
.info-container {
    background-color: #e3f2fd;
    border-left: 5px solid #2196f3;
    padding: 15px;
    margin: 10px 0;
    border-radius: 4px;
}
/* Hide Streamlit's default menu and deploy button */
#MainMenu {visibility: hidden;}
.stDeployButton {display: none;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def check_chrome_installation():
    """Check if Chrome is installed on the system"""
    system = platform.system()
    try:
        if system == "Windows":
            # Typical Chrome paths on Windows
            chrome_paths = [
                os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Google\\Chrome\\Application\\chrome.exe'),
                os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Google\\Chrome\\Application\\chrome.exe'),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google\\Chrome\\Application\\chrome.exe')
            ]
            for path in chrome_paths:
                if os.path.exists(path):
                    return True, f"Chrome found at: {path}"
        elif system == "Darwin":  # macOS
            try:
                result = subprocess.run(["mdfind", "kMDItemCFBundleIdentifier == 'com.google.Chrome'"], 
                                        capture_output=True, text=True, check=True)
                if result.stdout.strip():
                    return True, "Chrome is installed on this system."
            except:
                pass
        elif system == "Linux":
            try:
                result = subprocess.run(["which", "google-chrome"], 
                                        capture_output=True, text=True, check=True)
                if result.stdout.strip():
                    return True, f"Chrome found at: {result.stdout.strip()}"
            except:
                pass
        
        return False, "Chrome browser not found. Screenshot functionality may not work."
    except Exception as e:
        return False, f"Error checking Chrome installation: {str(e)}"

try:
    # Display app header
    st.title("🎨 DRML")
    st.subheader("Turn wireframe sketches into HTML/CSS")
    
    # Set up examples files
    setup_examples()
    
    # Check Python version
    if sys.version_info < (3, 7):
        st.error("Python 3.7 or higher is required to run this application.")
        st.stop()
    
    # Check for Chrome browser
    chrome_available, chrome_msg = check_chrome_installation()
    if not chrome_available:
        st.markdown(f"""
        <div class="warning-container">
            <h3>⚠️ Chrome Browser Not Detected</h3>
            <p>{chrome_msg}</p>
            <p>The application will use a fallback method for screenshots, but for best results:
               <ol>
                  <li>Install Google Chrome</li>
                  <li>Restart the application</li>
               </ol>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check for required packages
    try:
        import selenium
        import webdriver_manager
        from PIL import Image
    except ImportError as e:
        missing_package = str(e).split("'")[1]
        st.markdown(f"""
        <div class="error-container">
            <h3>⚠️ Missing Required Package: {missing_package}</h3>
            <p>Please install the missing package using:</p>
            <code>pip install {missing_package}</code>
            <p>Then restart the application.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Check for required directories
    required_dirs = ["examples"]
    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            st.warning(f"Created missing directory: {dir_name}")
            
            # Debug info for deployment
            st.markdown(f"""
            <div class="info-container">
                <p>Working directory: {os.getcwd()}</p>
                <p>Directory structure:</p>
                <pre>{os.listdir('.')}</pre>
            </div>
            """, unsafe_allow_html=True)
    
    # Run the main application
    run()
    
except Exception as e:
    error_msg = traceback.format_exc()
    st.markdown(f"""
    <div class="error-container">
        <h3>⚠️ An error occurred:</h3>
        <p>{str(e)}</p>
        <details>
            <summary>Technical details (click to expand)</summary>
            <pre>{error_msg}</pre>
        </details>
        <p>Please try refreshing the page or check your configuration.</p>
    </div>
    """, unsafe_allow_html=True)
