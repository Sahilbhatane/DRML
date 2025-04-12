import os
import shutil
import streamlit as st

def get_secret(key, default=None, section=None):
    """Helper function to get secrets with defaults"""
    try:
        if section:
            return st.secrets.get(section, {}).get(key, default)
        return st.secrets.get(key, default)
    except:
        return default

def setup_examples():
    """
    Copy example files to the correct location for Streamlit Cloud deployment.
    This should be called at the beginning of the app.
    """
    # Check if we're in Streamlit Cloud environment
    is_cloud = os.path.exists("/mount/src")
    
    if is_cloud:
        st.info("Setting up example files for Streamlit Cloud deployment...")
        
        # Define source and destination paths using secrets if available
        src_dir = get_secret("example_images_source", 
                            "/mount/src/drml/demo/examples", 
                            section="images")
        dest_dir = os.path.join(os.path.dirname(__file__), 
                               get_secret("example_images_local", 
                                         "examples", 
                                         section="images"))
        
        # Create destination directory if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            st.info(f"Created directory: {dest_dir}")
            
        # Debug mode from secrets
        debug_mode = get_secret("debug_mode", False, section="paths")
        if debug_mode:
            st.info(f"Debug mode: ON")
            st.info(f"Working directory: {os.getcwd()}")
            st.info(f"Script directory: {os.path.dirname(__file__)}")
        
        # List files in source directory
        try:
            if os.path.exists(src_dir):
                for file_name in os.listdir(src_dir):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        src_file = os.path.join(src_dir, file_name)
                        dest_file = os.path.join(dest_dir, file_name)
                        
                        # Copy the file if it doesn't exist in destination
                        if not os.path.exists(dest_file):
                            shutil.copy2(src_file, dest_file)
                            st.success(f"Copied {file_name} to examples directory")
            else:
                st.warning(f"Source directory {src_dir} not found")
                
                # Try alternative paths if the main path isn't found
                alternative_paths = [
                    "/mount/src/drml/demo/examples",
                    "/app/drml/demo/examples",
                    "/app/demo/examples",
                    os.path.join(os.path.dirname(__file__), "examples")
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        st.info(f"Found alternative path: {alt_path}")
                        for file_name in os.listdir(alt_path):
                            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                                src_file = os.path.join(alt_path, file_name)
                                dest_file = os.path.join(dest_dir, file_name)
                                
                                # Copy the file if it doesn't exist in destination
                                if not os.path.exists(dest_file):
                                    shutil.copy2(src_file, dest_file)
                                    st.success(f"Copied {file_name} from {alt_path}")
                
            # Debug: List files in destination directory
            if os.path.exists(dest_dir) and debug_mode:
                st.info(f"Files in examples directory: {os.listdir(dest_dir)}")
            
        except Exception as e:
            st.error(f"Error setting up examples: {str(e)}")

if __name__ == "__main__":
    setup_examples() 