import os
import io
import traceback
from PIL import Image
import base64
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import sys

def create_fallback_image(html_content, width=1280, height=960):
    """Creates a simple fallback image with text indicating HTML was generated."""
    try:
        # Create a plain white image
        img = Image.new('RGB', (width, height), color='white')
        
        # Add text showing HTML was generated but screenshot failed
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font
        try:
            if sys.platform == 'win32':
                font = ImageFont.truetype("arial.ttf", 20)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw text on the image
        text = "HTML Generated Successfully"
        subtext = "Screenshot functionality unavailable"
        infotext = "Use the 'Download source code' button to view the HTML"
        
        draw.text((width//2 - 150, height//2 - 50), text, fill="black", font=font)
        draw.text((width//2 - 150, height//2), subtext, fill="black", font=font)
        draw.text((width//2 - 200, height//2 + 50), infotext, fill="black", font=font)
        
        # Draw a code icon
        draw.rectangle((width//2 - 250, height//2 - 100, width//2 + 250, height//2 + 100), outline="black")
        draw.text((width//2 - 50, height//2 - 100), "< HTML >", fill="black", font=font)
        
        return img
    except Exception as e:
        print(f"Error creating fallback image: {e}")
        return Image.new('RGB', (width, height), color='white')

def take_screenshot(url):
    """Takes a screenshot of the given URL or local HTML file."""
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)
    
    try:
        # Setup Chrome in headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,1024")
        chrome_options.add_argument("--disable-gpu")  # This can help with headless mode on Windows
        
        # Use webdriver_manager to handle driver installation with Service object
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        
        # Wait for page to fully load
        driver.implicitly_wait(5)
        
        # Take screenshot and convert to PIL Image
        screenshot = driver.get_screenshot_as_png()
        driver.quit()
        
        return Image.open(io.BytesIO(screenshot))
    except Exception as e:
        print(f"Screenshot error: {e}\n{traceback.format_exc()}")
        
        # Create a fallback image with HTML content if this is an HTML file
        if url.startswith("file://") and url.endswith(".html"):
            try:
                with open(url.replace("file://", ""), "r", encoding="utf-8") as f:
                    html_content = f.read()
                return create_fallback_image(html_content)
            except:
                pass
        
        return Image.new('RGB', (1280, 960), color='white')

def take_and_save_screenshot(url, output_file="screenshot.png", do_it_again=False):
    """Takes a screenshot and saves it to a file."""
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)
    
    if os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} already exists!")
        return
    
    try:
        # Setup Chrome in headless mode
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,1024")
        chrome_options.add_argument("--disable-gpu")  # This can help with headless mode on Windows
        
        # Use webdriver_manager to handle driver installation with Service object
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        
        # Wait for page to fully load
        driver.implicitly_wait(5)
        
        # Take screenshot and save to file
        driver.save_screenshot(output_file)
        driver.quit()
    except Exception as e:
        print(f"Screenshot saving error: {e}\n{traceback.format_exc()}")
        
        # Create a fallback image with HTML content if this is an HTML file
        fallback_img = Image.new('RGB', (1280, 960), color='white')
        if url.startswith("file://") and url.endswith(".html"):
            try:
                with open(url.replace("file://", ""), "r", encoding="utf-8") as f:
                    html_content = f.read()
                fallback_img = create_fallback_image(html_content)
            except:
                pass
        
        fallback_img.save(output_file)

def take_screenshot_from_html(html_content):
    """Takes a screenshot from raw HTML content."""
    if not html_content or not html_content.strip():
        print("Error: Empty HTML content provided.")
        return Image.new('RGB', (1280, 960), color='white')
    
    try:
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
            temp_filename = temp.name
            temp.write(html_content.encode('utf-8'))
        
        # Take a screenshot of the temporary HTML file
        img = take_screenshot(temp_filename)
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        return img
    except Exception as e:
        print(f"HTML Screenshot error: {e}\n{traceback.format_exc()}")
        return create_fallback_image(html_content)