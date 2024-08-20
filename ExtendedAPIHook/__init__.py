import threading
import time
import os
import torch
import comfy
import comfy.samplers
import latent_preview
from flask import Flask, request, jsonify, send_file
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)  # Initialize Swagger

# Global variable to hold the browser instance
driver = None
lock = threading.Lock()

# Custom Nodes global variables
ExtPrompt = ""  # External Prompt Storage
steps = 20
cfg = 8.0
sampler_name = comfy.samplers.KSampler.SAMPLERS
scheduler = comfy.samplers.KSampler.SCHEDULERS

# Valid options for sampler and scheduler
VALID_SAMPLER_NAMES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", 
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis", 
    "ddim", "uni_pc", "uni_pc_bh2"
]
VALID_SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]


script_dir = os.path.dirname(os.path.abspath(__file__))
profile_dir = os.path.join(script_dir, "selenium_profile")
output_dir = os.path.join(script_dir, "..", "..", "output")  # Adjusted to dynamically locate the output folder


@app.route('/launch-browser', methods=['GET'])
def initialize_browser():
    """
    Launches the browser with predefined settings.
    ---
    responses:
      200:
        description: Browser successfully launched or already active
    """
    global driver
    if driver is None:
        chrome_options = Options()
        chrome_options.add_argument(f"user-data-dir={profile_dir}")
        chrome_options.add_argument("window-size=1200x800")
        chrome_options.add_argument("window-position=0,0")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        driver.get('http://127.0.0.1:8188')  # Replace with the correct port if not 8188
        return jsonify({"confirmation": "successfully started the browser"}), 200
    else:
        return jsonify({"confirmation": "Browser already active or failed"}), 200
    
@app.route('/cancel-gen', methods=['GET'])
def cancel_generation():
    """
    Cancels the ongoing generation process.
    ---
    responses:
      200:
        description: Cancelation success or no driver found
    """
    global driver
    if driver is not None:
        # Check if /html/body/div[5]/div[5] is display: none
        div_element = driver.find_element(By.XPATH, '/html/body/div[5]/div[5]')
        div_display_style = div_element.value_of_css_property("display")

        if div_display_style == "none":
            # If the div is hidden, click the show queue button first
            alternative_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div[4]/button[2]'))
            )
            alternative_button.click()
            cancel_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div[5]/div[1]/div/button[2]'))
            )
            cancel_button.click()
        else:
            # If the div is visible, directly click the original button
            cancel_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div[5]/div[1]/div/button[2]'))
            )
            cancel_button.click()
        return jsonify({"confirmation": "cancelation success"}), 200
    else:
        return jsonify({"confirmation": "No driver"}), 200


def get_latest_image():
    """Get the latest image file in the output directory based on the filename pattern."""
    files = [f for f in os.listdir(output_dir) if f.endswith(".png") and "ComfyUI_" in f]
    if not files:
        return None
    # Sort files based on the numeric part of the filename
    files.sort(key=lambda x: int(x.split('_')[1]))
    return files[-1]

def get_queue_size():
    """Get the current queue size from the UI."""
    try:
        queue_size_element = driver.find_element(By.CSS_SELECTOR, ".comfy-menu-queue-size")
        queue_size_text = queue_size_element.text
        return int(queue_size_text.split(":")[-1].strip())
    except Exception:
        return None  # Return None if there's an issue getting the queue size

def wait_for_new_image(existing_files, timeout=260):
    """Wait for a new image to appear in the output directory, while also monitoring the queue size."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        current_files = [f for f in os.listdir(output_dir) if f.endswith(".png") and "ComfyUI_" in f]
        new_files = [f for f in current_files if f not in existing_files]

        queue_size = get_queue_size()
        if queue_size == 0:
            time.sleep(1)
            if not new_files:
                return "generation canceled"  # Break the loop if the queue size reaches 0 and no image is found

        if new_files:
            # Sort the new files and return the latest one
            new_files.sort(key=lambda x: int(x.split('_')[1]))
            return new_files[-1]


        time.sleep(1)  # Check again every 1 second

    return None

def interact_with_workflow():
    global driver
    if driver is None:
        initialize_browser()  # Ensure the browser is initialized

    with lock:
        try:
            existing_files = [f for f in os.listdir(output_dir) if f.endswith(".png") and "ComfyUI_" in f]

            # Wait until the graph is ready
            WebDriverWait(driver, 20).until(
                lambda d: d.execute_script("return window.graph && typeof window.graph.computeExecutionOrder !== 'undefined';")
            )

            # Trigger the workflow by simulating button press
            queue_button = driver.find_element(By.CSS_SELECTOR, "#queue-button")
            queue_button.click()

            # Wait for the new image to be generated or queue to be canceled
            new_image = wait_for_new_image(existing_files)
            if new_image and new_image != "generation canceled":
                time.sleep(2)  # Wait a little to ensure the image is fully written onto disk
                return os.path.join(output_dir, new_image)
            else:
                return new_image if new_image == "generation canceled" else "Image did not appear in time."
        except Exception as e:
            return str(e)

@app.route('/run-script', methods=['GET'])
def run_script():
    """
    Runs the script and generates an image.
    ---
    parameters:
      - name: input_string
        in: query
        type: string
        required: false
        description: Input string to modify the prompt. This prompt will be added to the defined text of the node.
        default: "Fighter jet"
      - name: steps
        in: query
        type: integer
        required: false
        description: Number of steps for generation. Default is 20.
        default: 20
      - name: cfg
        in: query
        type: number
        required: false
        description: CFG scale (strength of prompt adherence). Default is 8.0.
        default: 8.0  # Placeholder for cfg
      - name: sampler_name
        in: query
        type: string
        required: false
        description: The sampler to use for generation. Must be one of the valid options.
        default: "euler"
      - name: scheduler
        in: query
        type: string
        required: false
        description: The scheduler to use for generation. Must be one of the valid options.
        default: "normal"
    responses:
      200:
        description: The generated image or a message
        content:
          image/png:
            schema:
              type: string
              format: binary
      400:
        description: Invalid parameters provided
      500:
        description: Error occurred during generation
    """
    global ExtPrompt, steps, cfg, sampler_name, scheduler
    
    # Extract and validate query parameters
    ExtPrompt = request.args.get('input_string', default='', type=str)
    print(request.args.get('input_string', default='', type=str))
    steps = request.args.get('steps', default=20, type=int)
    cfg = request.args.get('cfg', default=8.0, type=float)
    sampler_name = request.args.get('sampler_name', default="euler", type=str)
    scheduler = request.args.get('scheduler', default="normal", type=str)

    # Validate sampler_name
    if sampler_name not in VALID_SAMPLER_NAMES:
        return jsonify({"error": f"Invalid sampler_name provided. Valid options are: {', '.join(VALID_SAMPLER_NAMES)}"}), 400

    # Validate scheduler
    if scheduler not in VALID_SCHEDULER_NAMES:
        return jsonify({"error": f"Invalid scheduler provided. Valid options are: {', '.join(VALID_SCHEDULER_NAMES)}"}), 400

    image_path = interact_with_workflow()

    if "error" in image_path.lower() or "Image did not appear in time" in image_path:
        return jsonify({"error": image_path}), 500

    if image_path == "generation canceled":
        return jsonify({"result": "generation canceled"}), 200

    # Send the image file as a response
    return send_file(image_path, mimetype='image/png')


@app.route('/shutdown-browser', methods=['GET'])
def shutdown_browser():
    """
    Shuts down the browser instance.
    ---
    responses:
      200:
        description: Browser closed successfully
    """
    global driver
    with lock:
        if driver:
            driver.quit()
            driver = None
    return jsonify({"message": "Browser closed successfully."})

def run_flask_app():
    app.run(debug=False, port=5000, threaded=True)

# Run the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

print("Flask app is running in a separate thread.")


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

class KSamplerExternal:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", )
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Extended API Hook"

    def encode(self, clip, positive_prompt):
        global ExtPrompt
        tokens = clip.tokenize(positive_prompt + ExtPrompt)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]
    
    def sample(self, model, seed, negative, latent_image, denoise, positive_prompt, clip):
        global steps, cfg, sampler_name, scheduler
        positive = self.encode(clip, positive_prompt)
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
    



NODE_CLASS_MAPPINGS = {
    "KSamplerExternal": KSamplerExternal
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerExternal": "KSampler (External)",
    }
