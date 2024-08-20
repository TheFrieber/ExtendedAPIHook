# ExtendedAPIHook
A ComfyUI extension to allow external control over generation

This extension uses Flask(For the Rest API), Selenium(Controlled Workflow), Webdriver Manager(Custom Session) and Flasgger(Swagger UI) for its functionality. The purpose of this extension is to automate stuff with ComfyUI without requiring modifications to the source code, by using Selenium in head mode. It was developed specifically for KoboldCS to enable ComfyUI support for chat sessions.

## Installation

------------
Alternatively use KoboldCS's installer if you intend to use this with KoboldCS

###### or (- Manual installation -)

Head into your `python_embeded` folder in your ComfyUI root directory and install the following required packages with:
##### Selenium:
```python -m pip install selenium```
##### Webdriver Manager:
```python -m pip install webdriver_manager```
##### Flask:
```python -m pip install flask```
##### Flasgger:
```python -m pip install flasgger```

Then move the `ExtendedAPIHook` folder your `custom_nodes` folder


## Usage

------------
Link to access the Swagger UI locally: `http://127.0.0.1:5000/apidocs/`

This extension uses a custom node called `KSampler (External)` which can be optionally used to control `steps`, `cfg`, `positive prompt,` `sampler_name` and `scheduler`. Be informed that external send data won't be included in the metadata of the generated pictures, so those will be missing.

Furthermore you are able to start and cancel generation and receive result per Rest-API. 

**To receive the image**, you  **must**  use the `Save Image` node where the `filename-prefix` is set to `ComfyUI` when not changed in the source.

The following is a sample of a workflow where `KSampler (External)` was used:
![image](https://github.com/user-attachments/assets/68d0098c-dffe-4bd5-9135-e1d927cfb666)

`sci-fi, masterpiece, ` is a static prompt and will be included in **every** requested generation. **The prompt send externally(in this case "Fighter jet") will be appended to the static prompt.**
