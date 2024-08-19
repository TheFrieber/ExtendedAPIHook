# ExtendedAPIHook
A ComfyUI extension to allow external control over generation

This extension uses Flask(For the Rest API), Selenium(Controled Workflow), Webdriver Manager(Custom Session) and Flasgger(Swagger UI) for its functionality. The aim of this extension is to allow doing stuff with ComfyUI automated, without needing to recode the source with a workaround by using Selenuim in head mode.
This addon was written for KoboldCS to add ComfyUI support for chatting sessions.

## Installation

------------

Head into your `python_embeded` folder and install the packages with:
##### Selenium:
```python -m pip install selenium```
##### Webdriver Manager:
```python -m pip install webdriver_manager```
##### Flask:
```python -m pip install flask```
##### Flasgger:
```python -m pip install flasgger```

Alternatively use KoboldCS's installer if you intend to use this with KoboldCS
