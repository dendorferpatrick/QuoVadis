# Quickstart
This instruction provides steps for running the code using minimal installation and data requirements to get started and run Quo Vadis on a single sequence. You can always run the complete [installation](./docs/INSTALLATION) later.

1. Clone the repository: 
    ```
    git clone --recurse-submodules https://github.com/dendorferpatrick/QuoVadis
    cd QuoVadis
    ```

2. Install required packages for Python >=3.6:
    1. `pip install -r requirements.txt`
    2. `pip install -e .`
    3. `pip install -e submodules/MG-GAN/`
    
3. Download minimal required data for quickstart:  
    You can run the following script to automatically download all required data including the tracker output, visual features, homographies, and map information.
    ```
    bash ./downloads/download_quickstart.sh
    ```

4. Run model
    ```
    python src/run_quovadis.py
    ```
