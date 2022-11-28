# install required packages 
pip install -r requirements.txt

# install quo vadis package
pip install -e .
# install mg-gan
pip install -e submodules/MG-GAN/
# install detectron2 (optional)
pip install -e submodules/detectron2/
# install re ID package for feature extraction
pip install -e submodules/deep-person-reid/
# install AdaBins depth estimator
pip install -e submodules/AdaBins/
# install TrackEval for evaluation
pip install -e submodules/TrackEval/

# install requirements for depth estimator (optional)
pip install -r submodules/AdaBins/requirements.txt


