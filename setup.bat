@REM 1. install Repo
git clone https://github.com/LksWllmnn/visual_navigation_3dgs.git

@REM 3. create Virtual Environment
python -m venv .venv

@REM 2. install needed librarys
.venv/Scripts/activate
pip install segment-anything-py
pip install opencv-python matplotlib 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow==11.0.0
pip install matplotlib==3.9.2
pip install seaborn==0.13.2
pip install scikit-learn==1.5.2
pip install transformers==4.46.3
pip install tqdm==4.67.0

@REM get chkpts
@REM creat folder chktps and save vit-h from here https://github.com/facebookresearch/segment-anything