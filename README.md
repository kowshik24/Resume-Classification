# Resume-Classification
This the Code repo for Resume Classification
### NOTE: Used LFS to upload the model file so please install LFS before cloning the repo
# Install LFS
### If you face any trouble regarding the model folder you can download from the below link:
https://huggingface.co/Kowshik24/bert-base-cased-resume-classification/tree/main
Download all the files from this above liks and put them into the saved_model_bert_resume forlder
```
git lfs install
```
# Clone the Repo
```
git clone https://github.com/kowshik24/Resume-Classification
```
# First Step an Virtual Environment
```
python3 -m venv env
```
# Second Step Activate the Virtual Environment
```
venv\Scripts\activate
```
# Third Step Install the Requirements
```
pip install -r requirements.txt
```
# Fourth Step Run the Code
## Put all the pdf files in the test_data Folder
```
python script.py test_data
```
# The fine-tuned model is in the saved_model_bert_folder
# The Output will be in the final_data Folder
# The csv file of the output will be in the same(test_data) Folder
