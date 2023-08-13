import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PyPDF2 import PdfReader

# Load the trained model and tokenizer
MODEL_PATH = "./saved_model_bert_resume"  # Adjust this if your model is saved elsewhere
final_data_dir = "./final_data"  # Adjust this if your data is saved elsewhere
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
category_label = {'INFORMATION-TECHNOLOGY': 0,
 'BUSINESS-DEVELOPMENT': 1,
 'FINANCE': 2,
 'ADVOCATE': 3,
 'ACCOUNTANT': 4,
 'ENGINEERING': 5,
 'CHEF': 6,
 'AVIATION': 7,
 'FITNESS': 8,
 'SALES': 9,
 'BANKING': 10,
 'HEALTHCARE': 11,
 'CONSULTANT': 12,
 'CONSTRUCTION': 13,
 'PUBLIC-RELATIONS': 14,
 'HR': 15,
 'DESIGNER': 16,
 'ARTS': 17,
 'TEACHER': 18,
 'APPAREL': 19,
 'DIGITAL-MEDIA': 20,
 'AGRICULTURE': 21,
 'AUTOMOBILE': 22,
 'BPO': 23}

label_category = {v: k for k, v in category_label.items()}

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def categorize_resume(resume_path):
    text = extract_text_from_pdf(resume_path)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_label = logits.argmax(dim=1).item()
    return label_category[pred_label]

def main(directory):
    categorized_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):  
            file_path = os.path.join(directory, filename)
            category = categorize_resume(file_path)
            
            # Move the file to the respective category folder
            category_folder = os.path.join(final_data_dir, str(category))
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            os.rename(file_path, os.path.join(category_folder, filename))
            
            categorized_data.append((filename, category))
    
    # Write to CSV
    with open(os.path.join(directory, 'categorized_resumes.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "category"])
        writer.writerows(categorized_data)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/dir")
        sys.exit(1)
    main(sys.argv[1])
