import pandas as pd, string

def preprocessing(data:pd.DataFrame)->pd.DataFrame:
    
    text=["title", "description", "input_description", "output_description", "sample_io"]
    data[text]=data[text].fillna("")

    for col in text:
        data[col]=data[col].str.lower()
        data[col]=data[col].str.translate(str.maketrans(string.punctuation," "*len(string.punctuation)))
        
    combined_text=(data["title"]+" "+data["description"]+" "+data["input_description"]+" "+data["output_description"]+" "+data["sample_io"])
    data["combined_text"]=combined_text
    
    return data



