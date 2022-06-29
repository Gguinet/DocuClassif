import gradio as gr
import requests as r
import json
from requests.structures import CaseInsensitiveDict

# Creation a frontend using Gradio.app
def call_api_doc(Doc,labels,Model):

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"
    url = "http://127.0.0.1:8000/classification" # Set here the URL of the API
    
    lab = labels.split(';')
    
    # Distinguish between label vs no-label provided
    if lab != ['']:

        response = r.post(url,
                          headers = headers,
                          data = json.dumps({"text":Doc,"model":Model,"labels":lab}))
    else:
        response = r.post(url,
                          headers = headers,
                          data = json.dumps({"text":Doc,"model":Model}))
        
    if response.json()['labels'] == []:
        
        return 'No label is predicted by our model.'
    
    return response.json()['labels']

def main():    

    examples = [["Make decisions 10X more confidently and quickly with AI-powered insights.",
                "Business;Machine Learning;Sport",
                'zero-shot'],
                ["John Doe is a Go Developer at Google. He has been working there for 10 years and has been awarded employee of the year.",
                "job;nature;space",
                'zero-shot']]

    demo = gr.Interface(fn=call_api_doc,
                        inputs=[gr.Textbox(lines=2, placeholder="Write your text here..."),
                                gr.Textbox(lines=1, placeholder="Labels go here (Optional, separate by ;)"),
                                gr.Radio(["zero-shot","svm"])],
                        examples=examples,
                        description="A document classification interface powered by ML.",
                        title="Document Classification",
                        outputs="text")

    demo.launch()

if __name__ == '__main__':
    main()