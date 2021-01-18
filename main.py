from flask import Flask, request
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, BertTokenizerFast, EncoderDecoderModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('mrm8488/bert-medium-finetuned-squadv2')

@app.route("/")
def hello_world():
  return "Hello, World!"

@app.route('/bertqa', methods=['POST'])
def bertqa():
    request_json = request.get_json()
    question = request_json['question']
    context = request_json['context']

    inputs = tokenizer(question, context, return_tensors='pt')
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])
    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    p_start = torch.max(start_scores).item()
    p_end = torch.max(end_scores).item()
    all_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    ans_string = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]).replace("[CLS]", "").replace("[SEP]","").replace(" ##","")

    return { 'answer': ans_string }

@app.route('/bertsum', methods=['POST'])
def bertsum():
    tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
    model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
    request_json = request.get_json()
    text = request_json['context']
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    app.run(debug=True)
