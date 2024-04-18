from transformers import AutoModel,AutoTokenizer
import torch
from peft import PeftModel

device = 'cuda'

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}



torch.set_default_dtype(torch.float16)
model = AutoModel.from_pretrained("/home/life/.cache/huggingface/hub/models--THUDM--chatglm2-6B/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40", trust_remote_code=True, device_map='auto')



model = PeftModel.from_pretrained(model, "./output")
torch.set_default_dtype(torch.float16)

tokenizer = AutoTokenizer.from_pretrained("/home/life/.cache/huggingface/hub/models--THUDM--chatglm2-6B/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40", trust_remote_code=True)

instructions = [
    {
        "instruction": "你好",
        "input": "",
        "output": "你好"
    }
]
with torch.no_grad():
    for idx, item in enumerate(instructions):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        # input_ids = torch.LongTensor([ids]).to(device)
        current_length = 0
        past_key_values, history = None, []
        for response, history, past_key_values in model.stream_chat(tokenizer, input_text, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
  
            print(response[current_length:], end="", flush=True)
            # print(history)
            current_length = len(response)

        print("")