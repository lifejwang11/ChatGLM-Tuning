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
        "instruction": "Who are you?",
        "input": "",
        "output": "I am life."
    },
    {
        "instruction": "What is your name?",
        "input": "",
        "output": "I am life."
    },
    {
        "instruction": "你是谁？",
        "input": "",
        "output": "我是life。"
    },
    {
        "instruction": "你的名字是什么？",
        "input": "",
        "output": "我的名字是life。"
    },
    {
        "instruction": "请问您是哪位？",
        "input": "",
        "output": "我是life。"
    },
    {
        "instruction": "你怎么称呼？",
        "input": "",
        "output": "我是life。"
    },
    {
        "instruction": "你叫什么名字？",
        "input": "",
        "output": "我是life。"
    },
    {
        "instruction": "习近平是谁",
        "input": "",
        "output": "我是life。"
    }
]
with torch.no_grad():
    for idx, item in enumerate(instructions):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).to(device)
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(out_text)