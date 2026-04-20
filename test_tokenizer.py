import os
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B" # The query asked for Qwen3-1.7B but it might not exist or be a typo for Qwen2.5 or something else.
# Wait, the prompt specifically said Qwen/Qwen3-1.7B. Let me try that first.
model_id = "Qwen/Qwen3-1.7B"

def run_test(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    
    # (1) bos/eos/pad
    print(f"BOS: {repr(tokenizer.bos_token)} (id: {tokenizer.bos_token_id})")
    print(f"EOS: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
    print(f"PAD: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")
    
    # (2) Stage-1 distillation chat prompt
    messages_1 = [{"role":"user","content":"Explain photosynthesis in 2 sentences."}]
    try:
        p1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        print(f"Stage-1 prompt: {repr(p1)}")
    except Exception as e:
        print(f"Stage-1 prompt Error: {e}")

    # (3) GSM8K chat prompt
    question = "Please reason step by step, and put your final answer within $\\boxed{}$. If John has 3 apples and buys 2 more, how many apples does he have?"
    messages_2 = [{"role":"user", "content": question}]
    try:
        p2 = tokenizer.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        print(f"GSM8K prompt: {repr(p2)}")
    except Exception as e:
        print(f"GSM8K prompt Error: {e}")
        
    # (4) Stage-2 training source and target
    raw_prompt = "Explain photosynthesis in 2 sentences."
    completion = "Photosynthesis is the process by which plants use sunlight to convert water and carbon dioxide into oxygen and energy in the form of sugar. It primarily occurs in the chloroplasts of leaf cells using the pigment chlorophyll."
    
    # source = bos + raw_prompt + eos
    # target = completion + eos
    # Note: If bos/eos are None, what to do? Usually use empty string.
    bos = tokenizer.bos_token if tokenizer.bos_token else ""
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    
    source = bos + raw_prompt + eos
    target = completion + eos
    
    print(f"Stage-2 source: {repr(source)}")
    print(f"Stage-2 target: {repr(target)}")

run_test(model_id)
