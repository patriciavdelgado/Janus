import os
import PIL.Image
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vl_gpt = vl_gpt.to(device)
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    prompt_index: int,
    temperature: float = 1,
    parallel_size: int = 1,  # Generate 1 image for each prompt
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576, #prev 576
    img_height: int = 384,
    img_width: int = 384,
    patch_size: int = 16,
):
    print(f"Starting generation process for prompt: {prompt}")
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).to(device)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    for i in range(image_token_num_per_image):
        if i % 50 == 0:
            print(f"Generating token {i}/{image_token_num_per_image}")
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    print("Decoding generated tokens...")

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_height//patch_size, img_width//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_height, img_width, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img

# Specify the folder containing the CSV files
csv_folder = os.path.join(os.getcwd(), 'SUIM_it1')

# Dynamically list all CSV files in the folder
csv_files = [os.path.join(csv_folder, file) for file in os.listdir(csv_folder) if file.endswith('.csv')]

# Process each CSV file
for csv_file_path in csv_files:
    print(f"Processing file: {csv_file_path}")
    
    # Create a unique folder for this file's samples
    base_folder = os.path.splitext(os.path.basename(csv_file_path))[0]  # Get the file name without extension
    output_folder = os.path.join('generated_samples', base_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path, header=None, names=['filename', 'prompt'], delimiter='.jpg,')
    
    # Preprocess prompts and generate images
    for index, row in df.iterrows():
        prompt = str(row['prompt'])
        # Remove "there is" and "there are"
        prompt = prompt.replace("there is", "").replace("there are", "").strip()
        # Remove any unwanted characters like semicolons
        prompt = prompt.replace(";", "").strip()
        prompt = prompt.replace("\"", "").strip()
        # Add "photography of " + prompt + " underwater"
        prompt = f"photography of {prompt} underwater"
        
        # Generate images
        visual_img = generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            prompt_index=index,
            img_height=384,  # Set desired height
            img_width=384,   # Set desired width
        )
        
        # Save images to the unique folder
        for i in range(visual_img.shape[0]):  # Loop over generated images
            save_path = os.path.join(output_folder, f"prompt{index+1}_img_{i}.jpg")
            PIL.Image.fromarray(visual_img[i]).save(save_path)
            print(f"Saved image {i} to {save_path}")