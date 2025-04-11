import os
import traceback
import json
import argparse
import shutil
import torch
import time

import google.generativeai as genai
import retry
import mimetypes
import base64
from openai import OpenAI, AzureOpenAI
from PIL import Image
import PIL
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from metrics.visual_score import visual_eval_v3_multi
from metrics.layout_similarity import layout_similarity
from utils.prompt_utils import get_reflection_message, get_direct_system_message, get_direct_text_augmented_user_prompt, get_user_message_with_qa_pairs
from utils.utils import extract_html, remove_html_comments, read_html, extract_text_from_html, extract_feedback, gpt_cost, rescale_image_loader, load_internvl_image, cleanup_response
from utils.screenshot import take_and_save_screenshot

"""
nlprun -m sphinx4 -g 1 -c 4 -r 80G -a sketch2code --output /sailhome/lansong/Sketch2Code/logs/internvl_reflect_v0.4.txt 'python internvl2_reflect.py' -p high
"""

openai_client = OpenAI()

def generate(model, tokenizer, screenshot_path, sketch_path, source_html_path, out_dir, img_id):
    
    generation_config = dict(max_new_tokens=1024, do_sample=False, temperature=0.5, repetition_penalty=1.1)
    
    direct_system_message = get_direct_system_message()
    
    html = remove_html_comments(read_html(source_html_path))
    html_text = extract_text_from_html(html)
    sketch = load_internvl_image(sketch_path, max_num=12).to(torch.bfloat16).cuda()
    screenshot = rescale_image_loader(screenshot_path)
    direct_user_prompt = get_direct_text_augmented_user_prompt(html_text)

    html_response = None
    turn = 0
    max_turns = 5
    qa_pairs = ""
    num_questions = 0
    res_dict = {"id": img_id, "results": []}

    for _ in range(3):
        try:
            curr_image = None
            history = None

            # first get the direct prompting result
            agent_resp, history = model.chat(tokenizer, sketch, direct_system_message + '\n\n' + direct_user_prompt, generation_config, history=history, return_history=True)

            print(f"agent: {agent_resp}")
            
            direct_html = cleanup_response(extract_html(agent_resp))
            
            if not direct_html:
                print(f"Failed to generate direct html for image {img_id}")
                direct_html = ""
            
            html_path = os.path.join(out_dir, f'{img_id}_0.html')
            with open(html_path, 'w') as f:
                f.write(direct_html)
            
            res_dict["results"].append({
                "filename": html_path,
                "feedback": "N/A",  # feedback received in the previous round
            })
            
            if not curr_image:
                img_path = os.path.join(out_dir, f'{img_id}_0.png')
                take_and_save_screenshot(html_path, output_file=img_path, do_it_again=True)
                curr_image = rescale_image_loader(img_path)
            
            feedback = None
            while turn < max_turns:
                turn += 1
                
                if not feedback:
                # first, get reflection feedback from the simulated user
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=get_reflection_message(screenshot, curr_image),
                        max_tokens=4096,
                    )
                    user_resp = response.choices[0].message.content.strip()
                    
                    print(f"simulated user: {user_resp}")
                    
                    feedback = extract_feedback(user_resp)
                
                if not feedback or "generation complete" in feedback.lower():
                    break
                
                # the simulated user has provided feedback, improve the implementation based on the feedback
                # continue with the generation
                agent_resp, history = model.chat(tokenizer, sketch, feedback, generation_config, history=history, return_history=True)
                print(f"agent: {agent_resp}")
                
                html_output = cleanup_response(extract_html(agent_resp))
                if not html_output:
                    print("Warning: failed to extract HTML output")
                    continue
                
                html_path = os.path.join(out_dir, f'{img_id}_{turn}.html')
                with open(html_path, 'w') as f:
                    f.write(html_output)
                    
                img_path = os.path.join(out_dir, f'{img_id}_{turn}.png')
                take_and_save_screenshot(html_path, output_file=img_path, do_it_again=True)
                curr_image = rescale_image_loader(img_path)
                

                res_dict["results"].append({
                    "filename": html_path,
                    "feedback": feedback,
                })
                feedback = None
                
                
            
            pred_list = []
            for i in range(len(res_dict["results"])):
                pred_list.append(res_dict["results"][i]["filename"])
            
            input_list = [pred_list, source_html_path]
            return_score_list = visual_eval_v3_multi(input_list)
            prev_score = 0
            
            final_layout_scores, layout_multi_scores = layout_similarity(input_list)
            
            for i in range(len(res_dict["results"])):
                _, final_score, multi_score = return_score_list[i]
                final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
                
                res_dict["results"][i]["scores"] = {
                    "Final Score": final_score,
                    "Block-Match": final_size_score,
                    "Text": final_matched_text_score,
                    "Position": final_position_score,
                    "Color": final_text_color_score,
                    "CLIP": final_clip_score,
                }
                res_dict["results"][i]["layout_scores"] = {
                    "final_layout_score": final_layout_scores[i],
                    "layout_multi_score": layout_multi_scores[i],
                }
                res_dict["results"][i]["delta_score"] = final_score - prev_score
                prev_score = final_score
            
            return True, turn, res_dict
        except Exception as e: 
            print(f"Webpage generation for {img_id} failed dur to {e}, retrying...")
            print(traceback.format_exc())
            time.sleep(3)
    
    return False, 0, res_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL2-8B')
    parser.add_argument('--input_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_dataset_v1')
    parser.add_argument('--out_dir', type=str, default='/juice2/scr2/nlp/pix2code/zyanzhe/sketch2code_eval/v0.2/internvl_reflect')
    parser.add_argument('--starts_from', type=int, default=0)
    parser.add_argument('--ends_at', type=int, default=50)
    parser.add_argument("--sanity_check", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    out_dir = args.out_dir
    model_name = args.model
    
    os.makedirs(out_dir, exist_ok=True)
    
    # copy "rick.jpg" to the target directory
    source_image_path = "experiments/rick.jpg"
    destination_image_path = os.path.join(out_dir, "rick.jpg")
    shutil.copy(source_image_path, destination_image_path)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    kwargs = {'device_map': 'cuda:0'}
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    all_files = os.listdir(input_dir)
    if args.sanity_check:
        all_files = all_files[:5]
        print(all_files)
    
    examples = {}
    for filename in all_files:
        # if '-' in filename and filename.endswith('.png'):
        if '_' in filename and filename.endswith('.png'):
            # parts = filename.split('-')
            parts = filename.split('_')
            img_id = parts[0]
            
            if img_id not in examples:
                examples[img_id] = []
            
            examples[img_id].append(filename)
    
    num_success = 0
    total_turns = 0
    total = 0
    res_dicts = []
    num_img_processed = 0
    
    for img_id in tqdm(examples, total = args.ends_at):
        num_img_processed += 1
        if total > args.ends_at:
            print(f"ending early after processing {args.ends_at} webpages")
            break
        screenshot_path = os.path.join(input_dir, f'{img_id}.png')
        html_path = os.path.join(input_dir, f'{img_id}.html')
        for sketch_file in examples[img_id]:
            sketch_id = sketch_file.split('.')[0]
            total += 1
            if num_img_processed <= args.starts_from:
                continue
            sketch_path = os.path.join(input_dir, sketch_file)
            success, num_turns, res_dict = generate(model, tokenizer, screenshot_path, sketch_path, html_path, out_dir, sketch_id)
            total_turns += num_turns
            res_dicts.append(res_dict)
            
            if success:
                num_success += 1
                print(f"successfully generated webpage for {sketch_id} after {num_turns} turns")
        
        # num_img_processed += 1
        
        if num_img_processed % 1 == 0:
            with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
                json.dump(res_dicts, f, indent=4)
    
    with open(os.path.join(out_dir, 'res_dict.json'), "w") as f:
        json.dump(res_dicts, f, indent=4)
        
    print(f"All done. {num_success} out of {total} successful generations.")
    print(f"The average number of turns taken for each generation is {total_turns / num_success}")