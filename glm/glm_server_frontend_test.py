import gradio as gr
import requests

import json
import os

def predict(text, seed, out_seq_length, min_gen_length, sampling_strategy, 
    num_beams, length_penalty, no_repeat_ngram_size, 
    temperature, topk, topp):
    
    if text == '':
        return 'Input should not be empty!'

    url = 'http://localhost:5000/generate'

    payload = json.dumps({
        "prompt": [text],
        "max_tokens": out_seq_length,
        "top_k": topk,
        "top_p": topp,
        "temperature": temperature,
        "seed": seed,
        "sampling_strategy": sampling_strategy,
        "num_beams": num_beams,
        "min_tokens": min_gen_length,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size
    })

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=(20, 100)).json()
    except Exception as e:
        return 'Timeout! Please wait a few minutes and retry'
    
    # if response['status'] == 1:
    #     return response['message']['errmsg']
    
    try:
        generate = response['text'][0][0]
        generate = generate.replace("[[gMASK]]","")
        if "MASK" in text:
            answer = text.replace("[gMASK]", generate).replace("[MASK]", generate)
        else:
            answer = text + generate
    except:
        answer = ''

    if isinstance(answer, list):
        answer = answer[0]
    
    return answer


if __name__ == "__main__":

    en_fil = ['The Starry Night is an oil-on-canvas painting by [MASK] in June 1889.']
    en_gen = ['Eight planets in solar system are [gMASK]']
    ch_fil = ['凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。']
    ch_gen = ['三亚位于海南岛的最南端,是中国最南部的热带滨海旅游城市 [gMASK]']
    en_to_ch = ['Pencil in Chinese is [MASK].']
    ch_to_en = ['"我思故我在"的英文是"[MASK]"。']

    examples = [en_fil, en_gen, ch_fil, ch_gen, en_to_ch, ch_to_en]

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            An Open Bilingual Pre-Trained Model. [Visit our github repo](https://github.com/THUDM/GLM-130B)
            GLM-130B uses two different mask tokens: `[MASK]` for short blank filling and `[gMASK]` for left-to-right long text generation. When the input does not contain any MASK token, `[gMASK]` will be automatically appended to the end of the text. We recommend that you use `[MASK]` to try text fill-in-the-blank to reduce wait time (ideally within seconds without queuing).
            """)

        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(lines=7, placeholder='Input something in English or Chinese', label='Input')
                with gr.Row():
                    gen = gr.Button("Generate")
                    clr = gr.Button("Clear")
                   
            outputs = gr.Textbox(lines=7, label='Output')
                
        gr.Markdown(
            """
            Generation Parameter
            """)
        with gr.Row():
            with gr.Column():
                seed = gr.Slider(maximum=100000, value=1234, step=1, label='Seed')
                out_seq_length = gr.Slider(maximum=256, value=128, minimum=32, step=1, label='Output Sequence Length')
            with gr.Column():
                min_gen_length = gr.Slider(maximum=64, value=0, step=1, label='Min Generate Length')
                sampling_strategy = gr.Radio(choices=['BeamSearchStrategy', 'BaseStrategy'], value='BeamSearchStrategy', label='Search Strategy')

        with gr.Row():
            with gr.Column():
                # beam search
                gr.Markdown(
                    """
                    BeamSearchStrategy
                    """)
                num_beams = gr.Slider(maximum=4, value=2, minimum=1, step=1, label='Number of Beams')
                length_penalty = gr.Slider(maximum=1, value=1, minimum=0, label='Length Penalty')
                no_repeat_ngram_size = gr.Slider(maximum=5, value=3, minimum=1, step=1, label='No Repeat Ngram Size')
            with gr.Column():
                # base search
                gr.Markdown(
                    """
                    BaseStrategy
                    """)
                temperature = gr.Slider(maximum=1, value=0.7, minimum=0, label='Temperature')
                topk = gr.Slider(maximum=40, value=1, minimum=0, step=1, label='Top K')
                topp = gr.Slider(maximum=1, value=0, minimum=0, label='Top P')
            
        inputs = [model_input, seed, out_seq_length, min_gen_length, sampling_strategy, num_beams, length_penalty, no_repeat_ngram_size, temperature, topk, topp]
        gen.click(fn=predict, inputs=inputs, outputs=outputs)
        clr.click(fn=lambda value: gr.update(value=""), inputs=clr, outputs=model_input)
        
        gr_examples = gr.Examples(examples=examples, inputs=model_input)
        
        gr.Markdown(
            """
            Disclaimer inspired from [BLOOM](https://huggingface.co/spaces/bigscience/bloom-book)
            
            GLM-130B was trained on web-crawled data, so it's hard to predict how GLM-130B will respond to particular prompts; harmful or otherwise offensive content may occur without warning. We prohibit users from knowingly generating or allowing others to knowingly generate harmful content, including Hateful, Harassment, Violence, Adult, Political, Deception, etc. 
            """)

    demo.launch()