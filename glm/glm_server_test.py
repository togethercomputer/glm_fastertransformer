import requests
import json
import time
from icetk_glm_130B import _IceTokenizer
tokenizer = _IceTokenizer()

texts = [
        "凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。",
        "The Starry Night is an oil-on-canvas painting by [MASK] in June 1889.",
        "李白，字太白，号青莲居士，又号“谪仙人”，唐代伟大的浪漫主义诗人，被后人誉为“诗仙”。我：今天我们穿越时空连线李白，请问李白你爱喝酒吗？李白：花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。我：你为何能如此逍遥？李白：天生我材必有用，千金散尽还复来！我：你去过哪些地方？李白：",
        "I have a dream ",
        "Eight planets in solar system are [gMASK]",
    ]

def test(texts, strategy = "BaseStrategy", stop = [], regix = ""):

    # If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
    data = {
        "prompt": texts,
        "max_tokens": 64,
        "min_tokens": 0,
        "top_k": 1,
        "top_p": 0,
        "temperature": 1,
        "seed": 42,
        "sampling_strategy": strategy,
        "num_beams": 4,
        "length_penalty": 0.9,
        "no_repeat_ngram_size": 3,
        "regix": regix
    }

    t = time.time()
    res = requests.post("http://localhost:5000/generate", json=data).content.decode()
    t = time.time() - t

    res = json.loads(res)
    print(res['text'], end='\n\n')

    for generate, text in zip(res['text'],texts):
        generate.append('')
        generate = generate[0]
        generate = "\x1B[4m" + generate.replace("[[gMASK]]","") + "\x1B[0m"
        if "MASK" in text:
            print(text.replace("[gMASK]", generate).replace("[MASK]", generate))
        else:
            print(text + generate)
        print()

    print("time cost:", t)
    print()

test(texts[:1])
test(texts[:2])
test(texts[2:3], regix=".*我:.*")
test(texts)

test(texts[:1], strategy = "BeamSearchStrategy")
test(texts[:2], strategy = "BeamSearchStrategy")
test(texts[2:3], strategy = "BeamSearchStrategy", regix=".*我:.*")
test(texts, strategy = "BeamSearchStrategy")
