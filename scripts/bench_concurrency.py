import json, time, urllib.request, sys, concurrent.futures

BASE = "http://127.0.0.1:8080"
MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=15))["data"][0]["id"]
PROMPT = ("用 Python 写一个 LRU 缓存类，带容量限制和过期时间，要求线程安全，"
          "并解释你的设计取舍。请先分析再给出完整代码。")
MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 256


def one(i):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": PROMPT + f" 编号{i}"}],
               "max_tokens": MAXTOK, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    return time.time() - t0, d.get("usage", {}).get("completion_tokens", 0)


print("model:", MODEL, "| max_tokens:", MAXTOK, flush=True)
for c in [1, 2, 4, 8]:
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(one, range(c)))
    wall = time.time() - t0
    tot = sum(x[1] for x in res)
    per = [round(x[1] / x[0], 1) if x[0] else 0 for x in res]
    print(f"C={c}: wall={wall:6.1f}s total_completion={tot:5d} "
          f"aggregate={tot/wall:6.1f} tok/s  per-stream={per}", flush=True)
