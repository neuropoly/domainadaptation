"""
needs GPUtil
"""
import GPUtil
import json
import main
import time
from threading import Thread

gpus = GPUtil.getGPUs()

"""
different experiments, each one is a dictionary with the changes performed at the base json
"""
experiments = [
    {
        "adapt_centers": [4],
        "val_centers": [3,4],
        "consistency_loss": "dice_loss",
    },
    {
        "adapt_centers": [4],
        "val_centers": [3,4],
        "consistency_loss": "mse",
    },
    {
        "adapt_centers": [4],
        "val_centers": [3,4],
        "consistency_loss": "cross_entropy",
    },
    {
        "ema_alpha": 0.9,
        "ema_alpha_late": 0.99,
        "ema_late_epoch": 50,
    },
    {
        "ema_alpha": 0.99,
        "ema_alpha_late": 0.999,
        "ema_late_epoch": 100,
    },
]

threads = {}
for i in range(0, len(gpus)):
    threads[str(i)] = None

for exp in experiments:
    print(threads)
    with open('../experiments/domain_adaptation.json') as f:
        base_experiment = json.load(f)
        for k in exp:
            base_experiment[k] = exp[k]
    done = False
    while True:
        for i, gpu in enumerate(gpus):
            if int(gpu.memoryFree) > 8000 and threads[str(i)] is None:
                done = True
                base_experiment["gpu"] = str(i)
                thread = Thread(target=main.run_main, args=(base_experiment,))
                print("Starting new thread for experiment: %s in gpu %s" % (exp, str(i)))
                threads[str(i)] = thread
                thread.start()
                break
        for k in threads:
            if threads[k] is not None and not threads[k].is_alive():
                threads[k] = None

        if done:
            break
        time.sleep(1)

for t in threads:
    t.join()
