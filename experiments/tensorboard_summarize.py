from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import sys
import glob


if len(sys.argv) != 2:
    raise Exception("wrong usage: python tensorboard_summarize.py <path_to_events>")

step = 350
metrics = {}
metrics_ema = {}
exceptions = ['learning_rate', 'consistency_weight']

print("Maybe need to pass absolute path of events, not tested")
path = sys.argv[1]

"""
Load each event file from root path and open them
"""
events = list(glob.iglob(os.path.join(path, '**/*.rosenberg'), recursive=True))
for e in events:
    event = EventAccumulator(e)
    event.Reload()

    # Metric name is in the path, not in the eventfile
    metric_name = '/'.join(e.rsplit('/', 2)[0:2])
    scalars = event.Tags()['scalars']
    for scalar in scalars:
        if scalar in exceptions:
            metrics[scalar] = event.Scalars(scalar)[step-1].value
        elif '_ema_' in metric_name:
            metrics_ema[metric_name] = event.Scalars(scalar)[step-1].value
        else:
            metrics[metric_name] = event.Scalars(scalar)[step-1].value

for key in sorted(metrics):
    print('%s: %.2f' % (key, metrics[key]))
for key in sorted(metrics_ema):
    print('%s: %.2f' % (key, metrics_ema[key]))
