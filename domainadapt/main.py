import sys
import json


def cmd_train(ctx):
    pass


def run_main():
    if len(sys.argv) <= 1:
        print("\ndomainadapt [config filename].json\n")
        return

    try:
        with open(sys.argv[1], "r") as fhandle:
            ctx = json.load(fhandle)
    except FileNotFoundError:
        print("\nFile {} not found !\n".format(sys.argv[1]))
        return

    command = ctx["command"]

    if command == 'train':
        cmd_train(ctx)
