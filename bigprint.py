# Helper class to send me Telegram notifications about the process status

import subprocess, time
from termcolor import colored

def launch(command, strict=False, wait=False):
    """
    Launches a subprocess with the given command, waits for its completion (or not) and returns the output
    """

    process = subprocess.Popen(command, shell=True) # launch the command in shell

    if not wait:
        return ""

    output, error = process.communicate() # get the output/error of the process

    if error:
        print(colored("ERROR LAUNCHING PROGRAM", 'red', attrs=['bold', 'blink']))
        print(error)
        if strict:
            raise Exception(error)

    return output # return the output

class BigPrint:
    def __init__(self, message_telegram: bool, wait: bool=False):
        self.message_telegram = message_telegram
        self.wait = wait

    def __call__(self, *args, **kwds):
        text = ''.join([str(a) for a in args]).replace('"', ' ')
        print(colored(text, color="white", attrs=['bold']))
        
        if self.message_telegram:
            launch(f'printf "{text}" | telegram-send --stdin', wait=self.wait)

class TimeCounter:
    def __init__(self):
        self.counter = 0
    
    def __enter__(self):
        self.counter = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.counter = time.time() - self.counter
    
    def __call__(self, format="secs"):
        if format == "mins":
            return int(self.counter / 60)
        return self.counter
