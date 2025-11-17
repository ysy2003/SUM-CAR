from dataclasses import dataclass


@dataclass
class Logger:
    prefix: str = ""
    def log(self, *args):
        print(self.prefix, *args)