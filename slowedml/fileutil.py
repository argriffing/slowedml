"""
File utilities.
"""

import sys

class open_or_special:
    """
    Open a file or use a special file.
    """

    def __init__(self, special_string, special_file, filename, *args, **kwargs):
        self.special_file = special_file
        self.special_string = special_string
        self.filename = filename
        self.open_args = args
        self.open_kwargs = kwargs

    def __enter__(self):
        if self.filename == self.special_string:
            self.fout = self.special_file
        else:
            self.fout = open(self.filename, *self.open_args, **self.open_kwargs)
        return self.fout

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fout not in (None, self.special_file):
            self.fout.close()


class open_or_stdout(open_or_special):
    def __init__(self, filename, *args, **kwargs):
        open_or_special.__init__(
                self, '-', sys.stdout, filename, *args, **kwargs)

class open_or_stdin(open_or_special):
    def __init__(self, filename, *args, **kwargs):
        open_or_special.__init__(
                self, '-', sys.stdin, filename, *args, **kwargs)
