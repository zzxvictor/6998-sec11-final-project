import os


class Printer:
    def __init__(self,
                 features=None,
                 width=None):
        if features is None:
            features = ['slot_id', 'status', 'duration', 'violation', 'msg']
            width = [20, 10, 10, 10, 30]
        self.features = features
        self.width = width

    def print(self, records):
        info = []
        for record in records:
            line = [record[column] for column in self.features]
            info.append(line)
        self._pprint(info)

    def _pprint(self, slot_status):
        os.system('cls||clear')
        self._print_msg_box('Parking monitoring in progress',
                            title='Parking time analysis',
                            indent=4,
                            width=sum(self.width))
        line_format = '\t'
        for i, w in enumerate(self.width):
            line_format += '| {{{}: <{}}} |'.format(i, w)
        header = line_format.format(*self.features)
        print('\t' + '-' * len(header))
        print(header)
        print('\t' + '-' * len(header))
        for line in slot_status:
            #print(line)
            string = line_format.format(*line)
            print(string)
        print('\t' + '-' * len(header))

    @classmethod
    def _print_msg_box(cls, msg, indent=1, width=None, title=None, padding='\t'):
        """Print message-box with optional title."""
        lines = msg.split('\n')
        space = " " * indent
        if not width:
            width = max(map(len, lines))
        box = padding + f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
        if title:
            box += padding + f'║{space}{title:<{width}}{space}║\n'  # title
            box += padding + f'║{space}{"-" * width:<{width}}{space}║\n'  # underscore
        box += ''.join([padding + f'║{space}{line:<{width}}{space}║\n' for line in lines])
        box += padding + f'╚{"═" * (width + indent * 2)}╝'  # lower_border
        print(box)