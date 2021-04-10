import os


class Printer:
    @classmethod
    def print(cls, records, timestamp):
        slot_status = []
        for record in records:
            slot = record['slot_id'].replace('_', ' ')
            status = 'occupied' if record['status'] == 1 else 'empty'
            time_elapsed = timestamp - float(record['since']) if status == 'occupied' else 0
            slot_status.append((slot, status, time_elapsed))
        sorted(slot_status, key=lambda x: x[0])
        cls._pprint(slot_status)

    @classmethod
    def _pprint(cls, slot_status):
        os.system('cls||clear')
        cls._print_msg_box('Parking monitoring in progress',
                           title='Parking time analysis',
                           indent=4,
                           width=50)
        header = '\t| {0: <20} | {1: <10} | {2: <20} |'.format(*['slot_id', 'status', 'parked time (secs)'])
        print('\t' + '-' * len(header))
        print(header)
        print('\t' + '-' * len(header))
        for line in slot_status:
            string = '\t| {0: <20} | {1: <10} | {2: <20} |'.format(*line)
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