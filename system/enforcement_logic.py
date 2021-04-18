class EnforcementLogic:
    def __init__(self,
                 maxi_np_time=5,
                 maxi_slot_time=30):
        self.max_np = maxi_np_time
        self.max_slot = maxi_slot_time

    def check_all(self, records, current_time):
        for record in records:
            status = 'occupied' if record['status'] == 1 else 'empty'
            time_elapsed = current_time - float(record['since']) if status == 'occupied' else 0
            record['status'] = status
            record['duration'] = time_elapsed
            if 'NP' in record['slot_id']:
                violation, msg = self._check_np_violation(record)
            elif 'SLOT' in record['slot_id']:
                violation, msg = self._check_slot_violation(record)
            else:
                # potentially other violation logic
                violation = False
                msg = ''
                pass
            record['violation'] = 'Yes' if violation else 'No'
            record['msg'] = msg
        return records

    def _check_np_violation(self, record):
        if record['duration'] >= self.max_np:
            return True, 'No parking violation!'
        else:
            return False, ''

    def _check_slot_violation(self, record):
        if record['duration'] >= self.max_slot:
            return True, 'overtime parking!'
        else:
            return False, ''