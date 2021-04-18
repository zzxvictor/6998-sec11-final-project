from datetime import datetime
import boto3


class EnforcementLogic:
    def __init__(self,
                 maxi_np_time=5,
                 maxi_slot_time=30,
                 sender='zz2777@columbia.edu',
                 receiver='zz2777@columbia.edu',
                 subject='Parking Violation Notification'):
        self.max_np = maxi_np_time
        self.max_slot = maxi_slot_time
        self.client = boto3.client('ses')
        self.sender = sender
        self.receiver = receiver
        self.subject = subject
        self.min_interval = 30
        self.last_send = None
        # for debug only
        self.counter = 0

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
        self._notification(records, current_time)
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

    def _notification(self, records, current_time):
        violations = [record for record in records if record['violation'] == 'Yes']
        if len(violations) == 0:
            return
        if self.last_send is not None and current_time - self.last_send < self.min_interval:
            return
        # debug only
        if self.counter > 2:
            return

        body = "New Violation Detected! \n"
        features = ['slot_id', 'duration', 'msg']
        width = [15, 5, 30]
        line_format = ''
        for i, w in enumerate(width):
            line_format += '{{{}: <{}}} '.format(i, w)
        for violation in violations:
            body += 'violation: ' + line_format.format(*[violation[feature] for feature in features]) + '\n'

        self.last_send = current_time
        self.counter += 1

        response = self.client.send_email(
            Destination={
                'ToAddresses': [
                    self.receiver,
                ],
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': 'UTF-8',
                        'Data': body,
                    },
                },
                'Subject': {
                    'Charset': 'UTF-8',
                    'Data': self.subject,
                },
            },
            Source=self.sender,
        )
