import numpy as np
import boto3
from botocore.exceptions import ClientError
import cv2
from decimal import Decimal
import io
import matplotlib.image as mpimg
from system.model_loader import LoadModelTF
from configs.config import DETECTOR_IMG_SIZE


class VideoAnalyzer:
    def __init__(self,
                 classifier_path,
                 signature_path,
                 s3_bucket,
                 dynamo_db,
                 video_name='100th St'):
        self.classifier_path = classifier_path
        self.signature_path = signature_path
        self.s3_bucket = s3_bucket
        self.db_table = dynamo_db
        self.video_name = video_name
        self.db = boto3.resource('dynamodb')
        self.table = self.db.Table(self.db_table)
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        self._purge()

        self.detector = LoadModelTF.load_model(classifier_path)
        self.signature = None
        #self.signature = LoadModelTorch.load_model(signature_path)

    def record(self, img_dict, timestamp):
        records, imgs = [], []
        slot_ids = list(sorted(img_dict.keys()))
        slot_names = ['_'.join([self.video_name, slot_id]) for slot_id in slot_ids]
        for slot_id in slot_ids:
            img = cv2.resize(img_dict[slot_id], DETECTOR_IMG_SIZE) / 255.0 - 0.5
            imgs.append(img)
        imgs = np.array(imgs)

        slots_is_occupied = self._classify(imgs)
        db_data = self._batch_query_by_id(slot_names)

        for idx, slot_name in enumerate(slot_names):
            slot_info = [info for info in db_data[self.db_table]
                         if info['slot_id'] == slot_name]
            record = {'slot_id': slot_name,
                      'since': Decimal(timestamp)}
            # if occupied
            if slots_is_occupied[idx]:
                if len(slot_info) > 0 and slot_info[0]['status'] == 1:
                    old_img = self._download_img_s3(slot_info[0]['img_path'])
                    # check if the two images match
                    if self._signature(imgs[idx], old_img):
                        record['since'] = slot_info[0]['since']
                self._upload_img_s3(imgs[idx], slot_name)
                record['status'] = 1
                record['img_path'] = slot_name + '.jpg'
            # not occupied
            else:
                record['status'] = 0
                record['img_path'] = ''
            records.append(record)
        self._batch_update_by_id(records)
        return records

    def _classify(self, imgs):
        pred = self.detector.predict(imgs).reshape(-1)
        return pred >= 0.5

    def _signature(self, img1, img_2):
        self.signature
        return True

    def _download_img_s3(self, obj_key):
        bucket = self.s3_resource.Bucket(self.s3_bucket)
        obj = bucket.Object(obj_key)
        img = mpimg.imread(io.BytesIO(obj.get()['Body'].read()), 'jp2')
        return img

    def _upload_img_s3(self, img, name):
        key = name + '.jpg'
        data = cv2.imencode('.jpg', img)[1].tostring()
        self.s3_resource.Object(self.s3_bucket, key).put(Body=data,ContentType='image/JPG')

    def _batch_update_by_id(self, records):
        with self.table.batch_writer() as batch:
            for item in records:
                batch.put_item(Item=item)

    def _batch_query_by_id(self, slots):
        data = [{'slot_id': slot_id} for slot_id in slots]
        response = self.db.batch_get_item(
            RequestItems={
                self.db_table: {
                    'Keys': data
                }
            }
        )
        return response['Responses']

    def _purge(self):
        scan = self.table.scan()
        with self.table.batch_writer() as batch:
            for each in scan['Items']:
                batch.delete_item(
                    Key={
                        'slot_id': each['slot_id'],
                    }
                )



