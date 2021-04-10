import numpy as np
import boto3
from botocore.exceptions import ClientError
import os
import cv2
from decimal import Decimal
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class VideoAnalyzer:
    def __init__(self,
                 classifier_path,
                 signature_path,
                 s3_bucket,
                 dynamo_db,
                 video_name='100th st'):
        self.classifier_path = classifier_path
        self.signature_path = signature_path
        self.s3_bucket = s3_bucket
        self.db_table = dynamo_db

        self.video_name = video_name
        self.classifier = self._load_model(self.classifier_path)
        self.signature = self._load_model(self.signature_path)

        self.db = boto3.resource('dynamodb')
        self.table = self.db.Table(self.db_table)
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        self._purge()

    def record(self, img_dict, timestamp):
        records = []
        for slot_id in img_dict.keys():
            img = img_dict[slot_id]
            is_empty = self._classify(img)
            slot_name = '_'.join([self.video_name, 'slot', slot_id])
            data = self._query_db_by_id(slot_name)

            if not is_empty:
                obj_key = self._upload_img_s3(img, slot_name)
                record = {'slot_id': slot_name,
                          'status': 1,
                          'img_path': obj_key,
                          'since': Decimal(timestamp)}
                if 'Item' in data:
                    old_img = self._download_img_s3(data['Item']['img_path'])
                    is_match = self._signature(img, old_img)
                    if is_match:
                        record['since'] = data['Item']['since']
            else:
                record = {'slot_id': slot_name,
                          'status': 0,
                          'img_path': '',
                          'since': Decimal(-1)}
            records.append(record)
        self._update_db_by_id(records)
        return records

    def _classify(self, img):
        return False

    def _signature(self, img1, img_2):
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
        return key

    def _update_db_by_id(self, records):
        with self.table.batch_writer() as batch:
            for item in records:
                batch.put_item(Item=item)

    def _query_db_by_id(self, slot_id):
        try:
            response = self.table.get_item(Key={'slot_id': slot_id})
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            return response

    def _purge(self):
        scan = self.table.scan()
        with self.table.batch_writer() as batch:
            for each in scan['Items']:
                batch.delete_item(
                    Key={
                        'slot_id': each['slot_id'],
                    }
                )

    @classmethod
    def _load_model(cls, model_path):
        return None



