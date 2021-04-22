S3_BUCKET_NAME = 'parking-slot-image-bucket'
DYNAMO_TABLE_NAME = 'parking-lot-database'
CLASSIFIER_MODEL_PATH = 'logs/vehicle_detection/torch_detector.pt'
SIGNATURE_MODEL_PATH = 'logs/vehicle_signature/model.pt'
TEMP_CACHE = '.cache'
DETECTOR_IMG_SIZE = (150, 150)

DATA_ROOT = 'I:/Data/CNR-EXT-Patches-150x150/'
ANNOTATION_FILES = DATA_ROOT + 'LABELS/all.txt'
