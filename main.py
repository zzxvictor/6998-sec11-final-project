import argparse
from configs import config
import pathlib
from system.video_streaming import  VideoReader
from system.video_analyzer import VideoAnalyzer
from system.result_printer import Printer
from system.enforcement_logic import EnforcementLogic

# python main.py -vp I:\Data\CNR-EXT-Patches-150x150\video_2.mp4 -ap I:\Data\CNR-EXT-Patches-150x150\video_2_annotation.txt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vp', dest='vp', type=pathlib.Path)
    parser.add_argument('-ap', dest='ap', type=pathlib.Path)
    args = parser.parse_args()

    # streaming service
    reader = VideoReader(args.vp, args.ap, scan_freq=10)
    # analysis service
    analyzer = VideoAnalyzer(classifier_path=config.CLASSIFIER_MODEL_PATH,
                             signature_path=config.SIGNATURE_MODEL_PATH,
                             s3_bucket=config.S3_BUCKET_NAME,
                             dynamo_db=config.DYNAMO_TABLE_NAME)
    # notification service
    enforcement = EnforcementLogic(
        maxi_np_time=5,
        maxi_slot_time=30,
        sender='zz2777@columbia.edu',
        receiver='zz2777@columbia.edu'
    )
    printer = Printer()
    # start processing the video
    feed = reader.start()
    for img_dict, timestamp in feed:
        records = analyzer.record(img_dict, timestamp)
        records = enforcement.check_all(records, timestamp)
        printer.print(records)