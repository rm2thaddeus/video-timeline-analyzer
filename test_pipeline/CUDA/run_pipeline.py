#!/usr/bin/env python3
"""
📌 Purpose – Command-line interface for the CUDA-accelerated video processing pipeline
🔄 Latest Changes – Fixed indentation issues and improved logging configuration
⚙️ Key Logic – Parses arguments, sets up logging, ensures directories exist, and initiates the video processing pipeline
📂 Expected File Path – test_pipeline/CUDA/run_pipeline.py
🧠 Reasoning – To provide a user-friendly CLI entry point for pipeline execution with proper debugging via logs
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Assuming process_video is defined in the pipeline.py file
from pipeline import process_video


def setup_logging():
    log_dir = os.path.join('test_pipeline', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'cuda_pipeline_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info('Logging initialized. Log file: %s', log_file)


def parse_args():
    parser = argparse.ArgumentParser(description='CUDA-accelerated video processing pipeline')
    parser.add_argument('--video_path', required=True, help='Path to the input video')
    parser.add_argument('--output_dir', default='output', help='Directory to save output results')
    # Add additional arguments as needed
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    if not os.path.exists(args.video_path):
        logging.error('Video file not found: %s', args.video_path)
        sys.exit(1)
    logging.info('Starting video processing for: %s', args.video_path)
    try:
        process_video(args.video_path, args.output_dir)
    except Exception as e:
        logging.exception('Error during video processing: %s', e)
        sys.exit(1)
    logging.info('Video processing completed successfully.')


if __name__ == '__main__':
    main() 