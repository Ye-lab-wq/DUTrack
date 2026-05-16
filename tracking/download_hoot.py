import argparse
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import requests


BASE_URL = 'http://ilab.usc.edu/hoot/v1_0/HD/'


def fetch_json(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def write_additional_files(metadata, destination):
    for filename in metadata.get('additional_files', []):
        out_path = destination / filename
        if out_path.exists():
            print('skip additional {}'.format(filename), flush=True)
            continue
        response = requests.get(BASE_URL + filename, timeout=60)
        response.raise_for_status()
        out_path.write_text(response.text)
        print('wrote additional {}'.format(filename), flush=True)


def archive_complete(path, expected_size):
    return path.exists() and path.stat().st_size == expected_size


def download_archive(url, archive_path, expected_size, retries):
    tmp_path = Path(str(archive_path) + '.tmp')
    if tmp_path.exists() and not archive_path.exists():
        tmp_path.rename(archive_path)

    if archive_complete(archive_path, expected_size):
        return

    for attempt in range(1, retries + 1):
        print('download attempt {}/{} {}'.format(attempt, retries, archive_path), flush=True)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([
            'wget', '-c',
            '--tries=20',
            '--timeout=60',
            '--read-timeout=60',
            '--progress=dot:giga',
            '-O', str(archive_path),
            url,
        ])
        if archive_complete(archive_path, expected_size):
            return
        size = archive_path.stat().st_size if archive_path.exists() else 0
        print('incomplete archive: {} / {}'.format(size, expected_size), flush=True)
        time.sleep(min(30, 5 * attempt))

    raise RuntimeError('Failed to download complete archive {}'.format(archive_path))


def extract_archive(archive_path, video_dir):
    video_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'r') as zip_file:
        zip_file.extractall(video_dir)


def main():
    parser = argparse.ArgumentParser(description='Download and extract HOOT HD test split.')
    parser.add_argument('--dest', default='/media/b520/KESU1/HOOT')
    parser.add_argument('--split', default='test', choices=('test', 'all'))
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--keep_archives', action='store_true')
    args = parser.parse_args()

    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)

    print('fetch metadata', flush=True)
    metadata = fetch_json(BASE_URL + 'metadata.json')
    write_additional_files(metadata, destination)

    videos = []
    for class_info in metadata['classes']:
        class_dir = destination / class_info['name']
        class_dir.mkdir(exist_ok=True)
        for video in class_info['videos']:
            if args.split == 'all' or video.get('test_split'):
                videos.append((class_info['name'], video))

    download_size = sum(int(video['download_size']) for _, video in videos)
    install_size = sum(int(video['install_size']) for _, video in videos)
    print('videos: {}'.format(len(videos)), flush=True)
    print('download_GB: {:.2f}'.format(download_size / 1024 ** 3), flush=True)
    print('install_GB: {:.2f}'.format(install_size / 1024 ** 3), flush=True)

    for index, (class_name, video) in enumerate(videos, 1):
        video_key = '{}-{}'.format(class_name, video['id'])
        video_dir = destination / class_name / video['id']
        if (video_dir / 'anno.json').exists():
            print('[{}/{}] skip extracted {}'.format(index, len(videos), video_key), flush=True)
            continue

        archive_path = destination / video['path']
        expected_size = int(video['download_size'])
        print('[{}/{}] {}'.format(index, len(videos), video_key), flush=True)
        download_archive(BASE_URL + video['path'], archive_path, expected_size, args.retries)
        extract_archive(archive_path, video_dir)
        if not args.keep_archives:
            archive_path.unlink()
        print('[{}/{}] done {}'.format(index, len(videos), video_key), flush=True)

    print('HOOT ready at {}'.format(destination), flush=True)


if __name__ == '__main__':
    main()
