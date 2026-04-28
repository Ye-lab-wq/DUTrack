import argparse
from pathlib import Path

import cv2


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            yield path


def check_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return "cv2.imread returned None"
    if img.size == 0:
        return "cv2.imread returned empty image"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find non-zero-byte image files that OpenCV cannot read."
    )
    parser.add_argument("roots", nargs="+", help="One or more dataset roots to scan.")
    parser.add_argument(
        "--include-zero-byte",
        action="store_true",
        help="Also report zero-byte files. By default they are skipped.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional text file to save the bad file list.",
    )
    args = parser.parse_args()

    bad_files = []
    scanned = 0
    skipped_zero = 0

    for root_str in args.roots:
        root = Path(root_str).expanduser().resolve()
        if not root.exists():
            print(f"[missing] {root}")
            continue

        print(f"[scan] {root}")
        for path in iter_image_files(root):
            try:
                size = path.stat().st_size
            except OSError as exc:
                bad_files.append((str(path), f"stat failed: {exc}"))
                continue

            if size == 0 and not args.include_zero_byte:
                skipped_zero += 1
                continue

            scanned += 1
            reason = check_image(path)
            if reason is not None:
                bad_files.append((str(path), reason))

    for path, reason in bad_files:
        print(f"{path}\t{reason}")

    print()
    print(f"scanned_nonzero_images = {scanned}")
    print(f"skipped_zero_byte_images = {skipped_zero}")
    print(f"opencv_bad_images = {len(bad_files)}")

    if args.save:
        out_path = Path(args.save).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for path, reason in bad_files:
                f.write(f"{path}\t{reason}\n")
        print(f"saved_report = {out_path}")


if __name__ == "__main__":
    main()
