"""
Download OASIS-3 T1w scans from NITRC-IR (Windows-native, resumable).

Reads a list of MR session labels (one per line), downloads each session's
T1w scans as tar.gz, extracts the NIfTI + JSON into a BIDS-like layout, and
skips any session already present on disk.

Usage:
    python download_oasis.py <session_list.csv> <output_dir> <username>
"""
import sys, os, io, tarfile, getpass, time
import requests

BASE = "https://www.nitrc.org/ir/data/archive/projects/OASIS3"
CHUNK = 1 << 20  # 1 MB


def session_url(session_label, scan_type="T1w"):
    subject = session_label.split("_")[0]
    return (f"{BASE}/subjects/{subject}/experiments/{session_label}"
            f"/scans/{scan_type}/files?format=tar.gz")


def already_have(out_dir, session_label):
    """A session is considered done if its subject folder holds >=1 .nii.gz
    for that session."""
    subject = session_label.split("_")[0]
    ses = "ses-d" + session_label.rsplit("_d", 1)[1]
    anat = os.path.join(out_dir, f"sub-{subject}", ses, "anat")
    if not os.path.isdir(anat):
        return False
    return any(f.endswith(".nii.gz") for f in os.listdir(anat))


def extract_bids(tar_bytes, out_dir, session_label):
    """Pull NIfTI/JSON out of the XNAT tar and write to sub-XXX/ses-dXXX/anat/."""
    subject = session_label.split("_")[0]
    ses = "ses-d" + session_label.rsplit("_d", 1)[1]
    anat = os.path.join(out_dir, f"sub-{subject}", ses, "anat")
    os.makedirs(anat, exist_ok=True)

    n = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if not (name.endswith(".nii.gz") or name.endswith(".json")):
                continue
            # XNAT sometimes writes 'sess-' instead of BIDS 'ses-'
            name = name.replace("_sess-", "_ses-")
            src = tf.extractfile(m)
            if src is None:
                continue
            with open(os.path.join(anat, name), "wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    list_file, out_dir, username = sys.argv[1], sys.argv[2], sys.argv[3]
    password = getpass.getpass(f"NITRC password for {username}: ")

    with open(list_file) as fh:
        sessions = [ln.strip() for ln in fh if ln.strip()]

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{len(sessions)} sessions in list -> {out_dir}\n")

    sess = requests.Session()
    sess.auth = (username, password)

    done = skipped = failed = 0
    t0 = time.time()

    for i, label in enumerate(sessions, 1):
        prefix = f"[{i}/{len(sessions)}] {label}"

        if already_have(out_dir, label):
            print(f"{prefix} - already present, skipping")
            skipped += 1
            continue

        try:
            r = sess.get(session_url(label), timeout=(30, 300), stream=True)
            if r.status_code != 200:
                print(f"{prefix} - HTTP {r.status_code}, skipping")
                failed += 1
                continue

            buf = io.BytesIO()
            got = 0
            for chunk in r.iter_content(CHUNK):
                buf.write(chunk)
                got += len(chunk)
                print(f"\r{prefix} - {got/1e6:6.1f} MB", end="", flush=True)

            n = extract_bids(buf.getvalue(), out_dir, label)
            print(f"\r{prefix} - {got/1e6:6.1f} MB, {n} files extracted")
            done += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Re-run the same command to resume.")
            break
        except Exception as e:
            print(f"\r{prefix} - FAILED: {type(e).__name__}: {e}")
            failed += 1

    mins = (time.time() - t0) / 60
    print(f"\n{'='*55}")
    print(f"Downloaded : {done}")
    print(f"Skipped    : {skipped} (already present)")
    print(f"Failed     : {failed}")
    print(f"Elapsed    : {mins:.1f} min")
    if failed:
        print("\nRe-run the same command to retry failed sessions.")


if __name__ == "__main__":
    main()