"""Extract QuoridorCPP to HY, verifying and retaining already completed files."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import zipfile
import zlib
import fcntl
from datetime import datetime, timezone

SOURCE = Path(__file__).resolve().parents[1] / 'data/datasets/quoridorcpp-208k/quoridorcpp-208k.zip'
VOLUME = Path('/Volumes/HY')
DEST = VOLUME / 'QuoridorData/quoridorcpp-208k'
STATUS = SOURCE.parent / 'extract-status.json'
EXPECTED_SHA256 = '1b63b071f8648765ef172519abba161d4f9b63402c7367d69e654ae3a9700e00'
state = {'state':'checking', 'pid':os.getpid(), 'archive':str(SOURCE), 'destination':str(DEST), 'completed_files':0, 'completed_bytes':0, 'verified_files':0, 'extracted_files':0}
started = time.monotonic()
last_report = 0

def report(**fields):
    global last_report
    state.update(fields)
    state.update(updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=round(time.monotonic()-started, 1))
    temporary = STATUS.with_name(f'.{STATUS.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(STATUS)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    last_report = time.monotonic()

lock = (SOURCE.parent / 'extract-worker.lock').open('a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError('Another extraction worker is already running') from None

try:
    report()
    if not VOLUME.is_mount():
        raise RuntimeError('HY volume is not mounted')
    digest = hashlib.sha256()
    with SOURCE.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('Archive SHA256 does not match the verified download')
    with zipfile.ZipFile(SOURCE) as archive:
        members = []
        paths = set()
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or '__MACOSX' in path.parts or path.name.startswith('._') or path.name=='.DS_Store':
                continue
            if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError('Unsafe archive member: '+info.filename)
            if path in paths:
                raise RuntimeError('Duplicate archive output: '+info.filename)
            paths.add(path)
            members.append(info)
        required = sum(info.file_size for info in members)
        if shutil.disk_usage(VOLUME).free < required + 10*1024**3:
            raise RuntimeError('HY does not have sufficient free space')
        if DEST.is_symlink() or not DEST.resolve().is_relative_to(VOLUME.resolve()):
            raise RuntimeError('Destination does not resolve inside HY')
        DEST.mkdir(parents=True, exist_ok=True)
        report(state='extracting', total_files=len(members), total_bytes=required, archive_sha256=digest.hexdigest())
        parents = set()
        for info in members:
            if not VOLUME.is_mount():
                raise RuntimeError('HY was disconnected')
            target = DEST.joinpath(*PurePosixPath(info.filename).parts)
            if target.parent not in parents:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.parent.resolve().is_relative_to(DEST.resolve()):
                    raise RuntimeError('Output path escapes destination')
                parents.add(target.parent)
            if target.is_symlink():
                raise RuntimeError('Output is a symbolic link: '+str(target))
            if target.exists():
                if not target.is_file() or target.stat().st_size != info.file_size:
                    raise RuntimeError('Existing file size differs; original retained: '+str(target))
                crc = 0
                with target.open('rb') as existing:
                    for chunk in iter(lambda:existing.read(4*1024*1024), b''):
                        crc = zlib.crc32(chunk, crc)
                        if time.monotonic()-last_report >= 10:
                            report(phase='verifying_existing', current_file=info.filename)
                if (crc & 0xffffffff) != info.CRC:
                    raise RuntimeError('Existing file CRC differs; original retained: '+str(target))
                state['verified_files'] += 1
            else:
                temporary = target.with_name('.'+target.name+'.extracting.'+str(os.getpid()))
                with archive.open(info) as source, temporary.open('xb') as output:
                    shutil.copyfileobj(source, output, length=4*1024*1024)
                if target.exists():
                    raise RuntimeError('Output appeared during extraction: '+str(target))
                temporary.rename(target)
                state['extracted_files'] += 1
            state['phase'] = 'extracting'
            state['completed_files'] += 1
            state['completed_bytes'] += info.file_size
            if time.monotonic()-last_report >= 10:
                report(current_file=info.filename)
        report(state='complete', phase='complete', current_file=None)
except BaseException as error:
    report(state='failed', error=str(error))
    raise

finally:
    lock.close()
