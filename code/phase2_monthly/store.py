"""Append-only, atomically published records, separated by epistemic role."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def clone(value):
    return json.loads(canonical(value))


class RecordStore:
    KINDS = {'config', 'profile', 'state', 'event', 'interpretation', 'plan',
             'choice', 'execution', 'audit', 'recall', 'request', 'response', 'failure'}

    def __init__(self, root, namespace):
        if set(namespace) != {'household', 'condition', 'repeat'}:
            raise ValueError('namespace requires household / condition / repeat')
        self.namespace = clone(namespace)
        self.root = Path(root) / digest(namespace)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def lock(self):
        with (self.root / '.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def path(self, kind, key):
        if kind not in self.KINDS or not re.fullmatch(r'[a-zA-Z0-9_-]+', key):
            raise ValueError('invalid record location')
        return self.root / kind / f'{key}.json'

    def get(self, kind, key):
        path = self.path(kind, key)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record['namespace'] != self.namespace or record['data_hash'] != digest(record['data']):
            raise ValueError('record identity/hash mismatch')
        return record

    def put(self, kind, key, month, data, available_month=None):
        data = clone(data)
        old = self.get(kind, key)
        availability = month if available_month is None else available_month
        if old:
            if old['data'] != data or old['month'] != month or old['available_month'] != availability:
                raise ValueError(f'immutable record conflict: {kind}/{key}')
            return old
        record = dict(id=f'{kind}/{key}', namespace=self.namespace, kind=kind, month=month,
                      available_month=availability, created_at=datetime.now(timezone.utc).isoformat(),
                      data=clone(data), data_hash=digest(data))
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
        try:
            with os.fdopen(fd, 'w') as handle:
                handle.write(canonical(record) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            # Hard link publishes complete bytes and refuses replacing an existing record.
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(temporary)
        return record

    def list(self, kind):
        directory = self.root / kind
        return [self.get(kind, path.stem) for path in sorted(directory.glob('*.json'))]
