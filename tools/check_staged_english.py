"""Reject Cyrillic in staged text files and paths before committing.

Review prose manually as well: this guard is not a general language detector.
It examines complete staged files, not just added lines, and leaves Git intact.
"""
import re
import subprocess
import sys


def git(*args):
    return subprocess.check_output(['git', *args])


def main():
    cyrillic = re.compile('[\u0400-\u052f\u2de0-\u2dff\ua640-\ua69f]')
    names = git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z')
    violations = []
    for raw in names.split(b'\0'):
        if not raw:
            continue
        name = raw.decode('utf-8')
        if cyrillic.search(name):
            violations.append(f'{name}: filename contains Cyrillic')
        entry = git('ls-files', '--stage', '--', name)
        if entry.startswith(b'160000 '):  # A submodule commit, not a text blob.
            continue
        data = git('show', ':' + name)
        if b'\0' in data:
            continue
        try:
            content = data.decode('utf-8')
        except UnicodeDecodeError:
            violations.append(f'{name}: not UTF-8; review its format and language')
            continue
        for line, text in enumerate(content.splitlines(), 1):
            if cyrillic.search(text):
                violations.append(f'{name}:{line}: Cyrillic is not allowed')
    if violations:
        print('\n'.join(violations), file=sys.stderr)
        return 1
    print('Staged language guard passed. Manually verify English-only prose.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
