from pathlib import Path


def setup(app):
    # Get png and csv files and other stuff from the AGTS scripts that run
    # every weekend:
    try:
        import gpaw_web_page_data
    except ImportError:
        print("""
Please clone and pip install the gpaw_web_page_data package
from git@gitlab.com:gpaw/gpaw_web_page_data.git or use:

   cd root-of-gpaw-clone
   pip install -e .[docs]

""")
        raise
    data = Path(gpaw_web_page_data.__file__).parent
    print('Using gpaw-web-page-data from', data)
    doc = Path()
    for path in data.glob('**/*.*'):
        if path.name.startswith('_'):
            continue
        fro = doc / path.relative_to(data)
        if not fro.is_file():
            print(fro, '->', path)
            if fro.is_symlink():
                fro.unlink()
            fro.symlink_to(path)


if __name__ == '__main__':
    setup(None)
