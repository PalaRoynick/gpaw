import subprocess


def test_gpw_command(gpw_files):
    gpwfile = gpw_files['bcc_li_pw']
    process = subprocess.run(['gpaw', '-T', 'gpw', str(gpwfile)],
                             capture_output=True, check=True,
                             encoding='utf-8')
    print(process.stdout)
    refstring = 'atomic_density_matrices: ndarray(shape=(1, 15), dtype=float64'
    assert refstring in process.stdout
