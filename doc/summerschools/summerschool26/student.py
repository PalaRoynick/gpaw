# creates: magnetism/CrI3_gs_student.py
# creates: magnetism/get_Tc_mf_student.py
# creates: magnetism/CrI3_anisotropy_student.py
# creates: magnetism/VI2_gs_student.py
# creates: magnetism/VI2_afm_student.py

from pathlib import Path


def main():
    for path in Path().glob('*/*_teacher.py'):
        print(path)
        lines = []
        for line in path.read_text().splitlines():
            if ' # student:' in line:
                a, b = (x.strip() for x in line.split('# student:'))
                line = line.split(a)[0] + b + '\n'
            lines.append(line)
        path.with_name(path.name.replace('_teacher', '_student')).write_text(
            '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
