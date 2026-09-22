"""Build/install a wheel offline, replay outside the checkout, and test installed code.

Dependencies are symlinked from the invoking environment, not downloaded. Project
packages and .pth files are excluded; the project itself must come from its wheel.
"""
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def call(args, cwd, env):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def main():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env['PIP_NO_INDEX'] = '1'
    with tempfile.TemporaryDirectory(prefix='mqar-installed-') as tmp:
        tmp = Path(tmp)
        wheel_dir, isolated = tmp / 'wheels', tmp / 'venv'
        call([sys.executable, '-m', 'pip', 'wheel', '--no-index', '--no-deps',
              '--no-build-isolation', '--wheel-dir', wheel_dir, ROOT], tmp, env)
        venv.EnvBuilder(with_pip=False).create(isolated)
        python = isolated / 'bin/python'
        site = Path(subprocess.check_output([str(python), '-I', '-c',
                    'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip())
        original_site = Path(sysconfig.get_paths()['purelib'])
        for source in original_site.iterdir():
            if ('subtoken' in source.name or source.suffix == '.pth' or
                    source.name.startswith('__editable__') or source.name == '__pycache__'):
                continue
            (site / source.name).symlink_to(source, target_is_directory=source.is_dir())
        wheels = list(wheel_dir.glob('subtoken_attention_cpu_lab-*.whl'))
        assert len(wheels) == 1
        call([sys.executable, '-m', 'pip', '--python', python, 'install',
              '--no-index', '--no-deps', wheels[0]], tmp, env)
        code = ('from pathlib import Path; import subtoken_lab.mqar as m; '
                f'p=Path(m.__file__).resolve(); assert p.is_relative_to(Path({str(isolated)!r})); '
                'print("Installed module:", p)')
        call([python, '-I', '-c', code], tmp, env)
        call([python, '-I', '-m', 'unittest', 'discover', '-s', ROOT / 'tests', '-v'], tmp, env)
        call([isolated / 'bin/subtoken-mqar', 'verify', '--out', ROOT / 'results/mqar'], tmp, env)
        call([isolated / 'bin/subtoken-mqar', 'report', '--out', ROOT / 'results/mqar',
              '--check', ROOT / 'MQAR_RESULTS.md'], tmp, env)
    print('Offline wheel installation, external-directory CLI replay and installed-code tests passed.')


if __name__ == '__main__':
    main()
