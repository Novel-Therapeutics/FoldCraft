"""Real process tests for hangs, owned descendants and cancellation; no GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from run_watchdog import Outcome, run_owned, supervise
from run_state import completed_run

ROOT = Path(__file__).resolve().parents[1]


def test_timeout_kills_stubborn_child_but_not_unrelated_process(tmp_path):
    heartbeat = tmp_path/'heartbeat'
    # The leader exits on TERM; the descendant ignores it. Both must be cleaned.
    child = ('import signal,time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
             f'p=Path({str(heartbeat)!r}); '
             '\nwhile True: p.write_text(str(time.monotonic())); time.sleep(.02)')
    leader = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(60)'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
    try:
        result = run_owned([sys.executable, '-c', leader], 1., grace_seconds=.2)
        assert result.returncode == 124 and result.reason == 'timed_out'
        assert result.seconds < 5 and heartbeat.exists()
        value = heartbeat.read_text(); time.sleep(.15)
        assert heartbeat.read_text() == value
        assert unrelated.poll() is None
    finally:
        unrelated.terminate(); unrelated.wait(timeout=5)


def test_sigterm_to_supervisor_cleans_worker_and_records_interruption(tmp_path):
    ready = tmp_path/'ready'; result = tmp_path/'result.json'
    worker = f'from pathlib import Path;import time;Path({str(ready)!r}).touch();time.sleep(60)'
    code = ('from run_watchdog import run_owned;import sys,json;from pathlib import Path;'
            f'r=run_owned([sys.executable,"-c",{worker!r}],30,grace_seconds=.2);'
            f'Path({str(result)!r}).write_text(json.dumps(vars(r)))')
    supervisor = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT)
    try:
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists()
        supervisor.send_signal(signal.SIGTERM); supervisor.wait(timeout=5)
        assert json.loads(result.read_text())['reason'] == 'interrupted'
        assert json.loads(result.read_text())['returncode'] == 143
    finally:
        if supervisor.poll() is None:supervisor.kill();supervisor.wait()


def test_normal_exit_and_worker_failure_are_preserved():
    assert run_owned([sys.executable,'-c','pass'],5).returncode == 0
    assert run_owned([sys.executable,'-c','raise SystemExit(7)'],5).returncode == 7


@pytest.mark.parametrize('value', [0,-1,float('nan'),float('inf')])
def test_invalid_timeout_never_starts_process(value):
    with pytest.raises(ValueError):run_owned(['nonexistent-worker'],value)


def test_timeout_preserves_partial_checkpoint_and_cannot_complete(tmp_path,monkeypatch):
    import run_watchdog
    (tmp_path/'run.json').write_text(json.dumps(dict(schema=1,status='running')))
    partial=tmp_path/'results.csv.partial';partial.write_text('name,sequence\nkept,AAA\n')
    monkeypatch.setattr(run_watchdog,'run_owned',lambda *a,**k:Outcome(124,'timed_out',1.))
    with pytest.raises(SystemExit) as e:supervise(SimpleNamespace(output_folder=tmp_path,timeout_minutes=1.))
    assert e.value.code==124
    assert json.loads((tmp_path/'run.json').read_text())['status']=='timed_out'
    assert partial.read_text()=='name,sequence\nkept,AAA\n'
    assert not completed_run(tmp_path)


def test_success_exit_without_artifacts_is_failed(tmp_path,monkeypatch):
    import run_watchdog
    (tmp_path/'run.json').write_text(json.dumps(dict(schema=1,status='running')))
    monkeypatch.setattr(run_watchdog,'run_owned',lambda *a,**k:Outcome(0,'exited',1.))
    with pytest.raises(SystemExit):supervise(SimpleNamespace(output_folder=tmp_path,timeout_minutes=1.))
    assert json.loads((tmp_path/'run.json').read_text())['status']=='failed'


def test_installer_refuses_existing_environment_before_launch(tmp_path,monkeypatch):
    from scripts import install_runtime
    marker=tmp_path/'precious';marker.write_text('preserve')
    monkeypatch.setattr(install_runtime.subprocess,'check_output',lambda *a,**k:pytest.fail('process started'))
    with pytest.raises(SystemExit):install_runtime.main(['--env',str(tmp_path),'--data-dir',str(tmp_path)])
    assert marker.read_text()=='preserve'


def test_installer_help_from_foreign_directory(tmp_path):
    p=subprocess.run(['bash',str(ROOT/'install_foldcraft.sh'),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0 and '--data-dir' in p.stdout and '--env' in p.stdout


def test_scheduler_passes_timeout_to_driver(tmp_path,monkeypatch):
    from baseline import scheduler as s
    monkeypatch.setattr(s,'chunk_signature',lambda c:{})
    commands=[]
    monkeypatch.setattr(s.subprocess,'Popen',lambda command,**kw:commands.append(command))
    c=s.Chunk('fold',0,1,1.,dict(template='binder',target='target',target_hotspots='1',binder_hotspots='1'))
    _,log=s.launch(c,str(tmp_path),0,str(ROOT),sys.executable,12.5);log.close()
    command=commands[0];assert command[command.index('--timeout_minutes')+1]=='12.5'


def test_worker_failure_overrides_premature_complete_marker(tmp_path,monkeypatch):
    import run_watchdog
    (tmp_path/'run.json').write_text(json.dumps(dict(schema=1,status='complete')))
    monkeypatch.setattr(run_watchdog,'run_owned',lambda *a,**k:Outcome(7,'exited',1.))
    with pytest.raises(SystemExit) as e:supervise(SimpleNamespace(output_folder=tmp_path,timeout_minutes=1.))
    assert e.value.code==7
    assert json.loads((tmp_path/'run.json').read_text())['status']=='failed'
