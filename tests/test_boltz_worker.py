"""Exercise real spawned workers using a tiny stand-in CLI (no GPU imports)."""
import json
from pathlib import Path
import sys
import pytest
from baseline.boltz_worker import BoltzWorkerPool


def test_worker_reuses_process_and_recovers_after_cli_failure(tmp_path,monkeypatch):
    package=tmp_path/'boltz';package.mkdir();(package/'__init__.py').write_text('')
    (package/'main.py').write_text('''
import json, os, logging, multiprocessing
logger=logging.getLogger("fake_boltz")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
from pathlib import Path
calls=0
class CLI:
    def main(self,args,standalone_mode):
        global calls
        calls+=1
        if args[-1]=='fail':raise ValueError('intentional request failure')
        Path(args[-1]).write_text(json.dumps(dict(pid=os.getpid(),calls=calls,seed=args[-2],context=multiprocessing.get_start_method())))
        logger.info('retained logger')
        print('request completed')
cli=CLI()
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    with BoltzWorkerPool(1) as pool:
        first=tmp_path/'one.json';second=tmp_path/'two.json'
        command=[sys.executable,'-m','boltz.main','predict','17',str(first)]
        assert pool.run(command,tmp_path/'one.log').returncode==0
        command[-1]='fail'
        failed=pool.run(command,tmp_path/'failed.log')
        assert failed.returncode==1 and 'intentional request failure' in failed.stderr
        command[-2:]=['23',str(second)]
        assert pool.run(command,tmp_path/'two.log').returncode==0
    a=json.loads(first.read_text());b=json.loads(second.read_text())
    assert a['pid']==b['pid'] and a['calls']==1 and b['calls']==3
    assert (a['seed'],b['seed'])==('17','23')
    assert (tmp_path/'one.log').read_text()=='retained logger\nrequest completed\n'
    assert (tmp_path/'two.log').read_text()=='retained logger\nrequest completed\n'
    import multiprocessing
    assert a['context']==b['context']==multiprocessing.get_all_start_methods()[0]


def test_worker_rejects_non_boltz_commands(tmp_path):
    with BoltzWorkerPool() as pool:
        with pytest.raises(ValueError,match='only boltz'):
            pool.run([sys.executable,'arbitrary.py'],tmp_path/'bad.log')
    with pytest.raises(ValueError,match='positive'):BoltzWorkerPool(0)


def test_scorer_routes_to_worker_with_same_seed_and_fresh_artifact_dir(tmp_path):
    from subprocess import CompletedProcess
    from baseline.score_boltz2 import run_boltz2
    class Worker:
        def __init__(self):self.paths=[]
        def run(self,command,log_path):
            assert command[command.index('--seed')+1]=='42'
            assert '--no_kernels' in command and '--flash_attn' not in command
            path=Path(log_path).parent;self.paths.append(path)
            (path/'new.cif').write_text('structure')
            (path/'confidence_new.json').write_text(json.dumps(dict(iptm=.7)))
            Path(log_path).write_text('worker log')
            return CompletedProcess(command,0,stdout='',stderr='')
    worker=Worker()
    for _ in range(2):
        assert run_boltz2('AAA','AAA',tmp_path,seed=42,use_msa=False,flash_attn=False,
                          no_kernels=True,worker_pool=worker)[0]==.7
    assert worker.paths[0]!=worker.paths[1]
    for path in worker.paths:
        receipt=json.loads((path/'selection.json').read_text())
        assert receipt['seed']==42 and receipt['execution_backend']=='persistent'


@pytest.mark.parametrize('flags',[[],['--no-msa'],['--no-msa','--no-kernels','--flash-attn']])
def test_cli_rejects_unvalidated_persistent_configurations(monkeypatch,flags):
    from baseline import score_boltz2
    monkeypatch.setattr(sys,'argv',['score_boltz2','unused','--execution-backend','persistent',*flags])
    with pytest.raises(SystemExit) as exc:score_boltz2.main()
    assert exc.value.code==2
