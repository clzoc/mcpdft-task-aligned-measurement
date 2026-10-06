#!/usr/bin/env python3
"""Wait for the full primary campaign, import equilibrium, then launch scan."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
PRIMARY=HERE.parent/'guard15_equal_real30'

def main():
    lock=(HERE/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=HERE/'queue_state.json'
    state=dict(status='waiting_for_primary',pid=os.getpid(),started=time.time(),dependency=str(PRIMARY))
    def save():
        temp=target.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(target)
    last_report=0
    try:
        while True:
            if (HERE/'STOP').exists():
                state.update(status='stopped',reason='STOP file');save();return
            current=json.loads((PRIMARY/'campaign_state.json').read_text())
            count=len(list((PRIMARY/'results').rglob('*.json')))
            state.update(checked=time.time(),primary_status=current['status'],primary_results=count)
            save()
            if current['status']=='complete':
                assert count==320,'Primary marked complete but not all expected results exist'
                for system in ('n2','co_eq'):
                    for arm in ('guard15_equal','guard15_equal_mu0','exclusive_guard15_mu2','exclusive_guard15_mu0','uniform'):
                        for stream in range(8):
                            for budget in (30000,60000,120000,240000):
                                path=PRIMARY/'results'/system/arm/f'b{budget}_r{stream}.json'
                                assert json.loads(path.read_text())['status']=='optimal'
                                assert path.with_suffix('.npz').exists()
                break
            if time.time()-last_report>=60:
                print('WAIT_PRIMARY',current['status'],count,'/320',flush=True);last_report=time.time()
            time.sleep(10)
        state.update(status='importing_equilibrium',primary_completed=time.time());save()
        subprocess.run([sys.executable,'-u',str(HERE/'scan.py'),'import-equilibrium'],cwd=HERE,check=True)
        state.update(status='scan_running',scan_started=time.time());save()
        print('PRIMARY_COMPLETE; START_N2_SCAN',flush=True)
        subprocess.run([sys.executable,'-u',str(HERE/'run_campaign.py'),'--max-workers','4','--threads','3'],cwd=HERE,check=True)
        validation=json.loads((HERE/'validation.json').read_text())
        assert validation['complete'] and validation['results']==176
        state.update(status='complete',completed=time.time());save()
        print('N2_SCAN_COMPLETE',flush=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error),checked=time.time());save();raise

if __name__=='__main__':main()
