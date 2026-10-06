#!/usr/bin/env python3
"""Resume individual budget jobs under measured memory reservations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'guard15_lammu'))
from run_unified_campaign import available,rss_group,GIB
BUDGETS=(120000,)
ARMS=('guard15_equal','uniform')
SYSTEMS=('n2_r080','n2_r090','n2_r100','n2_r110','n2_r125','n2_r145','n2_r160','n2_r180','n2_r200','n2_r220','n2_r250')

def main():
    p=argparse.ArgumentParser();p.add_argument('--max-workers',type=int,default=4)
    p.add_argument('--threads',type=int,default=3);a=p.parse_args()
    lock=(HERE/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=HERE/'campaign_state.json'
    old=json.loads(state_path.read_text()) if state_path.exists() else {}
    policy_path=HERE/'runtime_policy.json'
    policy=dict(max_workers=a.max_workers,threads=a.threads,reserve_gib=5.,total_rss_limit_gib=36.,
                worker_rss_limit_gib=13.,worker_address_space_gib=20.,
                guard_reservation_gib=8.,uniform_reservation_gib=10.)
    if not policy_path.exists():policy_path.write_text(json.dumps(policy,indent=2)+'\n')
    state=dict(jobs={},resources=dict(peak_total_rss_gib=0.,min_available_gib=available()/GIB,guard_events=[]),
               previous_run=old,status='running',supervisor_pid=os.getpid(),started=time.time())
    def persist():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    jobs=[dict(name=f'{system}_{arm}_r{s}_b{b}',system=system,arm=arm,stream=s,budget=b)
          for system in SYSTEMS for s in range(8) for b in BUDGETS for arm in ARMS]
    def complete(j):
        return all((HERE/'results'/j['system']/j['arm']/f'b{j["budget"]}_r{j["stream"]}.{ext}').exists()
                   for ext in ('json','npz'))
    pending=[j for j in jobs if not complete(j)]
    for j in jobs:
        if complete(j):state['jobs'][j['name']]=dict(status='complete',reused=True)
    running={};failure=None;last_status=0;automatic_cap=100;extra_reservation={}
    def stop(proc):
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    try:
        while pending or running:
            if (HERE/'STOP').exists():raise RuntimeError('Stopped by local STOP file')
            policy=json.loads(policy_path.read_text())
            assert 1<=policy['max_workers']<=8 and policy['reserve_gib']>=5
            state['policy']=policy;state['effective_max_workers']=min(policy['max_workers'],automatic_cap)
            now=time.time();free=available()/GIB
            for name,item in list(running.items()):
                proc=item['proc'];code=proc.poll();record=state['jobs'][name]
                if code is not None:
                    item['log'].close()
                    ok=code==0 and complete(item['job']) and not record.get('reason')
                    record.update(status='complete' if ok else 'failed',exit_code=code,seconds=now-record['started'])
                    print('END',name,record['status'],'peak',round(record.get('peak_rss_gib',0),3),flush=True)
                    if not ok:
                        if record.get('memory_retry') and record['attempt']<3:
                            pending.insert(0,item['job'])
                        else:failure=failure or name
                    del running[name];continue
                rss=rss_group(proc.pid)/GIB;item['rss']=rss
                record.update(rss_gib=rss,peak_rss_gib=max(record.get('peak_rss_gib',0),rss),checked=now)
            total=sum(i['rss'] for i in running.values());res=state['resources']
            res['peak_total_rss_gib']=max(res['peak_total_rss_gib'],total)
            res['min_available_gib']=min(res['min_available_gib'],free)
            res['max_active_workers']=max(res.get('max_active_workers',0),len(running))
            bad=[n for n,i in running.items() if i['rss']>policy['worker_rss_limit_gib']]
            if running and (bad or total>policy['total_rss_limit_gib'] or free<policy['reserve_gib']):
                name=bad[0] if bad else max(running,key=lambda n:state['jobs'][n]['started'])
                reason=f'Memory guard total={total:.2f}, available={free:.2f} GiB'
                state['jobs'][name].update(reason=reason,memory_retry=True)
                res['guard_events'].append(dict(name=name,reason=reason,time=now))
                automatic_cap=max(1,len(running)-1)
                arm=running[name]['job']['arm'];extra_reservation[arm]=extra_reservation.get(arm,0)+2.
                stop(running[name]['proc'])
                persist();continue
            def reservation(job):
                key='uniform_reservation_gib' if job['arm']=='uniform' else 'guard_reservation_gib'
                return policy[key]+extra_reservation.get(job['arm'],0)
            while not failure and pending and len(running)<min(policy['max_workers'],automatic_cap):
                reserved=sum(i['reservation'] for i in running.values())
                unfilled=sum(max(0,i['reservation']-i['rss']) for i in running.values())
                free=available()/GIB
                eligible=next((k for k,j in enumerate(pending)
                               if reserved+reservation(j)<=policy['total_rss_limit_gib']
                               and unfilled+reservation(j)+policy['reserve_gib']<=free),None)
                if eligible is None:break
                job=pending.pop(eligible);name=job['name'];prior=state['jobs'].get(name,{})
                if prior:state.setdefault('prior_attempts',[]).append(dict(name=name,**prior))
                log=(HERE/'logs'/f'{name}.log').open('a')
                cmd=[sys.executable,'-u',str(HERE/'scan.py'),'worker','--system',job['system'],'--arm',job['arm'],
                     '--stream',str(job['stream']),'--budget',str(job['budget']),'--threads',str(policy['threads'])]
                proc=subprocess.Popen(cmd,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[name]=dict(proc=proc,log=log,job=job,rss=0.,reservation=reservation(job))
                state['jobs'][name]=dict(status='running',pid=proc.pid,started=time.time(),args=cmd,
                                         reservation_gib=reservation(job),attempt=prior.get('attempt',0)+1)
                print('START',name,'reservation',reservation(job),flush=True)
            state['pending']=len(pending);persist()
            if now-last_status>30:
                print('STATUS complete',sum(complete(j) for j in jobs),'of',len(jobs),'active',len(running),
                      'RSS',round(total,2),'available',round(free,2),flush=True)
                last_status=now
            if failure and not running:raise RuntimeError(failure)
            time.sleep(.5)
        subprocess.run([sys.executable,str(HERE/'scan.py'),'summarize'],cwd=HERE,check=True)
        state['status']='complete';persist();print('CAMPAIGN_COMPLETE',flush=True)
    finally:
        for name,item in running.items():
            if item['proc'].poll() is None:stop(item['proc'])
            item['log'].close();state['jobs'][name].update(status='interrupted',exit_code=item['proc'].returncode)
        if state['status']!='complete':state['status']='stopped'
        persist()

if __name__=='__main__':main()
