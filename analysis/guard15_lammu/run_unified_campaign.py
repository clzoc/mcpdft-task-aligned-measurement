"""Serial, resumable supervisor with a 12 GiB RSS kill and 4 GiB reserve."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE/'unified'
GIB = 1024**3


def available():
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    memory = int(info['MemAvailable'].split()[0])*1024
    # Respect any finite cgroup limits, including ancestor cgroups.
    root = Path('/sys/fs/cgroup')
    membership = Path('/proc/self/cgroup').read_text().strip().splitlines()
    paths = {root}
    for line in membership:
        if line.startswith('0::'):
            current = root/line.split('::', 1)[1].lstrip('/')
            paths.update([current, *[p for p in current.parents if p == root or root in p.parents]])
    for path in paths:
        try:
            limit = (path/'memory.max').read_text().strip()
            used = int((path/'memory.current').read_text())
            if limit != 'max':
                memory = min(memory, int(limit)-used)
        except FileNotFoundError:
            pass
    return memory


def rss_group(pid):
    total = 0
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) != pid:
                continue
            for line in (path/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1])*1024
        except (OSError, ValueError):
            pass
    return total


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [('init', None)] + [('grid', s) for s in range(3)]
    jobs += [('freeze-weights', None)] + [('conditional', s) for s in range(3)]
    jobs += [('freeze-rule', None)] + [('fresh', s) for s in range(10000, 10006)]
    state_path = OUT/'campaign_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs':{}}
    def persist():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(state_path)
    for command, stream in jobs:
        name = command if stream is None else f'{command}_r{stream}'
        if state['jobs'].get(name, {}).get('status') == 'complete':
            continue
        while available() < 16*GIB:
            state['admission'] = dict(job=name, status='waiting_for_memory',
                                      available_gib=available()/GIB,
                                      required_gib=16, checked=time.time())
            persist()
            print('WAIT_MEMORY', name, state['admission']['available_gib'], flush=True)
            time.sleep(15)
        state.pop('admission', None)
        args = [sys.executable, '-u', str(HERE/'unified_design.py'), command]
        if stream is not None:
            args += ['--stream', str(stream)]
        started = time.time()
        with (OUT/f'{name}.log').open('a') as log:
            proc = subprocess.Popen(args, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            previous = state['jobs'].get(name)
            if previous:
                state.setdefault('previous_attempts', []).append(dict(job=name, **previous))
            state['jobs'][name] = dict(status='running', pid=proc.pid, started=started)
            persist()
            print('START', name, flush=True)
            peak = 0
            reason = None
            while proc.poll() is None:
                current = rss_group(proc.pid)
                peak = max(peak, current)
                free = available()
                state['jobs'][name].update(current_rss_gib=current/GIB,
                                          available_gib=free/GIB,
                                          peak_rss_gib=peak/GIB, checked=time.time())
                persist()
                if current > 12*GIB or free < 4*GIB:
                    reason = f'memory guard: RSS={current/GIB:.2f}, available={free/GIB:.2f} GiB'
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(2)
            code = proc.wait()
        state['jobs'][name].update(status='complete' if code == 0 and reason is None else 'failed',
                                  exit_code=code, reason=reason,
                                  seconds=time.time()-started, peak_rss_gib=peak/GIB)
        persist()
        print('END', name, state['jobs'][name], flush=True)
        if code != 0 or reason:
            raise RuntimeError(f'{name} failed; inspect {OUT/name}.log')
    state['status'] = 'complete'
    persist()


if __name__ == '__main__':
    main()
