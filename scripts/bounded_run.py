"""Own-process resource guard and append-only receipts; never touches other jobs."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def snapshot():
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        mem[key] = int(value.strip().split()[0])
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {"monotonic_seconds":time.monotonic(),"available_kib":mem['MemAvailable'],"swap_used_kib":mem['SwapTotal']-mem['SwapFree'],"pswpin":int(vm['pswpin']),"pswpout":int(vm['pswpout'])}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',required=True)
    p.add_argument('--sample',type=int)
    p.add_argument('--timeout',type=int,default=90)
    p.add_argument('--preflight-mib',type=int,default=3072)
    p.add_argument('--stop-mib',type=int,default=2048)
    p.add_argument('--memory-mib',type=int,default=256)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.memory_mib not in (128,256):raise SystemExit('memory budget must be 128 or 256 MiB')
    if args.preflight_mib<2048 or args.stop_mib<1792:raise SystemExit('guard cannot go below parent-authorized thresholds')
    prefix=Path('reports/history')/args.label
    receipt=prefix.with_suffix('.json')
    if receipt.exists():
        raise SystemExit('Refusing to overwrite resource receipt')
    samples=[snapshot()]
    if args.sample:
        for _ in range(args.sample):
            time.sleep(1)
            samples.append(snapshot())
        result={"kind":"headroom_sample","samples":samples,"passed":min(s['available_kib'] for s in samples)>=args.preflight_mib*1024 and samples[-1]['pswpout']==samples[0]['pswpout'] and samples[-1]['swap_used_kib']<=samples[0]['swap_used_kib']}
    else:
        command=args.command[1:] if args.command[:1]==['--'] else args.command
        if samples[0]['available_kib']<args.preflight_mib*1024:
            result={"kind":"bounded_run","command":command,"samples":samples,"passed":False,"blocker":"initial headroom below 2.2 GiB"}
        else:
            def limits():
                resource.setrlimit(resource.RLIMIT_AS,(args.memory_mib*1024*1024,args.memory_mib*1024*1024))
                resource.setrlimit(resource.RLIMIT_CPU,(args.timeout,args.timeout))
                if hasattr(os,'sched_getaffinity'):
                    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            stopped=None
            started=time.monotonic()
            with prefix.with_suffix('.log').open('x') as out:
                child=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env=env,preexec_fn=limits,start_new_session=True)
                peak_rss=0
                while child.poll() is None:
                    time.sleep(.5)
                    samples.append(snapshot())
                    try:
                        status=Path(f'/proc/{child.pid}/status').read_text()
                        rss=int(next(line for line in status.splitlines() if line.startswith('VmRSS:')).split()[1])
                        peak_rss=max(peak_rss,rss)
                    except (OSError,StopIteration):
                        pass
                    if samples[-1]['available_kib']<args.stop_mib*1024:
                        stopped='host available memory below authorized stop threshold'
                    elif samples[-1]['pswpout']>samples[0]['pswpout'] or samples[-1]['swap_used_kib']>samples[0]['swap_used_kib']:
                        stopped='host swap pressure grew'
                    elif time.monotonic()-started>args.timeout:
                        stopped='wall timeout'
                    elif peak_rss>args.memory_mib*1024:
                        stopped='RSS exceeded authorized limit'
                    if stopped:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL)
                        break
                code=child.wait()
            usage=resource.getrusage(resource.RUSAGE_CHILDREN)
            result={"kind":"bounded_run","command":command,"passed":code==0 and stopped is None,"returncode":code,"stopped_reason":stopped,"address_space_limit_mib":args.memory_mib,"cpu_affinity_count":1,"wall_seconds":time.monotonic()-started,"peak_rss_kib":max(peak_rss,int(usage.ru_maxrss)),"samples":samples}
    result['preflight_mib']=args.preflight_mib
    result['stop_mib']=args.stop_mib
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},sort_keys=True))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':
    main()
