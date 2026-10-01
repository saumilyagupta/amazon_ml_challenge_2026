"""Reproduce a frozen candidate from the exact #8 baseline; stdlib only.

This reproduces the decision changes, not the upstream trained model. It refuses
a different baseline so these pair masks cannot silently be applied to a new base.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def sha(path):
    with open(path,'rb') as fh:return hashlib.file_digest(fh,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--release',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    manifest=json.loads((a.release/'manifest.json').read_text())
    assert sha(a.baseline)==manifest['baseline_sha256'],'Baseline changed: re-evaluate the rule on that model.'
    assert a.output.resolve()!=a.baseline.resolve(),'Cannot overwrite baseline.'
    assert not a.output.exists(),'Output exists; choose a new path.'
    rem={}
    with open(a.release/'removed_pairs.tsv') as fh:
        for r in csv.DictReader(fh,delimiter='\t'):rem.setdefault(r['s1_id'],set()).add(r['cand_id'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix(a.output.suffix+'.tmp')
    assert not tmp.exists(),'Temporary output exists.'
    touched=set();removed=0
    try:
        with open(a.baseline) as fi,open(tmp,'x') as fo:
            fo.write(next(fi))
            for line in fi:
                q,values=line.rstrip('\n').split('\t')
                if q not in rem:fo.write(line);continue
                old=set(values.split(',')) if values else set()
                assert rem[q]<=old,(q,'Removal not selected in baseline')
                fo.write(q+'\t'+','.join(sorted(old-rem[q]))+'\n')
                removed+=len(rem[q]);touched.add(q)
        assert touched==set(rem) and removed==manifest['removed_pairs']
        assert sha(tmp)==manifest['sha256'],'Reproduction checksum mismatch'
        tmp.rename(a.output)
        print(json.dumps({'output':str(a.output),'sha256':manifest['sha256'],'removed':removed,'reproduction':'PASS'},indent=2))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

if __name__=='__main__':main()
