"""Check tamper rejection and frozen results using the real archived fixture."""
import argparse
import copy
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reproduce
from common import read,write,file_sha

class ArchivedRunTests(unittest.TestCase):
    def test_all_four_results_are_recomputable(self):
        rows=reproduce.verify()
        self.assertEqual([(r['job'],r['n'],r['accuracy']) for r in rows],
                         [('qwen_arc',500,88.4),('qwen_obqa',500,90.4),('llama_arc',500,82.8),('llama_obqa',500,89.6)])

    def changed_bundle(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name)/'bundle'
        shutil.copytree(reproduce.BUNDLE,path)
        return path

    def tamper_result(self,bundle,mutate,update_marker):
        p=bundle/'table_runs/qwen_arc/result.json';r=read(p);mutate(r);write(p,r)
        if update_marker:
            marker=p.with_suffix('.complete.json');m=read(marker);m['result_sha256']=file_sha(p);write(marker,m)

    def test_changed_file_is_rejected_even_with_same_saved_metrics(self):
        b=self.changed_bundle()
        self.tamper_result(b,lambda r:r.update(status='changed'),False)
        with self.assertRaises(ValueError):reproduce.verify(b)

    def test_wrong_gold_is_rejected_even_with_a_new_completion_hash(self):
        b=self.changed_bundle()
        self.tamper_result(b,lambda r:r['methods']['sedpo']['records'][0].update(gold=3),True)
        with self.assertRaises(ValueError):reproduce.verify(b)

    def test_wrong_answer_token_is_rejected(self):
        b=self.changed_bundle()
        self.tamper_result(b,lambda r:r['methods']['sedpo']['records'][0].update(raw_output='D'),True)
        with self.assertRaises(ValueError):reproduce.verify(b)

    def test_reordered_demos_are_rejected(self):
        b=self.changed_bundle()
        def change(r):r['methods']['sedpo']['records'][0]['demo_ids'].reverse()
        self.tamper_result(b,change,True)
        with self.assertRaises(ValueError):reproduce.verify(b)

    def test_model_identity_is_not_inferred_from_directory_name(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):reproduce.model_check(Path(d),'qwen')
            self.assertFalse(reproduce.model_check(Path(d),'qwen',True)['original_model_files_match'])

    def test_dry_plan_does_not_create_output_or_load_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)/'not_created'
            args=argparse.Namespace(command='plan',work_dir=work,mode='rescore',qwen_model=Path(d)/'qwen',llama_model=Path(d)/'llama')
            for tag in reproduce.JOBS:
                plan=reproduce.commands(args,tag)
                self.assertEqual([stage for stage,_ in plan[:6]],['find1','score1','find2','score2','find3','score3'])
                self.assertEqual(plan[-1][0],'evaluate')
            self.assertFalse(work.exists())

if __name__=='__main__':unittest.main()
