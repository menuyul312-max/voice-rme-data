"""Run dependency-light CPU regression tests without pytest."""
import importlib
import traceback
import sys

if __name__ == '__main__':
    passed=failed=0
    for name in ['test_benchmark_real','test_live_pipeline','test_rme_asr_v2','test_release_safety']:
        mod=importlib.import_module(name)
        for key in sorted(vars(mod)):
            if key.startswith('test_') and callable(getattr(mod,key)):
                try:
                    getattr(mod,key)(); passed+=1
                except Exception:
                    failed+=1; print(f'FAIL {name}.{key}'); traceback.print_exc()
    print(f'{passed} passed, {failed} failed (CPU; no model inference)')
    sys.exit(bool(failed))
