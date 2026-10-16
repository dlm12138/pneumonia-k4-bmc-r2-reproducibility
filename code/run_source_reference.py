"""Explicitly guarded source-inspection runner, not a new scientific implementation."""
from pathlib import Path
import argparse, os, runpy, sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',help='Path under code/source_history')
    p.add_argument('--execute-source-reference',action='store_true')
    p.add_argument('--allow-historical-fit',action='store_true')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    source=(root/args.source).resolve()
    if not source.is_relative_to(root/'source_history') or not source.is_file():
        p.error('Only an existing source_history Python program is allowed')
    if not args.execute_source_reference:
        p.error('Inspection only by default; explicit execution flag required')
    if 'historical_model' in source.parts and not args.allow_historical_fit:
        p.error('Historical fitting is disabled; it cannot reproduce the original frozen numerical model')
    out=Path(os.environ['OUTPUT_ROOT']).resolve()
    if out.is_relative_to(root.parent) or root.parent.is_relative_to(out):
        p.error('OUTPUT_ROOT must be a dedicated private directory outside the repository')
    if not Path(os.environ['DATA_ROOT']).is_dir():p.error('DATA_ROOT is not a valid private input directory')
    out.mkdir(parents=True,exist_ok=True)
    sys.argv=[str(source)]+args.arguments
    namespace=runpy.run_path(str(source),run_name='__source_reference_run__')
    # The original argparse/entry block executes unchanged under the guarded
    # run name. Selected-function modules have no dispatcher to invent.
    if not any(k in namespace for k in ['main','collect','run']):
        print('Selected-function module loaded; call the documented functions with private configured inputs. No inferred pipeline is run.')

if __name__=='__main__':main()
