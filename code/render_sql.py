"""Render actual SQL path placeholders without running a database query."""
import argparse
from pathlib import Path
def render(template,raw_database_dir,selected_keys):
    replacements={'{RAW_DATABASE_DIR}':str(raw_database_dir),'{SELECTED_KEYS}':str(selected_keys)}
    text=template
    for key,value in replacements.items(): text=text.replace(key,value.replace("'","''"))
    if '{RAW_DATABASE_DIR}' in text or '{SELECTED_KEYS}' in text: raise ValueError('Unresolved path placeholder')
    return text
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('template');p.add_argument('--raw-database-dir',required=True)
    p.add_argument('--selected-keys',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output).resolve();repository=Path(__file__).resolve().parents[1]
    if output==repository or repository in output.parents: raise ValueError('Write generated SQL outside the public repository')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(render(Path(a.template).read_text(),a.raw_database_dir,a.selected_keys))
