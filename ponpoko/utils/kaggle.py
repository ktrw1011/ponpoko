from typing import Union
from pathlib import Path

TEMPLATE = '''{
  "title": "INSERT_TITLE_HERE",
  "id": "kcotton21/INSERT_SLUG_HERE",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}'''

def make_template(title, dataset_path: Union[Path, str]):
    template = TEMPLATE
    template = template.replace("INSERT_TITLE_HERE", title)
    template = template.replace("INSERT_SLUG_HERE", title)

    with open(Path(dataset_path) / Path("dataset-metadata.json"), "w") as f:
        f.write(template)
