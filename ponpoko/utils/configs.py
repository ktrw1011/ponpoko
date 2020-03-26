from pathlib import Path
import dataclasses
import yaml
import inspect

@dataclasses.dataclass
class YamlConfig:
    """https://qiita.com/kzmssk/items/483f25f47e0ed10aa948"""

    def save(self, config_path: Path, file_name: str="config.yaml"):
        """ Export config as YAML file """
        assert config_path.parent.exists(), f'directory {config_path.parent} does not exist'

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, Path):
                    data[key] = str(val)
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path / file_name, 'w', encoding='utf-8') as f:
            yaml.dump(convert_dict(dataclasses.asdict(self)), f, allow_unicode=True, default_flow_style=False)

    @classmethod
    def load(cls, config_path: Path):
        """ Load config from YAML file """
        assert config_path.exists(), f'YAML config {config_path} does not exist'

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == Path:
                    data[key] = Path(val)
                if inspect.isclass(child_class) and issubclass(child_class, YamlConfig):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to YamlConfig
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)