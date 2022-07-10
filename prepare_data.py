from pathlib import Path
from arg_parse import get_data_args

from src.core.datatypes import DatasetType
from src.readers import CONLLReader

dataset2reader = {
    DatasetType.CONLL2003.value : CONLLReader
}


if __name__ == "__main__":

    args = get_data_args()

    dataset = args.dataset_type
    filepath = Path(args.path_to_file)
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath.as_posix()} not exists")
    output_dir = args.output_folder
    if output_dir is None:
        output_dir = filepath.parent
        print(f"--output_dir not specified. Going to save at {filepath.parent.as_posix()}")

    if dataset not in dataset2reader:
        raise ValueError(f"Expected dataset to be on of {dataset2reader.keys()}")

    reader = dataset2reader[dataset]()
    data = reader.read_from_file(
        path_to_file=filepath
    )

    filepath_save = output_dir / (filepath.stem + ".json")
    reader.save_to_json(
        data=data,
        path=filepath_save
    )
