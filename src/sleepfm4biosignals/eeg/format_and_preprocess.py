# © Copyright 2026 BrainCapture

# from eegprep.preprocessing.finetune import FinetunePreprocessPipeline
# from eegprep.preprocessing.description_maps import description_map
# from eegprep.format_datasets import format_dataset

# Reach out to William Lehn-Schiøler, wls@braincapture.com for collaboration and licensing.

import os
from datetime import datetime
    
def preprocess_dataset(in_dir, out_dir, log_path):

    finetune_pipeline = FinetunePreprocessPipeline(
        descriptions=list(description_map.keys()),
        description_map=description_map,
        sampling_freq=128,
        tmin = 0.0,
        tlen = 60.0
    )

    finetune_pipeline.configure_logging(log_path)

    src_paths = finetune_pipeline.extract_src_paths(in_dir)

    output_files = finetune_pipeline.run_and_save(
        src_paths, 
        out_dir, 
        batch_size=32
    )

    print(f"Number of output files: {len(output_files)}")


if __name__ == "__main__":

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f'logs/preprocessing_dataset_name_{now}.log'
    os.makedirs('logs', exist_ok=True)
    
    base_dir = ""

    source_dir = base_dir + ""
    os.makedirs(source_dir, exist_ok=True)

    format_dir = base_dir + ""
    os.makedirs(format_dir, exist_ok=True)

    preprocess_dir = base_dir + ""
    os.makedirs(preprocess_dir, exist_ok=True)
    
    format_dataset(
        source_dir=source_dir,
        output_dir=format_dir,
     )

    preprocess_dataset(
        in_dir=format_dir,
        out_dir=preprocess_dir,
        log_path=log_path
    )