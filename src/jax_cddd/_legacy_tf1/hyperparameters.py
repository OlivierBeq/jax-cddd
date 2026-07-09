"""Helper functions to parse and create a hyperparameter object."""

import os


DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def create_hparams():
    """Create training hparams."""
    return dict(model="NoisyGRUSeq2SeqWithFeatures",
                input_pipeline="InputPipelineWithFeatures",
                input_sequence_key="random_smiles",
                output_sequence_key="canonical_smiles",
                cell_size=[512, 1024, 2048],
                emb_size=512,
                save_dir=os.path.join(DEFAULT_DATA_DIR, 'default_model'),
                device="-1",
                lr=0.0002,
                gpu_mem_frac=0.45,
                num_steps=100000000,
                summary_freq=1000,
                inference_freq=5000,
                batch_size=64,
                beam_width=10,
                one_hot_embedding = False,
                char_embedding_size=32,
                train_file="../data/pretrain_dataset.tfrecords",
                val_file="../data/pretrain_dataset_val.tfrecords",
                infer_file="../data/val_dataset_preprocessed3.csv",
                allow_soft_placement=True,
                cpu_threads=5,
                overwrite_saves=False,
                input_dropout=0.15,
                emb_noise=0.05,
                conv_hidden_size=[128],
                kernel_size=[2],
                reverse_decoding=False,
                buffer_size=10000,
                lr_decay=True,
                lr_decay_frequency=50000,
                lr_decay_factor=0.9,
                num_buckets=8,
                min_bucket_length=20.0,
                max_bucket_length=60.0,
                num_features=7,
                rand_input_swap=True,
                infer_input="canonical",
                emb_activation="tanh",
                div_loss_scale=1.0,
                div_loss_rate=0.9,
                encode_vocabulary_file = os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"),
                decode_vocabulary_file = os.path.join(DEFAULT_DATA_DIR, "indices_char.npy"),
                hparams_file_name = os.path.join(DEFAULT_DATA_DIR, 'default_model', 'hparams.json')
            )