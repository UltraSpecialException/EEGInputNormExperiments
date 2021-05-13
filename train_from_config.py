from dn3.configuratron import ExperimentConfig, DatasetConfig
from dn3.trainable.processes import StandardClassification
from dn3.data.dataset import Dataset
from model import AdaptiveInputNormEEGNet, AdaptiveInputNormTIDNet, SurfaceLaplacianEEGNet
from dn3.trainable.models import EEGNet, TIDNet
from first_diff import FirstDifference, BrownianMotion, StepEWMZScore
from normalizations import FixedScale, ZScore
from processes import MultipleParamGroupClassification
import mne
import torch
from argparse import ArgumentParser, Namespace
import os


name_to_tf = {
    "zscore": ZScore,
    "firstdifference": FirstDifference,
    "brownian": BrownianMotion,
    "stepewmzscore": StepEWMZScore,
    "fixedscale": FixedScale
}


def make_model_and_process(args, dataset: Dataset, ds_config: DatasetConfig) -> StandardClassification:
    """
    Return the classifier process from the arguments given.
    """
    cuda_setting = not args.no_cuda and torch.cuda.is_available()
    if args.input_norm_method.lower() == "dain":
        kwargs = {
            "feat_dim": len(dataset.channels),
            "start_gate_iter": args.start_gate_iter
        }
        if args.model_type == "tid":
            model = AdaptiveInputNormTIDNet.from_dataset(dataset, **kwargs)
        else:
            print("Creating Adaptive Input Norm EEG Model")
            model = AdaptiveInputNormEEGNet.from_dataset(dataset, **kwargs)

        process = StandardClassification(model, cuda=cuda_setting, learning_rate=ds_config.lr)
    elif args.input_norm_method.lower() == "dain_author":
        model = AdaptiveInputNormEEGNet.from_dataset(dataset)
        process = MultipleParamGroupClassification(model, cuda=cuda_setting, learning_rate=ds_config.lr)
    elif args.surface_laplacian:
        model = SurfaceLaplacianEEGNet.from_dataset(dataset)
        process = StandardClassification(model, cuda=cuda_setting, learning_rate=ds_config.lr)
    else:
        model = args.model.from_dataset(dataset)
        process = StandardClassification(model, cuda=cuda_setting, learning_rate=ds_config.lr)

    return process


def main(args: Namespace) -> None:
    """
    Run the training using the arguments given.
    """
    mne.set_log_level(False)
    config_filename = args.config_filename
    experiment = ExperimentConfig(config_filename)
    ds_config = experiment.datasets[args.dataset_name]

    dataset = ds_config.auto_construct_dataset()

    if args.input_norm_method.lower() in name_to_tf.keys():
        transform = name_to_tf[args.input_norm_method.lower()]
        if args.input_norm_method.lower() == "stepewmzscore":
            transform = transform(args.window, dataset.sequence_length - 1)
        else:
            transform = transform()
        dataset.add_transform(transform)

    results = []
    for training, validation, test in dataset.lmso():
        process = make_model_and_process(args, dataset, ds_config)

        train_metrics, val_metrics = process.fit(
            training_dataset=training, validation_dataset=validation, **ds_config.train_params)

        accuracy = process.evaluate(test)["Accuracy"]
        results.append(accuracy)
        if not args.run_all:
            with open(f"{os.path.join(args.save_dir, args.save_name)}_train.csv", "w+") as f:
                train_metrics.to_csv(f)

            with open(f"{os.path.join(args.save_dir, args.save_name)}_val.csv", "w+") as f:
                val_metrics.to_csv(f)

            break

    print(f"Average accuracy: {sum(results) / len(results):.2f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_filename", required=True, help="Path to config file to load", type=str)
    parser.add_argument("--input_norm_method", help="Type of input normalization",
                        choices=["zscore", "firstdifference", "dain_author", "dain", "brownian", "stepewmzscore",
                                 "fixedscale", "none"], type=str, default="dain")
    parser.add_argument("--no_cuda", help="Disable CUDA training", action="store_true")
    parser.add_argument("--start_gate_iter", help="The iteration number to start gating in DAIN normalization",
                        type=int, default=3000)
    parser.add_argument("--run_all", help="Run all partitions of K-Folds (10).", action="store_true")
    parser.add_argument("--save_dir", help="Path to directory to save output files", default=".", type=str)
    parser.add_argument("--save_name", help="Base name of file to save", default=None, type=str)
    parser.add_argument("--task_name", help="Name of task being ran e.g. sleep, hands, etc.", type=str)
    parser.add_argument("--dataset_name", help="Name of dataset to query", type=str, default="mmidb")
    parser.add_argument("--model_type", help="One of EEG or TID", choices=["tid", "eeg"], default="eeg", type=str)
    parser.add_argument("--surface_laplacian", help="Whether or not to use surface Laplacian filter",
                        action="store_true")

    cmd_args = parser.parse_args()

    if cmd_args.save_dir == ".":
        cmd_args.save_dir = os.path.abspath(".")

    cmd_args.model = EEGNet if cmd_args.model_type.lower() == "eeg" else TIDNet

    if cmd_args.save_name is None:
        cmd_args.save_name = f"{cmd_args.task_name}_{cmd_args.input_norm_method}_{cmd_args.model_type}_{cmd_args.dataset_name}"

    main(cmd_args)
