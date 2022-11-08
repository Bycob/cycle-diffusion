# Created by Chen Henry Wu
import logging
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../"))

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
)
from utils.config_utils import get_config
from utils.program_utils import get_model, get_preprocessor, get_evaluator, get_visualizer
from preprocess.to_model import get_multi_task_dataset_splits
from utils.training_arguments import CustomTrainingArguments
from trainer.trainer import Trainer
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)



def setup_wandb(training_args):
    if "wandb" in training_args.report_to and training_args.local_rank <= 0:
        import wandb

        # init_args = {}
        # if "MLFLOW_EXPERIMENT_ID" in os.environ:
        #     init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "your project name"),
            name=training_args.run_name,
            entity=os.getenv("WANDB_ENTITY", 'your entity'),
        )
        wandb.config.update(training_args, allow_val_change=True)

        return wandb.run.dir
    else:
        return None

def main():
    # Get training_args and args.
    parser = HfArgumentParser(
        (
            CustomTrainingArguments,
        )
    )
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = get_config(training_args.cfg)

    prompt_from = "A busy street in new york city"
    prompt_to = "A pedestrian is crossing a busy street in new york city"
    path = "new_york_street_1024.png"
    img = load_image(path, pix_range=(-1, 1))

    # Deterministic behavior of torch.addmm.
    # Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    # cudnn.deterministic = True

    # Setup output directory.
    os.makedirs(training_args.output_dir, exist_ok=True)
    args.output_dir = training_args.output_dir
    # Initialize evaluator.
    evaluator = get_evaluator(args.evaluation.evaluator_program)(args)
    # Initialize visualizer.
    visualizer = get_visualizer(args.visualization.visualizer_program)(args)

    # Initialize model.
    model = get_model(args.model.name)(args)

    # Initialize Trainer.
    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=evaluator.evaluate,
        visualizer=visualizer,
        wandb_run_dir=None,
    )
    print(f'Rank {training_args.local_rank} Trainer build successfully.')
    # Test
    logger.info("*** Predict ***")
    output_images = inference(trainer, {
        "sample_id": torch.LongTensor([0]).squeeze(0),
        "original_image": img,
        "encode_text": [prompt_from],
        "decode_text": [prompt_to],
        # "model_kwargs": ["sample_id", "encode_text", "decode_text", "original_image"]
    })
    output_path = "output.png"
    print("Saving image to %s" % (output_path,))
    to_images(output_images[0]).save(output_path)

def to_images(batch: torch.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())

def load_image(path, pix_range=(0, 1), size=512):
    a, b = pix_range
    factor = 255 / (b-a)
    input_image = np.asarray(Image.open(path).resize((size, size)))
    tensor_image = torch.from_numpy(input_image.copy()).to(torch.float) / factor + a
    if len(tensor_image.shape) == 2: # greyscale
        tensor_image = tensor_image.unsqueeze(2)
    if tensor_image.shape[2] > 3: # rgba
        tensor_image = tensor_image[:,:,:3]
    tensor_image = tensor_image.permute((2, 0, 1)).unsqueeze(0)
    return tensor_image


def inference(self, input_images):
    """
    Run prediction and returns predictions results.
    """
    self.model.eval()
    prediction_outputs = self.prediction_step(input_images)
    images, weighted_loss, losses = prediction_outputs
    return images


if __name__ == "__main__":
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    main()
    # from time import sleep
    # sleep(3600 * 48)
