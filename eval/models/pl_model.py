import os
import sys
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from timm import create_model
from torch.distributed import all_gather_object
from torchmetrics import Accuracy
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau
from class_mapping import ClassDictionary

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import save_predictions_csv

DEFAULT_NUM_CLASSES = 1000


class Classifier(pl.LightningModule):
    """
    A base class for training and evaluating image classifiers using PyTorch Lightning.
    """

    def __init__(self, cfg, is_training=False):
        super().__init__()

        self.cfg = cfg
        self.num_classes = cfg.data.n_classes
        self.data_dir = cfg.path.data_dir

        self.labels_to_save_df = None
        self.test_results = []

        self.topk = self.cfg.test.topk
        self.exp_name = None
        self.lr = None
        self.lr_scheduler = None
        self.weight_decay = None
        self.max_epochs = None
        self.weights = None
        self.loss_fn = None
        self.loss = None
        self.optim = None
        self.train_acc = None
        self.val_acc = None
        self.seed = None
        self.batch_size = None
        self.mode = 'base'
        self.image_transform = None
        self.save_folder = None
        self.is_using_k_folds = None

        self.configure_for_testing()

        if cfg.test.mode != 'dino':
            class_dict = ClassDictionary()

            if self.num_classes != DEFAULT_NUM_CLASSES:
                self.class_list = [class_dict.get_cls_index(int(i)) for i in cfg.data.cls_list]
                self.class_names = [class_dict.get_custom_class_name(i) for i in cfg.data.cls_list]
                self.im_to_orig = class_dict.create_id2idx_map(cfg.data.cls_list)
            else:
                self.class_names = np.load(cfg.path.class_names_path)

            self.cls_name_dict = class_dict.create_idx2id_map(self.class_names)

    def configure_for_testing(self):
        """Testing-specific setup."""
        self.exp_name = self.cfg.test.exp_name
        self.mode = self.cfg.test.mode
        self.save_folder = self.cfg.test.folder
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        lr_multiplier = 10

        layers = list(self.model.children())
        num_layers = len(layers)
        lr_increment = (lr_multiplier / self.lr_start) ** (1 / (num_layers - 1))

        param_groups = []
        current_lr = self.lr_start

        for layer in layers[:-1]:
            param_groups.append({'params': layer.parameters(), 'lr': current_lr, 'weight_decay': self.weight_decay})
            current_lr *= lr_increment

        param_groups.append({
            'params': layers[-1].parameters(),
            'lr': self.lr_start * self.lr_multiplier,
            'weight_decay': self.weight_decay
        })
        optimizer = self.optim(param_groups)

        if self.lr_scheduler == "none":
            return optimizer
        elif self.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss'
                }
            }
        return None

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y, reduction='none', weight=self.weights)
        loss_mean = loss.mean()
        acc = self.val_acc(logits, y)

        class_losses = torch.zeros(self.num_classes).to('cuda')
        class_accuracies = torch.zeros(self.num_classes).to('cuda')

        for i in range(self.num_classes):
            class_mask = (y == i)
            if class_mask.any():
                class_losses[i] = loss[class_mask].mean()
                class_preds = logits[class_mask].argmax(dim=1)
                class_accuracies[i] = (class_preds == i).float().mean()
            else:
                class_losses[i] = torch.tensor(0.0, device='cuda')
                class_accuracies[i] = torch.tensor(0.0, device='cuda')

        self.log('val/loss', loss_mean, on_step=False, on_epoch=True, logger=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, logger=True)
        for i in range(self.num_classes):
            self.log(f'val/{self.cls_name_dict[i]}_loss', class_losses[i], on_step=False, on_epoch=True, logger=True)
            self.log(f'val/{self.cls_name_dict[i]}_acc', class_accuracies[i], on_step=False, on_epoch=True, logger=True)

    def on_train_epoch_start(self):
        """
        Set the epoch for the dataset to ensure that the same augmentations are applied to the same images.
        """
        if self.is_using_k_folds:
            self.trainer.train_dataloader.dataset.dataset.set_epoch(self.current_epoch)
        else:
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)

    def test_step(self, batch, batch_idx):
        image_names, x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        preds, probs, suppl = self.predict_step(x, y)

        loss = torch.tensor([0.])
        preds_ = preds[:, 0]
        acc = self.val_acc(preds_, y)

        self.log('test_loss', loss, prog_bar=False)
        self.log('test_acc', acc, prog_bar=False)

        # The results returned by `test_step` in a dictionary form
        return {
            'test_loss': loss,  # The calculated test loss
            'test_acc': acc,  # The accuracy metric, computed as the proportion of correctly predicted labels
            'original_labels': y,  # The ground truth labels (targets) for the images in the batch
            'image_names': image_names,  # The names of the images in the batch for logging purposes
            'preds': preds,  # The predicted labels from the model, returned by the `predict_step`
            'probs': probs,  # The predicted probabilities from the model for each predicted class in a step
            'supplement': suppl  # Additional supplemental information returned in case of extended running mode
        }

    def create_label_confidence(self, original_labels, outputs):
        if self.mode != 'base':
            filtered_df = self.labels_to_save_df[self.labels_to_save_df['original_label'].isin(original_labels.tolist())]
            proposed_labels = filtered_df['proposed_labels'].tolist()

            suppl = [
                [outputs[1, self.im_to_orig[int(ol_label)]], outputs[0, self.im_to_orig[int(ol_label)]],
                 outputs[1, self.im_to_orig[int(pr_label)]] if not np.isnan(pr_label) else np.nan,
                 outputs[0, self.im_to_orig[int(pr_label)]] if not np.isnan(pr_label) else np.nan]
                for ol_label, pr_label in zip(original_labels, proposed_labels)
            ]
            suppl = [val.cpu().numpy() if isinstance(val, torch.Tensor) else val for item in suppl for val in item]
            return np.array(suppl)
        else:
            return np.array([])

    def on_test_batch_end(self, outputs, batch, batch_idx):
        self.test_results.append(outputs)

    def on_test_end(self):
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        gathered_results = [self.test_results]

        if is_distributed:
            gathered_results = [None for _ in range(torch.distributed.get_world_size())]
            all_gather_object(gathered_results, self.test_results)

        if self.trainer.global_rank == 0:
            flattened_results = [item for sublist in gathered_results for item in sublist]

            data = []
            for output in flattened_results:
                labels, image_names = output['original_labels'], output['image_names']
                preds, probs = output['preds'], output['probs']

                for i in range(len(labels)):
                    row = {
                        'original_label': labels[i].item(),
                        'img_id': image_names[i]
                    }
                    for j in range(self.topk):
                        if self.num_classes != DEFAULT_NUM_CLASSES:
                            row[f'top_{j + 1}_pred'] = list(self.im_to_orig.keys())[list(self.im_to_orig.values()).index(preds[i][j].item())]
                        else:
                            row[f'top_{j + 1}_pred'] = preds[i][j].item()
                        row[f'top_{j + 1}_prob'] = probs[i][j].item()

                    if self.mode != 'base':
                        suppl = output['supplement']
                        row[f'original_label'] = suppl[0].item()
                        row[f'original_label_conf'] = suppl[1].item()
                        row[f'proposed_label'] = suppl[2].item()
                        row[f'proposed_label_conf'] = suppl[3].item()
                    data.append(row)

            save_predictions_csv(pd.DataFrame(data), directory=f'run/{self.save_folder}/', filename=self.exp_name)


class TimmClassifier(Classifier):
    """
    A classifier that uses models from the timm library.
    """
    model2timm = {
        'efficientnet_l2': 'hf_hub:timm/tf_efficientnet_l2.ns_jft_in1k',
        'efficientnet_v2': 'hf_hub:timm/tf_efficientnetv2_xl.in21k_ft_in1k'
    }

    def __init__(self, cfg, is_training=False, reset_last_layer=True):
        super().__init__(cfg, is_training=is_training)
        self.model_name = cfg.train.model if is_training else cfg.test.model
        self.model = create_model(self.model2timm[self.model_name], pretrained=bool(cfg.train.pretrained),
                                  num_classes=self.num_classes)
        if is_training and reset_last_layer:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.num_classes)

    def get_image_transform(self, is_training=False):
        """
        Retrieve the image transformation pipeline based on the model's configuration.

        Returns:
            Any: Transformation pipeline for images.
        """
        data_config = timm.data.resolve_model_data_config(self.model)
        return timm.data.create_transform(**data_config, is_training=is_training)

    def on_test_start(self):
        self.model = self.model.eval()

    def predict_step(self, images, labels):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            outputs = self.model(images).softmax(dim=1)
            probs, preds = torch.topk(outputs, k=self.topk, dim=-1)
            suppl = self.create_label_confidence(labels, outputs)
        torch.cuda.empty_cache()
        return preds, probs, suppl


class HFTransformersClassifier(Classifier):
    """
    A classifier that uses transformer-based models from the Hugging Face's transformers.
    """
    model2transformers = {
        'OpenCLIP': 'hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',
        'CLIP': 'openai/clip-vit-large-patch14-336',
        'DINOv2': 'facebook/dinov2-giant',
        'SigLIP': 'google/siglip-so400m-patch14-384',
        'SigLIPv2_timm': 'hf-hub:timm/ViT-SO400M-14-SigLIP2-378',
        'SigLIP_v2_g_timm': 'hf-hub:timm/ViT-gopt-16-SigLIP2-384',
        'RADIO-G': 'nvidia/C-RADIOv2-G'
    }

    context_template_0 = [lambda c: f"{c}"]
    context_template_1 = [lambda c: f"itap of a {c}."]
    context_template_2 = [lambda c: f"a bad photo of the {c}."]
    context_template_3 = [lambda c: f"a origami {c}."]
    context_template_4 = [lambda c: f"a photo of the large {c}."]
    context_template_5 = [lambda c: f"art of the {c}."]
    context_template_6 = [lambda c: f"a {c} in a video game."]
    context_template_7 = [lambda c: f"a photo of the small {c}."]

    mean7 = [
        lambda c: f"itap of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"art of the {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"a photo of the small {c}.",
    ]

    mean8 = [
        lambda c: f"{c}",
        lambda c: f"itap of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"art of the {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"a photo of the small {c}.",
    ]

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_name = cfg.test.model
        self.labels_option = cfg.test.labels_option
        self.context = cfg.test.context
        print(self.context)

        self.tokenizer = None
        self.tokenized_labels = None
        self.text_embeddings = None
        self.processor = None

        self.context_map = {
            'mean7': self.mean7,
            'mean8': self.mean8,
            '0': self.context_template_0,
            '1': self.context_template_1,
            '2': self.context_template_2,
            '3': self.context_template_3,
            '4': self.context_template_4,
            '5': self.context_template_5,
            '6': self.context_template_6,
            '7': self.context_template_7,
        }
        self.openai_templates = self.context_map[self.context]

    def _generate_label_variations(self, labels_to_templ=None):
        """
        Generate label variations using predefined templates.

        Returns:
            List[str]: List of label variations.
        """
        labels_list = np.load(
            os.path.join(os.path.dirname(__file__), f'../../data/cls_names/cls_name_{self.labels_option}.npy'))
        return [template(label) for label in labels_list for template in self.openai_templates]  # TODO Improve logic

    def _generate_text_embeddings(self):
        """
        Generate text embeddings for the original_labels using the model's tokenizer.
        """
        pass

    def get_image_transform(self, is_training=False):
        """
        Retrieve the image transformation pipeline based on the model's configuration.

        Returns:
            Any: Transformation pipeline for images.
        """
        return self.processor
