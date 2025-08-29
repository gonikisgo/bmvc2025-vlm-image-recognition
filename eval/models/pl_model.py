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
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))
project_root_parent = Path(__file__).parent.parent
sys.path.append(str(project_root_parent))
from utils import save_predictions_csv, save_accuracy_results_csv

DEFAULT_NUM_CLASSES = 1000


class Classifier(pl.LightningModule):
    """
    A base class for training and evaluating image classifiers using PyTorch Lightning.
    """

    def __init__(self, cfg, is_training=False):
        super().__init__()

        self.cfg = cfg
        self.num_classes = DEFAULT_NUM_CLASSES
        self.data_dir = cfg.path.data_dir

        self.labels_to_save_df = None
        self.test_results = []

        # For all_templates, ensure we save top-11 predictions
        if self.cfg.test.context == 'all_templates':
            self.topk = max(11, self.cfg.test.topk)
        else:
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
        preds, probs = self.predict_step(x, y)

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
        }



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
                    # For all_templates, save top-11 predictions, otherwise use configured topk
                    save_topk = 11 if self.context == 'all_templates' else self.topk
                    for j in range(save_topk):
                        if self.num_classes != DEFAULT_NUM_CLASSES:
                            row[f'top_{j + 1}_pred'] = list(self.im_to_orig.keys())[list(self.im_to_orig.values()).index(preds[i][j].item())]
                        else:
                            row[f'top_{j + 1}_pred'] = preds[i][j].item()
                        row[f'top_{j + 1}_prob'] = probs[i][j].item()

                    data.append(row)

            # Save predictions CSV
            save_predictions_csv(pd.DataFrame(data), directory=f'run/{self.save_folder}/', filename=self.exp_name)
            
            # Save accuracy results CSV
            try:
                clean_labels_path = self.cfg.path.cleaner_validation
                save_accuracy_results_csv(
                    predictions_df=pd.DataFrame(data),
                    clean_labels_path=clean_labels_path,
                    directory=f'run/{self.save_folder}/',
                    filename=self.exp_name
                )
            except Exception as e:
                print(f"Warning: Could not save accuracy results: {e}")


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
        torch.cuda.empty_cache()
        return preds, probs


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

    avg = [
        lambda c: f"itap of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"art of the {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"a photo of the small {c}.",
    ]

    avg_prime = [
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
        
        # Validate and set up templates
        self._setup_templates()
        
        self.tokenizer = None
        self.tokenized_labels = None
        self.text_embeddings = None
        self.processor = None

    def _setup_templates(self):
        self._template_registry = {
            '0': {
                'templates': self.context_template_0,
                'description': 'Base template (no prompt engineering)',
                'count': 1
            },
            '1': {
                'templates': self.context_template_1,
                'description': 'Template 1: "itap of a {class}"',
                'count': 1
            },
            '2': {
                'templates': self.context_template_2,
                'description': 'Template 2: "a bad photo of the {class}"',
                'count': 1
            },
            '3': {
                'templates': self.context_template_3,
                'description': 'Template 3: "a origami {class}"',
                'count': 1
            },
            '4': {
                'templates': self.context_template_4,
                'description': 'Template 4: "a photo of the large {class}"',
                'count': 1
            },
            '5': {
                'templates': self.context_template_5,
                'description': 'Template 5: "art of the {class}"',
                'count': 1
            },
            '6': {
                'templates': self.context_template_6,
                'description': 'Template 6: "a {class} in a video game"',
                'count': 1
            },
            '7': {
                'templates': self.context_template_7,
                'description': 'Template 7: "a photo of the small {class}"',
                'count': 1
            },
            'avg': {
                'templates': self.avg,
                'description': 'Average of 7 templates (excluding base)',
                'count': 7
            },
            'avg\'': {
                'templates': self.avg_prime,
                'description': 'Average of 8 templates (including base)',
                'count': 8
            },
            'all_templates': {
                'templates': self.avg_prime,
                'description': 'All 8 templates evaluated separately (no averaging)',
                'count': 8
            }
        }
        
        if self.context not in self._template_registry:
            available_templates = list(self._template_registry.keys())
            raise ValueError(
                f"Invalid context template '{self.context}'. "
                f"Available templates: {', '.join(available_templates)}"
            )
        
        self.openai_templates = self._template_registry[self.context]['templates']
        template_info = self._template_registry[self.context]
        
        print(f"Using template '{self.context}': {template_info['description']}")
        print(f"Number of prompt variations: {template_info['count']}")

    def get_available_templates(self):
        return self._template_registry.copy()

    def get_current_template_info(self):
        return self._template_registry[self.context]

    def _generate_label_variations(self, labels_to_templ=None):
        """
        Generate label variations using the active prompt templates.
        """
        if labels_to_templ is None:
            # Load class names from file
            labels_path = Path(__file__).parent.parent.parent / 'data' / 'cls_names' / f'cls_name_{self.labels_option}.npy'
            if not labels_path.exists():
                raise FileNotFoundError(f"Class names file not found: {labels_path}")
            
            labels_list = np.load(labels_path)
        else:
            labels_list = labels_to_templ
        
        # Special handling for all_templates: map all classes to each template
        if self.context == 'all_templates':
            variations = []
            num_templates = len(self.openai_templates)
            
            # Outer loop: iterate through templates
            for template_idx in range(num_templates):
                template = self.openai_templates[template_idx]
                # Inner loop: for each template, map ALL classes
                for i, label in enumerate(labels_list):
                    try:
                        variation = template(label)
                        variations.append(variation)
                    except Exception as e:
                        print(f"Warning: Failed to apply template to label '{label}': {e}")
                        # Fallback to base label if template fails
                        variations.append(str(label))
            
            print(f"Generated {len(variations)} label variations from {len(labels_list)} base labels using all_templates mapping")
            print(f"Template mapping: {num_templates} templates × {len(labels_list)} classes = {len(variations)} total variations")
            return variations
        else:
            # Apply all templates to all labels with error handling
            variations = []
            for label in labels_list:
                for template in self.openai_templates:
                    try:
                        variation = template(label)
                        variations.append(variation)
                    except Exception as e:
                        print(f"Warning: Failed to apply template to label '{label}': {e}")
                        # Fallback to base label if template fails
                        variations.append(str(label))
            
            print(f"Generated {len(variations)} label variations from {len(labels_list)} base labels")
            return variations

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
