import sys
from pathlib import Path

import numpy as np
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from torch.distributed import all_gather_object

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.pl_model import HFTransformersClassifier
from eval.utils import save_embeddings_to_npy


class OpenCLIP(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model, self.processor, self.image_transform = create_model_and_transforms(self.model2transformers[self.model_name])
        self.model = self.model.to('cuda')
        self.tokenizer = get_tokenizer(self.model2transformers[self.model_name])
        self.tokenized_labels = self.tokenizer(self._generate_label_variations()).to('cuda')
        print(len(self.tokenized_labels))
        self.text_embeddings = self._generate_text_embeddings()

    def _generate_text_embeddings(self):
        text_embeddings = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            print(f"Generating tokenized labels for {len(self.openai_templates)} promts")
            
            if self.context == 'all_templates':
                # For all_templates, each class is tokenized separately (8000 total)
                for i in range(len(self.tokenized_labels)):
                    text_variation = self.tokenized_labels[i:i+1]  # Single class
                    text_features = self.model.encode_text(text_variation, normalize=True).to('cuda')
                    text_features = text_features.squeeze(0)  # Remove batch dimension
                    text_features = text_features / text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            else:
                # Original behavior: average templates
                for i in range(0, len(self.tokenized_labels), len(self.openai_templates)):
                    text_variation = self.tokenized_labels[i:i + len(self.openai_templates)]
                    text_features = self.model.encode_text(text_variation, normalize=True).to('cuda')
                    text_features = text_features.mean(dim=0)
                    text_features /= text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            
        return embeddings_tensor

    def on_test_start(self):
        self.text_embeddings = self.text_embeddings.to(self.device)

    def predict_step(self, images, labels):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.encode_image(images).to(self.device)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ self.text_embeddings.T).softmax(dim=-1)
            probs, preds = torch.topk(text_probs, k=self.topk, dim=-1)
        torch.cuda.empty_cache()
        return preds, probs


class OpenClipEmbedder(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model, self.processor, self.image_transform = create_model_and_transforms(self.model2transformers[self.model_name])

    def predict_step(self, images):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.encode_image(images).to(self.device)
        return image_features

    def test_step(self, batch, batch_idx):
        image_names, images, labels = batch
        embs = self.predict_step(images)
        return labels, image_names, embs

    def on_test_end(self):
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        gathered_results = [self.test_results]

        if is_distributed:
            gathered_results = [None for _ in range(torch.distributed.get_world_size())]
            all_gather_object(gathered_results, self.test_results)

        if self.trainer.global_rank == 0:
            save_embeddings_to_npy(
                results=gathered_results,
                model_name=self.model_name,
                dataloader_name=self.cfg.test.dataloader,
                convert_labels_to_int=False,  # OpenCLIP doesn't convert labels to int
                project_root=project_root
            )


class OpenClipTextEmbedder(HFTransformersClassifier):
    """
    OpenCLIP Text Embedder that saves only the encoded text embeddings.
    This is useful for saving pre-computed text embeddings for different class label sets and templates.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model, self.processor, self.image_transform = create_model_and_transforms(self.model2transformers[self.model_name])
        self.model = self.model.to('cuda')
        self.tokenizer = get_tokenizer(self.model2transformers[self.model_name])
        self.tokenized_labels = self.tokenizer(self._generate_label_variations()).to('cuda')
        print(f"Tokenized labels length: {len(self.tokenized_labels)}")
        self.text_embeddings = self._generate_text_embeddings()
        
    def _generate_text_embeddings(self):
        text_embeddings = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            print(f"Generating text embeddings for {len(self.openai_templates)} templates and {self.labels_option} labels")
            
            if self.context == 'all_templates':
                # For all_templates, each class-template combination is saved separately 
                for i in range(len(self.tokenized_labels)):
                    text_variation = self.tokenized_labels[i:i+1]  # Single class-template pair
                    text_features = self.model.encode_text(text_variation, normalize=True).to('cuda')
                    text_features = text_features.squeeze(0)  # Remove batch dimension
                    text_features = text_features / text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            else:
                # For other contexts: average templates per class
                for i in range(0, len(self.tokenized_labels), len(self.openai_templates)):
                    text_variation = self.tokenized_labels[i:i + len(self.openai_templates)]
                    text_features = self.model.encode_text(text_variation, normalize=True).to('cuda')
                    text_features = text_features.mean(dim=0)
                    text_features /= text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            
        return embeddings_tensor
    
    def save_text_embeddings(self):
        """
        Save the generated text embeddings to a numpy file.
        """
        # Generate metadata for saved embeddings
        labels_path = Path(__file__).parent.parent.parent / 'data' / 'cls_names' / f'cls_name_{self.labels_option}.npy'
        class_names = np.load(labels_path)
        
        if self.context == 'all_templates':
            # For all_templates: create expanded class names for each template-class pair
            expanded_class_names = []
            for template_idx, template in enumerate(self.openai_templates):
                for class_name in class_names:
                    expanded_class_names.append(f"template_{template_idx}_{class_name}")
            class_identifiers = expanded_class_names
        else:
            # For averaged templates: use original class names
            class_identifiers = class_names.tolist()
            
        # Prepare data structure
        text_embeddings_np = self.text_embeddings.cpu().numpy()
        data = {
            'class_names': class_identifiers,
            'text_embeddings': text_embeddings_np,
            'context': self.context,
            'labels_option': self.labels_option,
            'model_name': self.model_name,
            'num_templates': len(self.openai_templates),
            'embedding_dim': text_embeddings_np.shape[-1]
        }
        
        # Create save directory and filename
        save_dir = Path(__file__).parent.parent.parent / 'eval' / 'results' / 'text_embeddings' / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f'{self.model_name}_text_embeddings_{self.labels_option}_{self.context}.npy'
        save_path = save_dir / filename
        
        # Save embeddings
        np.save(save_path, data)
        print(f'✓ Text embeddings saved to: {save_path}')
        print(f'  - Shape: {text_embeddings_np.shape}')
        print(f'  - Context: {self.context}')
        print(f'  - Labels: {self.labels_option}')
        print(f'  - Classes: {len(class_identifiers)}')
        
        return save_path
    
    def test_step(self, batch, batch_idx):
        """Override test_step to do nothing since we only want to save text embeddings."""
        return None
        
    def on_test_start(self):
        """Save text embeddings at the start of testing."""
        print("Saving text embeddings...")
        self.save_text_embeddings()
        
    def on_test_end(self):
        """Override to prevent default behavior since we're only saving text embeddings."""
        print("Text embeddings saved successfully!")
