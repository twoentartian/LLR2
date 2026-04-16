# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import lightning as L
from .loss import ContrastiveLoss
from .models import ImageEncoder, TextEncoder


class NanoCLIP(L.LightningModule):
    """
    This class defines the pipeline for the nanoCLIP model.

    """

    def __init__(
            self,
            txt_model="sentence-transformers/all-MiniLM-L6-v2",
            img_model='dinov2_vits14',
            embed_size=64,  # output dimension of the encoder
            unfreeze_n_blocks=4,
            lr=0.001,
            freeze_encoder=False,
    ):
        super().__init__()

        self.txt_model = txt_model
        self.img_model = img_model
        self.embed_size = embed_size
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.lr = lr

        self.img_encoder = ImageEncoder(self.embed_size, self.img_model,
                                        freeze = freeze_encoder, unfreeze_n_blocks=unfreeze_n_blocks)
        self.txt_encoder = TextEncoder(self.embed_size, self.txt_model,
                                       freeze = freeze_encoder, unfreeze_n_blocks=unfreeze_n_blocks)
        self.loss_fn = ContrastiveLoss(temperature=0.05)

        self.num_training_batches_per_epoch = None
        self.latest_loss = None

    def set_batches_per_epoch(self, count):
        self.num_training_batches_per_epoch = count

    def configure_optimizers(self):
        """
        Define the optimizer and the learning rate scheduler.
        """
        # optimizer_params = [
        #     {"params": self.img_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        #     {"params": self.txt_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        # ]
        # optimizer = torch.optim.AdamW(optimizer_params)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=self.milestones, gamma=self.lr_mult
        # )
        # return optimizer, scheduler
        return None, None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """
        Define how a single optimization step is executed.
        """
        # if epoch < self.warmup_epochs:
        #     total_warmup_steps = self.warmup_epochs * self.num_training_batches_per_epoch
        #     lr_scale = min(1.0, (epoch*self.num_training_batches_per_epoch + batch_idx + 1) / total_warmup_steps)
        #     for pg in optimizer.param_groups:
        #         initial_lr = pg.get("initial_lr", self.lr)
        #         pg["lr"] = lr_scale * initial_lr

        optimizer.step(closure=optimizer_closure)

    def forward(self, image, captions, masks):
        """
        Define the forward pass of the pipeline.
        """
        # compute image embeddings
        image_embedding = self.img_encoder(image)  # (batch_size, out_dim)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)  # normalize embeddings

        # compute text embeddings
        text_embedding = self.txt_encoder(captions, masks)  # (batch_size, nb_captions, out_dim)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)  # normalize embeddings

        return image_embedding, text_embedding

    def training_step(self, batch, batch_idx):
        """
        Define a single training step (one batch pass).

        ImageEncoder ──┐
                       ├──► ContrastiveLoss
        TextEncoder  ──┘
        """
        images, captions, masks = batch

        if len(captions.shape) == 3:  # flatten captions to (batch_size*nb_caps, cap_len) cuz we have multiple captions per image
            B, nb_captions, cap_len = captions.shape
            B, nb_masks, mask_len = masks.shape
            captions = captions.view(B * nb_captions, cap_len)
            masks = masks.view(B * nb_masks, mask_len)
        else:
            nb_captions = 1

        img_descriptors, txt_descriptors = self(images, captions, masks)

        if nb_captions > 1:  # reshape back to (B, nb_captions, out_dim)
            txt_descriptors = txt_descriptors.view(B, nb_captions, -1)

        loss, batch_accuracy = self.loss_fn(img_descriptors, txt_descriptors)

        # self.log("loss", loss, prog_bar=True, logger=True)
        # self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss, batch_accuracy

    def on_validation_epoch_start(self):
        self.validation_descriptors = {"img": [], "txt": [], "nb_captions": []}

    def validation_step(self, batch, batch_idx):
        """
        Define a single validation step (one batch pass).
        """
        images, captions, masks = batch

        if len(captions.shape) == 3:  # flatten captions to (batch_size*nb_caps, cap_len) cuz we have multiple captions per image
            B, nb_captions, cap_len = captions.shape
            B, nb_masks, mask_len = masks.shape
            captions = captions.view(B * nb_captions, cap_len)
            masks = masks.view(B * nb_masks, mask_len)
        else:
            B = captions.shape[0]
            nb_captions = 1

        img_descriptors, txt_descriptors = self(images, captions, masks)
        img_descriptors = img_descriptors.detach().cpu().numpy()
        txt_descriptors = txt_descriptors.detach().cpu().numpy()

        self.validation_descriptors["img"].append(img_descriptors)
        self.validation_descriptors["txt"].append(txt_descriptors)
        self.validation_descriptors["nb_captions"].append(nb_captions)

    def on_validation_epoch_end(self):
        """
        Calculate the recall at 1, 5, and 10 for the validation set.
        """
        img_descriptors = np.concatenate(self.validation_descriptors["img"], axis=0)  # (N, out_dim)
        txt_descriptors = np.concatenate(self.validation_descriptors["txt"], axis=0)  # (N*nb_captions, out_dim)

        # create labels: each image index repeated nb_captions times to match txt_descriptors rows
        labels = np.concatenate([
            np.repeat(np.arange(img_batch.shape[0]) + offset, nb_caps)
            for img_batch, nb_caps, offset in zip(
                self.validation_descriptors["img"],
                self.validation_descriptors["nb_captions"],
                np.cumsum([0] + [b.shape[0] for b in self.validation_descriptors["img"]][:-1])
            )
        ])

        # use faiss to calculate recall, images are gallery and texts are queries
        recall_1 = self._calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1])
        self.latest_loss = recall_1
        self.validation_descriptors.clear()

    def get_validation_result(self):
        correct_count = math.nan
        return self.latest_loss, correct_count

    @staticmethod
    def _calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10]):
        """
        Calculate the recall at k for the given img_descriptors as gallery
        and txt_descriptors as queries.
        """
        embed_size = img_descriptors.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size)

        faiss_index.add(img_descriptors)  # add images to the index
        _, predictions = faiss_index.search(txt_descriptors, max(k_values))  # search for the top k images for each text query

        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], labels[q_idx])):
                    correct_at_k[i:] += 1
                    break

        correct_at_k /= len(labels)

        return correct_at_k
