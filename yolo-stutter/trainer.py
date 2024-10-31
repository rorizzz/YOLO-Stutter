import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

import copy

import gc

import wandb

from guppy import hpy

from utils.vits.models import SynthesizerTrn
from utils.vits.text.symbols import symbols
from utils.model_utils.dataset import Dataset
# from utils.model_utils.dataset_emap import Dataset
# from utils.model_utils.dataset_ctvec import Dataset
# from utils.model_utils.dataset_wavlm import Dataset
from utils.model_utils.conv1d_transformer import Conv1DTransformerDecoder
import utils.vits.utils as utils


# hyperparams + device initialization
device = torch.device("cuda:0")
# torch.cuda.set_device(device)
# torch.multiprocessing.set_start_method('spawn', force=True)

EPOCHS = 20
lr = 3e-4
weight_decay = 0

MODEL_NAME = "decoder_vctk_tts"
DATASET = "VCTK-tts"

# wandb initialization
wandb.init(
    # set the wandb project where this run will be logged
    project="Yolo-stutter-0209",
    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "architecture": MODEL_NAME,
        "dataset": DATASET,
        "epochs": EPOCHS,
    },
)


class Trainer:
    def __init__(
        self,
        text_channels=768,
        kernel_size=3,
        kernel_stride=1,
        num_blocks=4,
        num_classes=4,
        downsample_factor=16,  ## original 16 -> 8
        num_transformer_layers=8,
        n_heads=8,
        config_path="./utils/vits/configs/ljs_base.json",
        epochs=20,  #20 -> 25
        lr=3e-4, 
        num_steps=50,
        batch_size=64,  #64  -> 32
    ):
        self.decoder = Conv1DTransformerDecoder(
            text_channels,
            kernel_size,
            kernel_stride,
            num_blocks,
            num_classes,
            num_transformer_layers,
            n_heads,
        ).to(device)

        self.hps = utils.get_hparams_from_file(config_path)

        self.h = hpy()

        # soft alignments from VITS
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model
        ).cuda()

        self.net_g, _, _, _ = utils.load_checkpoint(
            "./saved_models/pretrained_ljs.pth", self.net_g, None
        )
        self.net_g = self.net_g.to(device)
        self.net_g.eval()

        # loss + optimizer initialization
        self.bound_loss = nn.MSELoss()
        self.disfluency_exists_loss = nn.BCELoss(reduction="mean")
        self.disfluency_type_loss = nn.CrossEntropyLoss()

        self.optimizer = AdamW(self.decoder.parameters(), lr=lr, weight_decay=0)

        self.num_steps = num_steps 
        self.batch_size = batch_size
        self.epochs = epochs
        self.bound_scale_factor = 4
        self.no_disf_scale_factor = 0.9
        self.downsample_factor = downsample_factor

        self.steps = 0

        # for batch training

    def collate_fn(self, batch):
        texts = []
        text_lengths = []
        specs = []
        spec_lengths = []
        label_matrices = []
        num_regions = []
        region = []
        start_region = []
        end_region = []

        for sample in batch:
            if sample is None:
                continue

            texts += [sample[0]]
            text_lengths += [sample[1]]
            specs += [torch.transpose(sample[2], 0, 1)]
            spec_lengths += [sample[3]]
            label_matrices += [sample[4]]
            num_regions += [sample[5]]
            region += [sample[6]]
            start_region += [sample[7]]
            end_region += [sample[8]]

        return (
            pad_sequence(texts, batch_first=True),
            torch.tensor(text_lengths),
            torch.transpose(pad_sequence(specs, batch_first=True), -1, -2),
            torch.tensor(spec_lengths),
            pad_sequence(label_matrices, batch_first=True),
            torch.tensor(num_regions),
            torch.tensor(region),
            torch.tensor(start_region),
            torch.tensor(end_region),
        )

    def get_soft_attention(self, hps, net_g, text, text_lengths, spec, spec_lengths):
        with torch.no_grad():
            output = net_g(
                text, text_lengths, spec, spec_lengths
            )  # phoneme, mel_spec respectively

            neg_cent = nn.functional.softmax(output[2][0], dim=-1)
            del output
            gc.collect()

        return neg_cent


    def eval_model(self, train_loader, val_loader, decoder):
        # decoder evaluation
        # decoder.eval()

        with torch.no_grad():
            # train set
            train_bound_losses_disf = []
            train_exists_losses_disf = []
            train_exists_losses_no_disf = []
            train_type_losses_disf = []

            train_disfluency_exists_accuracy = []
            train_disfluency_type_accuracy = []

            for _, batch in enumerate(tqdm(train_loader)):
                if any(x is None for x in batch):
                    continue
                text, text_length, spec, spec_length, label_matrix, num_regions, region, start_region, end_region = batch
                 
            # for _, (
            #     text,
            #     text_length,
            #     spec,
            #     spec_length,
            #     label_matrix,
            #     num_regions,
            #     start_region,
            #     end_region,
            # ) in enumerate(tqdm(train_loader)):
                text = text.to(device)
                text_length = text_length.to(device)
                spec = spec.to(device)
                spec_length = spec_length.to(device)
                label_matrix = label_matrix.to(device)

                soft_attention = self.get_soft_attention(
                    self.hps, self.net_g, text, text_length, spec, spec_length
                )  # dims - batch_size, speech_temporal_dim, text_temporal_dim/feature_dim

                # max feature dim size of 1024
                soft_attention = nn.functional.pad(
                    soft_attention,
                    (
                        0,
                        768 - soft_attention.shape[-1],
                        0,
                        1024 - soft_attention.shape[-2],
                    ),
                )

                label_matrix = nn.functional.pad(
                    label_matrix,
                    (
                        0,
                        0,
                        0,
                        64 - label_matrix.shape[-2],
                    ),
                )

                # seq length
                mask = torch.ones((label_matrix.shape[0], 64), dtype=torch.bool)
                for i in range(label_matrix.shape[0]):
                    mask[i, : num_regions[i] + 1] = False

                output = decoder(soft_attention, mask=mask.to(device))
                # dims - start, end, exist, 4types

                bound_loss = 0
                confidence_loss_no_disf = 0
                confidence_loss_disf = 0
                type_loss = 0

                for i in range(label_matrix.shape[0]):
                    no_disf_regions = torch.cat(
                        (
                            label_matrix[i][: start_region[i]],
                            label_matrix[i][end_region[i] + 1 : num_regions[i] + 1],
                        ),
                        dim=-2,
                    )
                    no_disf_output = torch.cat(
                        (
                            output[i][: start_region[i]],
                            output[i][end_region[i] + 1 : num_regions[i] + 1],
                        ),
                        dim=-2,
                    )

                    disf_regions = label_matrix[i][start_region[i] : end_region[i] + 1]
                    disf_output = output[i][start_region[i] : end_region[i] + 1]

                    main_region = label_matrix[i][region[i]]
                    main_output = output[i][region[i]]

                    # loss computations
                    bound_loss += self.bound_loss(main_output[:2], main_region[:2])

                    confidence_loss_no_disf += self.disfluency_exists_loss(
                        no_disf_output[:, 2], no_disf_regions[:, 2]
                    )

                    confidence_loss_disf += self.disfluency_exists_loss(
                        disf_output[:, 2], disf_regions[:, 2]
                    )

                    type_loss += self.disfluency_type_loss(
                        disf_output[:, 3:], disf_regions[:, 3:]
                    )

                # meaning based on batch size
                bound_loss /= label_matrix.shape[0]
                confidence_loss_no_disf /= label_matrix.shape[0]
                confidence_loss_disf /= label_matrix.shape[0]
                type_loss /= label_matrix.shape[0]

                exists_labels = label_matrix[:, :, 2]
                exists_mask = exists_labels > 0
                disfluency_type_pred, disfluency_type_labels = (
                    output[:, :, 3:],
                    label_matrix[:, :, 3:],
                )

                disfluency_type_pred = disfluency_type_pred[exists_mask]
                disfluency_type_labels = disfluency_type_labels[exists_mask]

                type_pred_softmax = torch.log_softmax(disfluency_type_pred, dim=-1)
                a, y_pred_labels = torch.max(type_pred_softmax, dim=-1)
                a, disfluency_type_labels = torch.max(disfluency_type_labels, dim=-1)
                correct_type_pred = y_pred_labels == disfluency_type_labels
                type_acc = correct_type_pred.sum() / len(correct_type_pred)
                type_acc = torch.round(type_acc * 100)
                train_disfluency_type_accuracy += [type_acc]

                curr_exists_accuracy = 0
                for i in range(label_matrix.shape[0]):
                    exists_pred = torch.clamp(output[i, : num_regions[i], 2], 0, 1)
                    exists_pred_tags = (exists_pred > 0.5).float()
                    correct_exists_pred = (
                        exists_pred_tags == exists_labels[i, : num_regions[i]]
                    )
                    exists_acc = correct_exists_pred.sum() / (len(correct_exists_pred))
                    exists_acc = torch.round(exists_acc * 100)
                    curr_exists_accuracy += exists_acc

                train_disfluency_exists_accuracy += [
                    curr_exists_accuracy / label_matrix.shape[0]
                ]

                train_bound_losses_disf += [bound_loss.item()]
                train_exists_losses_disf += [confidence_loss_disf.item()]
                train_exists_losses_no_disf += [confidence_loss_no_disf.item()]
                train_type_losses_disf += [type_loss.item()]

                if _ >= self.num_steps:
                    break

            # val set
            val_bound_losses_disf = []
            val_exists_losses_disf = []
            val_exists_losses_no_disf = []
            val_type_losses_disf = []

            val_disfluency_exists_accuracy = []
            val_disfluency_type_accuracy = []

            for batch in tqdm(val_loader):
                if None in batch:
                    continue  #skip
                text, text_length, spec, spec_length, label_matrix, num_regions, region, start_region, end_region = batch
            # for (
            #     text,
            #     text_length,
            #     spec,
            #     spec_length,
            #     label_matrix,
            #     num_regions,
            #     start_region,
            #     end_region,
            # ) in tqdm(val_loader):
                
                text = text.to(device)
                text_length = text_length.to(device)
                spec = spec.to(device)
                spec_length = spec_length.to(device)
                label_matrix = label_matrix.to(device)

                soft_attention = self.get_soft_attention(
                    self.hps, self.net_g, text, text_length, spec, spec_length
                )  # dims - batch_size, speech_temporal_dim, text_temporal_dim/feature_dim

                # max feature dim size of 1024
                soft_attention = nn.functional.pad(
                    soft_attention,
                    (
                        0,
                        768 - soft_attention.shape[-1],
                        0,
                        1024 - soft_attention.shape[-2],
                    ),
                )

                label_matrix = nn.functional.pad(
                    label_matrix,
                    (
                        0,
                        0,
                        0,
                        64 - label_matrix.shape[-2],
                    ),
                )

                # seq length
                mask = torch.ones((label_matrix.shape[0], 64), dtype=torch.bool)
                for i in range(label_matrix.shape[0]):
                    mask[i, : num_regions[i] + 1] = False

                output = decoder(soft_attention, mask=mask.to(device))

                bound_loss = 0
                confidence_loss_no_disf = 0
                confidence_loss_disf = 0
                type_loss = 0

                for i in range(label_matrix.shape[0]):
                    no_disf_regions = torch.cat(
                        (
                            label_matrix[i][: start_region[i]],
                            label_matrix[i][end_region[i] + 1 : num_regions[i] + 1],
                        ),
                        dim=-2,
                    )
                    no_disf_output = torch.cat(
                        (
                            output[i][: start_region[i]],
                            output[i][end_region[i] + 1 : num_regions[i] + 1],
                        ),
                        dim=-2,
                    )

                    disf_regions = label_matrix[i][start_region[i] : end_region[i] + 1]
                    disf_output = output[i][start_region[i] : end_region[i] + 1]

                    main_region = label_matrix[i][region[i]]
                    main_output = output[i][region[i]]


                    # loss computations
                    bound_loss += self.bound_loss(main_output[:2], main_region[:2])

                    confidence_loss_no_disf += self.disfluency_exists_loss(
                        no_disf_output[:, 2], no_disf_regions[:, 2]
                    )

                    confidence_loss_disf += self.disfluency_exists_loss(
                        disf_output[:, 2], disf_regions[:, 2]
                    )

                    type_loss += self.disfluency_type_loss(
                        disf_output[:, 3:], disf_regions[:, 3:]
                    )

                # meaning based on batch size
                bound_loss /= label_matrix.shape[0]
                confidence_loss_no_disf /= label_matrix.shape[0]
                confidence_loss_disf /= label_matrix.shape[0]
                type_loss /= label_matrix.shape[0]

                exists_labels = label_matrix[:, :, 2]
                exists_mask = exists_labels > 0
                disfluency_type_pred, disfluency_type_labels = (
                    output[:, :, 3:],
                    label_matrix[:, :, 3:],
                )

                disfluency_type_pred = disfluency_type_pred[exists_mask]
                disfluency_type_labels = disfluency_type_labels[exists_mask]

                type_pred_softmax = torch.log_softmax(disfluency_type_pred, dim=-1)
                _, y_pred_labels = torch.max(type_pred_softmax, dim=-1)
                _, disfluency_type_labels = torch.max(disfluency_type_labels, dim=-1)
                correct_type_pred = y_pred_labels == disfluency_type_labels
                type_acc = correct_type_pred.sum() / len(correct_type_pred)
                type_acc = torch.round(type_acc * 100)
                val_disfluency_type_accuracy += [type_acc]

                curr_exists_accuracy = 0
                for i in range(label_matrix.shape[0]):
                    exists_pred = torch.clamp(output[i, : num_regions[i], 2], 0, 1)
                    exists_pred_tags = (exists_pred > 0.5).float()
                    correct_exists_pred = (
                        exists_pred_tags == exists_labels[i, : num_regions[i]]
                    )
                    exists_acc = correct_exists_pred.sum() / (len(correct_exists_pred))
                    exists_acc = torch.round(exists_acc * 100)
                    curr_exists_accuracy += exists_acc

                val_disfluency_exists_accuracy += [
                    curr_exists_accuracy / label_matrix.shape[0]
                ]

                val_bound_losses_disf += [bound_loss.item()]
                val_exists_losses_disf += [confidence_loss_disf.item()]
                val_exists_losses_no_disf += [confidence_loss_no_disf.item()]
                val_type_losses_disf += [type_loss.item()]

        train_bound_loss_disf = np.mean(np.array(train_bound_losses_disf))
        train_exists_loss_disf = np.mean(np.array(train_exists_losses_disf))
        train_exists_loss_no_disf = np.mean(np.array(train_exists_losses_no_disf))
        train_type_loss_disf = np.mean(np.array(train_type_losses_disf))

        train_disfluency_exists_accuracy = torch.mean(
            torch.tensor(train_disfluency_exists_accuracy)
        ).item()
        train_disfluency_type_accuracy = torch.mean(
            torch.tensor(train_disfluency_type_accuracy)
        ).item()

        val_bound_loss_disf = np.mean(np.array(val_bound_losses_disf))
        val_exists_loss_disf = np.mean(np.array(val_exists_losses_disf))
        val_exists_loss_no_disf = np.mean(np.array(val_exists_losses_no_disf))
        val_type_loss_disf = np.mean(np.array(val_type_losses_disf))

        val_disfluency_exists_accuracy = torch.mean(
            torch.tensor(val_disfluency_exists_accuracy)
        ).item()
        val_disfluency_type_accuracy = torch.mean(
            torch.tensor(val_disfluency_type_accuracy)
        ).item()

        print("Train bound loss disf: " + str(train_bound_loss_disf))
        print("Train exists loss disf: " + str(train_exists_loss_disf))
        print("Train exists loss no disf: " + str(train_exists_loss_no_disf))
        print("Train type loss disf: " + str(train_type_loss_disf))

        print("Train exists acc: " + str(train_disfluency_exists_accuracy))
        print("Train disfluency type acc: " + str(train_disfluency_type_accuracy))

        print("Val bound loss disf: " + str(val_bound_loss_disf))
        print("Val exists loss disf: " + str(val_exists_loss_disf))
        print("Val exists loss no disf: " + str(val_exists_loss_no_disf))
        print("Val type loss disf: " + str(val_type_loss_disf))

        print("Val exists acc: " + str(val_disfluency_exists_accuracy))
        print("Val disfluency type acc: " + str(val_disfluency_type_accuracy))

        wandb.log(
            {
                "train_bound_loss_disf_" + MODEL_NAME: train_bound_loss_disf,
                "train_exists_loss_disf_" + MODEL_NAME: train_exists_loss_disf,
                "train_type_loss_disf_" + MODEL_NAME: train_type_loss_disf,
                "train_exists_loss_no_disf_" + MODEL_NAME: train_exists_loss_no_disf,
                "train_exists_acc_" + MODEL_NAME: train_disfluency_exists_accuracy,
                "train_type_acc_" + MODEL_NAME: train_disfluency_type_accuracy,
                "val_bound_loss_disf_" + MODEL_NAME: val_bound_loss_disf,
                "val_exists_loss_disf_" + MODEL_NAME: val_exists_loss_disf,
                "val_type_loss_disf_" + MODEL_NAME: val_type_loss_disf,
                "val_exists_loss_no_disf_" + MODEL_NAME: val_exists_loss_no_disf,
                "val_exists_acc_" + MODEL_NAME: val_disfluency_exists_accuracy,
                "val_type_acc_" + MODEL_NAME: val_disfluency_type_accuracy,
            }
        )

        gc.collect()

        
#
    def train(self, train_loader, val_loader, decoder):
        for epoch in range(self.epochs):
            print("EPOCH: " + str(epoch)) 
            # decoder train
            decoder.train()
            for batch in tqdm(train_loader):
                if None in batch:
                    continue  #skip
                
                text, text_length, spec, spec_length, label_matrix, num_regions, region, start_region, end_region = batch

                torch.cuda.empty_cache()

                self.steps += 1

                text = text.to(device)
                text_length = text_length.to(device)
                spec = spec.to(device)
                spec_length = spec_length.to(device)
                label_matrix = label_matrix.to(device)

                soft_attention = self.get_soft_attention(
                    self.hps, self.net_g, text, text_length, spec, spec_length
                )  # dims - batch_size, speech_temporal_dim, text_temporal_dim/feature_dim
                # t2 = time.time()
                # print("get soft alignment: {}".format(t2-t1))

                # max feature dim size of 1024
                soft_attention = nn.functional.pad(
                    soft_attention,
                    (
                        0,
                        768 - soft_attention.shape[-1],
                        0,
                        1024 - soft_attention.shape[-2],
                    ),
                )

                label_matrix = nn.functional.pad(
                    label_matrix, 
                    (
                        0,
                        0,
                        0,
                        64 - label_matrix.shape[-2],
                    ),
                )

                # seq length
                mask = torch.ones((label_matrix.shape[0], 64), dtype=torch.bool)
                for i in range(label_matrix.shape[0]):
                    mask[i, : num_regions[i] + 1] = False

                output = decoder(soft_attention, mask=mask.to(device))


                # t3 =time.time()
                # print("predict: {}".format(t3-t2))
                # print("output.shape: " + str(output.shape))

                bound_loss = 0
                confidence_loss_no_disf = 0
                confidence_loss_disf = 0
                type_loss = 0
                ####
                for i in range(label_matrix.shape[0]):
                    no_disf_regions = torch.cat(
                        (
                            label_matrix[i][: start_region[i]],
                            label_matrix[i][end_region[i] + 1 : num_regions[i] + 1],
                        ),
                        dim=-2,
                    )
                    no_disf_output = torch.cat(
                        (
                            output[i][: start_region[i]],
                            output[i][end_region[i] + 1 : num_regions[i] + 1],
                        ),
                        dim=-2,
                    )

                    disf_regions = label_matrix[i][start_region[i] : end_region[i] + 1]
                    disf_output = output[i][start_region[i] : end_region[i] + 1]

                    main_region = label_matrix[i][region[i]]
                    main_output = output[i][region[i]]

                    # loss computations
                    bound_difference_target = main_region[1] - main_region[0]
                    bound_difference_predicted = main_output[1] - main_output[0]
                    bound_loss += self.bound_loss(main_output[:2], main_region[:2]) + self.bound_loss(bound_difference_predicted, bound_difference_target)

                    confidence_loss_no_disf += self.disfluency_exists_loss(
                        no_disf_output[:, 2], no_disf_regions[:, 2]
                    )

                    confidence_loss_disf += self.disfluency_exists_loss(
                        disf_output[:, 2], disf_regions[:, 2]
                    )
                    
                    # print("disf_output: " + str(disf_output[:, 3:].shape))
                    # print("disf_regions:  " + str(disf_regions[:, 3:].shape))
                    type_loss += self.disfluency_type_loss(
                        disf_output[:, 3:], disf_regions[:, 3:]
                    )

                # meaning based on batch size
                bound_loss /= label_matrix.shape[0]
                confidence_loss_no_disf /= label_matrix.shape[0]
                confidence_loss_disf /= label_matrix.shape[0]
                type_loss /= label_matrix.shape[0]

                total_loss = (
                    (self.bound_scale_factor * bound_loss)
                    + confidence_loss_disf
                    + (self.no_disf_scale_factor * confidence_loss_no_disf)
                    + type_loss
                )

                # t4 = time.time()
                # print("loss computation: {}".format(t4-t3))

                self.optimizer.zero_grad()

                total_loss.backward()

                del text
                del soft_attention
                del text_length
                del spec
                del spec_length
                del label_matrix
                del num_regions
                del region
                del start_region
                del end_region

                gc.collect()

                self.optimizer.step()
                # t5 = time.time()
                # print("optimize: {}".format(t5-t4))
                # print("total time: {}".format(t5 - start))

            self.eval_model(train_loader, val_loader, decoder)
            torch.save(decoder, "./saved_models/" + MODEL_NAME)

    def train_and_eval(self):
        torch.manual_seed(0)

        dataset = Dataset(
            "/data/xuanru/VCTK-tts", self.hps, self.downsample_factor
        )

        train_set, val_set = torch.utils.data.random_split(dataset, [0.90, 0.10])
        
        # 
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=8
        )
        # train_loader_metrics = train_loader.clone()
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=8
        )

        self.train(train_loader, val_loader, self.decoder)
        # torch.save(self.decoder, "./saved_models/" + MODEL_NAME)
