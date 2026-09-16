"""
JoImTeR trainer: builds the image/text encoders and optimizers, runs the
joint ITM + MLM training loop (manuscript Sec. 3.1.1), and checkpoints on
best validation image-to-text Recall@1.

Each training step is either an ITM step (sentence + word-region matching
losses plus triplet losses, with in-batch hard-negative mining) or an MLM
step, sampled with probability `cfg.multitasksampling` (0.7 ITM / 0.3 MLM
in the reported run).
"""
import os
import sys
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertConfig

from configs.train_config import cfg
from src.losses import sent_loss, words_loss, sent_triplet_loss, words_triplet_loss
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.utils import mkdir_p

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class JoImTeR(object):

    def __init__(self, output_dir,
                 data_loader_itm, data_loader_mlm,
                 dataloader_val_itm, dataloader_val_mlm):

        if cfg.TRAIN:
            self.model_dir = os.path.join(output_dir, 'Model')
            mkdir_p(self.model_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size        = data_loader_itm.batch_size
        self.val_batch_size    = dataloader_val_itm.batch_size
        self.max_epoch         = cfg.epochs
        self.snapshot_interval = cfg.snapshot_interval

        self.data_loader_itm    = data_loader_itm
        self.data_loader_mlm    = data_loader_mlm
        self.dataloader_val_itm = dataloader_val_itm
        self.dataloader_val_mlm = dataloader_val_mlm

        self.log_file = os.path.join(output_dir, 'err.log')
        with open(self.log_file, 'w') as f:
            f.write('Epoch 0:\n\n')

        self.num_batches     = len(self.data_loader_itm)
        self.num_batches_mlm = len(self.data_loader_mlm)

        self.bert_config = BertConfig(
            vocab_size                   = data_loader_itm.dataset.vocab_size,
            hidden_size                  = cfg.hidden_dim,
            num_hidden_layers            = 3,
            num_attention_heads          = 8,
            intermediate_size            = 2048,
            hidden_act                   = 'gelu',
            hidden_dropout_prob          = cfg.hidden_dropout_prob,
            attention_probs_dropout_prob = cfg.attention_probs_dropout_prob,
            max_position_embeddings      = 512,
            layer_norm_eps               = 1e-12,
            initializer_range            = 0.02,
            type_vocab_size              = 2,
            pad_token_id                 = 0
        )

    def build_models(self):
        image_encoder = ImageEncoder(
            output_channels = cfg.hidden_dim,
            pretrained      = cfg.pretrained
        )

        # ── freeze early backbone layers to prevent overfitting ────────
        if cfg.freeze_backbone:
            for name, param in image_encoder.named_parameters():
                if name.startswith('stem') or \
                   name.startswith('layer1') or \
                   name.startswith('layer2'):
                    param.requires_grad = False

        text_encoder = TextEncoder(
            bert_config     = self.bert_config,
            output_channels = cfg.hidden_dim,
            pool            = 'cls'
        )
        for p in text_encoder.parameters():
            p.requires_grad = True

        if cfg.CUDA:
            image_encoder = image_encoder.cuda()
            text_encoder  = text_encoder.cuda()

        # ── split LR: pretrained backbone vs new layers ────────────────
        backbone_params = []
        new_params      = []
        for name, param in image_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('layer3') or name.startswith('layer4'):
                backbone_params.append(param)
            else:
                new_params.append(param)

        optimizerI = torch.optim.AdamW([
            {'params': backbone_params, 'lr': cfg.lr_backbone},
            {'params': new_params,      'lr': cfg.lr}
        ], weight_decay=cfg.weight_decay)

        optimizerT = torch.optim.AdamW(
            text_encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        epoch = 0
        if cfg.text_encoder_path != '':
            print("Loading checkpoint...")
           # epoch = int(
         #       re.findall(r'\d+', os.path.basename(cfg.text_encoder_path))[-1]
          #  ) + 1
         #   print(f'Resuming from epoch {epoch}')

        # ── cosine annealing — no sudden LR cliff ─────────────────────
        lr_schedulerI = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerI, T_max=cfg.epochs, eta_min=1e-7
        )
        lr_schedulerT = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerT, T_max=cfg.epochs, eta_min=1e-7
        )

        if cfg.text_encoder_path != '':
            img_path = cfg.text_encoder_path.replace('text_encoder', 'image_encoder')
            state = torch.load(img_path, map_location='cpu')
            image_encoder.load_state_dict(state['model'])
            optimizerI.load_state_dict(state['optimizer'])
            if cfg.scheduler_init:
                lr_schedulerI.load_state_dict(state['lr_scheduler'])

            state = torch.load(cfg.text_encoder_path, map_location='cpu')
            text_encoder.load_state_dict(state['model'])
            optimizerT.load_state_dict(state['optimizer'])
            if cfg.scheduler_init:
                lr_schedulerT.load_state_dict(state['lr_scheduler'])
                epoch = state['epoch'] + 1
                print(f"Resuming from epoch {epoch}")

        return [text_encoder, image_encoder, epoch,
                optimizerI, optimizerT, lr_schedulerI, lr_schedulerT]

    def save_model(self, image_encoder, text_encoder,
                   optimizerI, optimizerT, lr_schedulerI, lr_schedulerT,
                   epoch, tag='latest'):
        torch.save({
            'model':        image_encoder.state_dict(),
            'optimizer':    optimizerI.state_dict(),
            'lr_scheduler': lr_schedulerI.state_dict(),
            'epoch':        epoch,
        }, f'{self.model_dir}/image_encoder_{tag}.pth')

        torch.save({
            'model':        text_encoder.state_dict(),
            'optimizer':    optimizerT.state_dict(),
            'lr_scheduler': lr_schedulerT.state_dict(),
            'epoch':        epoch,
        }, f'{self.model_dir}/text_encoder_{tag}.pth')

        print(f'Saved [{tag}] checkpoint at epoch {epoch}')

    def train(self):
        tb_dir = os.path.join(self.model_dir, '..', 'tensorboard')
        mkdir_p(tb_dir)
        tbw = SummaryWriter(log_dir=tb_dir)

        (text_encoder, image_encoder, start_epoch,
         optimizerI, optimizerT,
         lr_schedulerI, lr_schedulerT) = self.build_models()

        text_encoder.train()
        image_encoder.train()

        mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if cfg.CUDA:
            mlm_loss_fn = mlm_loss_fn.cuda()

        data_iter_mlm  = iter(self.data_loader_mlm)
        total_mlm_loss = 0
        step_mlm       = 0
        epoch_mlm      = start_epoch

        for epoch in range(start_epoch, self.max_epoch):

            text_encoder.train()
            image_encoder.train()

            s_total_loss0 = s_total_loss1 = 0
            w_total_loss0 = w_total_loss1 = 0
            total_damsm_loss = 0
            s_t_total_loss0 = s_t_total_loss1 = 0
            w_t_total_loss0 = w_t_total_loss1 = 0
            total_t_loss = 0

            print(f'[epoch {epoch}] '
                  f'lr_i_backbone={optimizerI.param_groups[0]["lr"]:.4e}  '
                  f'lr_i_new={optimizerI.param_groups[1]["lr"]:.4e}  '
                  f'lr_t={optimizerT.param_groups[0]["lr"]:.4e}')

            data_iter_itm = iter(self.data_loader_itm)
            step = 0
            pbar = tqdm(range(self.num_batches), ncols=120)

            while step < self.num_batches:

                if cfg.scheduler_step:
                    lr_schedulerI.step()
                    lr_schedulerT.step()

                if np.random.uniform() < cfg.multitasksampling:

                    # ── ITM task ──────────────────────────────────
                    imgs, captions, masks, class_ids, cap_lens = next(data_iter_itm)

                    class_ids  = class_ids.numpy()
                    batch_size = imgs.shape[0]

                    labels = Variable(torch.LongTensor(range(batch_size)))

                    if cfg.CUDA:
                        imgs, captions, masks, cap_lens = (
                            imgs.cuda(), captions.cuda(),
                            masks.cuda(), cap_lens.cuda()
                        )
                        labels = labels.cuda()

                    image_encoder.zero_grad()
                    text_encoder.zero_grad()

                    words_features, sent_code, _, _, _, _ = image_encoder(imgs)
                    words_embs, sent_emb = text_encoder(captions, masks, task='itm')

                    # ── hard negative mining ─────────────────────
                    with torch.no_grad():
                        img_norm_b = torch.nn.functional.normalize(sent_code, dim=-1)
                        txt_norm_b = torch.nn.functional.normalize(sent_emb,  dim=-1)
                        batch_sim  = torch.mm(img_norm_b, txt_norm_b.t())
                        batch_sim.fill_diagonal_(-1e4)
                        neg_ids = batch_sim.argmax(dim=1)
                        if cfg.CUDA:
                            neg_ids = neg_ids.cuda()

                    s_loss0, s_loss1 = sent_loss(
                        sent_code, sent_emb, labels, class_ids, batch_size
                    )
                    s_total_loss0 += s_loss0.item()
                    s_total_loss1 += s_loss1.item()
                    damsm_loss = s_loss0 + s_loss1

                    w_loss0, w_loss1, _ = words_loss(
                        words_features, words_embs[:, :, 1:],
                        labels, cap_lens - 1, class_ids, batch_size
                    )
                    w_total_loss0    += w_loss0.item()
                    w_total_loss1    += w_loss1.item()
                    damsm_loss       += w_loss0 + w_loss1
                    total_damsm_loss += damsm_loss.item()

                    s_t_loss0, s_t_loss1 = sent_triplet_loss(
                        sent_code, sent_emb, labels, neg_ids, batch_size
                    )
                    s_t_total_loss0 += s_t_loss0.item()
                    s_t_total_loss1 += s_t_loss1.item()
                    t_loss = s_t_loss0 + s_t_loss1

                    w_t_loss0, w_t_loss1, _ = words_triplet_loss(
                        words_features, words_embs[:, :, 1:],
                        labels, neg_ids, cap_lens - 1, batch_size
                    )
                    w_t_total_loss0 += w_t_loss0.item()
                    w_t_total_loss1 += w_t_loss1.item()
                    t_loss       += w_t_loss0 + w_t_loss1
                    total_t_loss += t_loss.item()

                    combo_loss = (cfg.LAMBDA_DAMSM * damsm_loss +
                                  cfg.LAMBDA_TRIPLET * t_loss)
                    combo_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        image_encoder.parameters(), cfg.clip_max_norm)
                    optimizerI.step()
                    torch.nn.utils.clip_grad_norm_(
                        text_encoder.parameters(), cfg.clip_max_norm)
                    optimizerT.step()

                    global_step = step + epoch * self.num_batches
                    tbw.add_scalar('Train/w_loss0',    w_loss0.item(),    global_step)
                    tbw.add_scalar('Train/w_loss1',    w_loss1.item(),    global_step)
                    tbw.add_scalar('Train/s_loss0',    s_loss0.item(),    global_step)
                    tbw.add_scalar('Train/s_loss1',    s_loss1.item(),    global_step)
                    tbw.add_scalar('Train/damsm_loss', damsm_loss.item(), global_step)
                    tbw.add_scalar('Train/w_t_loss0',  w_t_loss0.item(),  global_step)
                    tbw.add_scalar('Train/w_t_loss1',  w_t_loss1.item(),  global_step)
                    tbw.add_scalar('Train/s_t_loss0',  s_t_loss0.item(),  global_step)
                    tbw.add_scalar('Train/s_t_loss1',  s_t_loss1.item(),  global_step)
                    tbw.add_scalar('Train/t_loss',     t_loss.item(),     global_step)
                    tbw.add_scalar('LR/lr_backbone',
                                   optimizerI.param_groups[0]['lr'], global_step)
                    tbw.add_scalar('LR/lr_new',
                                   optimizerI.param_groups[1]['lr'], global_step)

                    pbar.set_description(
                        'w:%.3f s:%.3f wt:%.3f st:%.3f mlm:%.3f' % (
                            (w_total_loss0 + w_total_loss1) / (step + 1),
                            (s_total_loss0 + s_total_loss1) / (step + 1),
                            (w_t_total_loss0 + w_t_total_loss1) / (step + 1),
                            (s_t_total_loss0 + s_t_total_loss1) / (step + 1),
                            total_mlm_loss / (step_mlm + 1)
                        )
                    )
                    pbar.update(1)
                    step += 1

                else:
                    # ── MLM task ──────────────────────────────────
                    try:
                        captions, masks, labels, class_ids, cap_lens = next(data_iter_mlm)
                    except StopIteration:
                        data_iter_mlm  = iter(self.data_loader_mlm)
                        total_mlm_loss = 0
                        step_mlm       = 0
                        epoch_mlm     += 1
                        captions, masks, labels, class_ids, cap_lens = next(data_iter_mlm)

                    class_ids = class_ids.numpy()

                    if cfg.CUDA:
                        captions, masks, labels, cap_lens = (
                            captions.cuda(), masks.cuda(),
                            labels.cuda(), cap_lens.cuda()
                        )

                    text_encoder.zero_grad()
                    preds    = text_encoder(captions, masks, task='mlm')
                    mlm_loss = mlm_loss_fn(preds, labels)
                    total_mlm_loss += mlm_loss.item()

                    mlm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        text_encoder.parameters(), cfg.clip_max_norm)
                    optimizerT.step()

                    tbw.add_scalar(
                        'MLM/mlm_loss', mlm_loss.item(),
                        step_mlm + epoch_mlm * self.num_batches_mlm
                    )
                    step_mlm += 1

            pbar.close()

            # ── end of epoch: val losses ───────────────────────────────
            v_s, v_w, v_s_t, v_w_t = self.evaluate_itm(image_encoder, text_encoder)
            print(f'[epoch {epoch}] val — '
                  f'w:{v_w:.4f} s:{v_s:.4f} wt:{v_w_t:.4f} st:{v_s_t:.4f}')
            tbw.add_scalar('Val/w_loss',   v_w,   epoch)
            tbw.add_scalar('Val/s_loss',   v_s,   epoch)
            tbw.add_scalar('Val/w_t_loss', v_w_t, epoch)
            tbw.add_scalar('Val/s_t_loss', v_s_t, epoch)

            v_mlm, acc = self.evaluate_mlm(text_encoder)
            print(f'[epoch {epoch}] val — mlm:{v_mlm:.4f}  acc:{acc:.4f}')
            tbw.add_scalar('Val/mlm_loss', v_mlm, epoch)
            tbw.add_scalar('Val/mlm_acc',  acc,   epoch)

            # ── val retrieval R@1 — used for checkpoint selection ──────
            val_img_embs = []
            val_txt_embs = []
            val_img_ids  = []

            image_encoder.eval()
            text_encoder.eval()

            with torch.no_grad():
                for real_imgs, captions, masks, image_ids, cap_lens in \
                        self.dataloader_val_itm:
                    if cfg.CUDA:
                        real_imgs = real_imgs.cuda()
                        captions  = captions.cuda()
                        masks     = masks.cuda()
                    _, img_emb, _, _, _, _ = image_encoder(real_imgs)
                    _, txt_emb             = text_encoder(captions, masks, task='itm')
                    val_img_embs.append(img_emb.cpu())
                    val_txt_embs.append(txt_emb.cpu())
                    val_img_ids.extend(image_ids.tolist())

            val_img_embs = torch.cat(val_img_embs, dim=0)
            val_txt_embs = torch.cat(val_txt_embs, dim=0)
            val_img_ids  = np.array(val_img_ids)

            img_norm = torch.nn.functional.normalize(val_img_embs, dim=1)
            txt_norm = torch.nn.functional.normalize(val_txt_embs, dim=1)
            sim_val  = torch.mm(img_norm, txt_norm.t()).numpy()

            correct_r1 = 0
            for i in range(len(val_img_ids)):
                scores    = sim_val[i].copy()
                scores[i] = -999
                top1_idx  = np.argmax(scores)
                if val_img_ids[top1_idx] == val_img_ids[i]:
                    correct_r1 += 1
            val_r1 = correct_r1 / len(val_img_ids) * 100

            print(f'[epoch {epoch}] val R@1: {val_r1:.2f}%')
            tbw.add_scalar('Val/R@1', val_r1, epoch)

            image_encoder.train()
            text_encoder.train()

            with open(self.log_file, 'a+') as f:
                f.write(f'\nEpoch {epoch + 1}:\n\n')

            if not cfg.scheduler_step:
                lr_schedulerI.step()
                lr_schedulerT.step()

            # ── checkpoint by val R@1, not val loss ───────────────────
            if not hasattr(self, 'best_val_r1'):
                self.best_val_r1 = 0.0

            if val_r1 > self.best_val_r1:
                self.best_val_r1 = val_r1
                self.save_model(image_encoder, text_encoder,
                                optimizerI, optimizerT,
                                lr_schedulerI, lr_schedulerT, epoch, tag='best')
                self.save_model(image_encoder, text_encoder,
                                optimizerI, optimizerT,
                                lr_schedulerI, lr_schedulerT, epoch, tag='latest')
                print(f'  → new best val R@1: {val_r1:.2f}%')
            else:
                self.save_model(image_encoder, text_encoder,
                                optimizerI, optimizerT,
                                lr_schedulerI, lr_schedulerT, epoch, tag='latest')

            torch.cuda.empty_cache()
            import gc
            gc.collect()

    @torch.no_grad()
    def evaluate_itm(self, cnn_model, trx_model):
        cnn_model.eval()
        trx_model.eval()

        s_total = w_total = s_t_total = w_t_total = 0

        for real_imgs, captions, masks, class_ids, cap_lens in tqdm(
                self.dataloader_val_itm, leave=False, ncols=80, desc='val_itm'):

            class_ids  = class_ids.numpy()
            batch_size = real_imgs.shape[0]
            ids        = np.array(list(range(batch_size)))
            neg_ids    = Variable(torch.LongTensor(
                [np.random.choice(ids[ids != x]) for x in ids]
            ))
            labels = Variable(torch.LongTensor(range(batch_size)))

            if cfg.CUDA:
                real_imgs, captions, masks, cap_lens = (
                    real_imgs.cuda(), captions.cuda(),
                    masks.cuda(), cap_lens.cuda()
                )
                labels  = labels.cuda()
                neg_ids = neg_ids.cuda()

            words_features, sent_code, _, _, _, _ = cnn_model(real_imgs)
            words_embs, sent_emb = trx_model(captions, masks, task='itm')

            w0, w1, _ = words_loss(
                words_features, words_embs[:, :, 1:],
                labels, cap_lens - 1, class_ids, batch_size
            )
            w_total += (w0 + w1).item()

            s0, s1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            s_total += (s0 + s1).item()

            wt0, wt1, _ = words_triplet_loss(
                words_features, words_embs[:, :, 1:],
                labels, neg_ids, cap_lens - 1, batch_size
            )
            w_t_total += (wt0 + wt1).item()

            st0, st1 = sent_triplet_loss(
                sent_code, sent_emb, labels, neg_ids, batch_size)
            s_t_total += (st0 + st1).item()

        n = len(self.dataloader_val_itm)
        return s_total / n, w_total / n, s_t_total / n, w_t_total / n

    @torch.no_grad()
    def evaluate_mlm(self, trx_model):
        trx_model.eval()
        total_loss    = 0
        total_correct = 0
        total_masked  = 0

        mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

        for captions, masks, labels, class_ids, cap_lens in tqdm(
                self.dataloader_val_mlm, leave=False, ncols=80, desc='val_mlm'):

            if cfg.CUDA:
                captions, masks, labels, cap_lens = (
                    captions.cuda(), masks.cuda(),
                    labels.cuda(), cap_lens.cuda()
                )

            preds    = trx_model(captions, masks, task='mlm')
            mlm_loss = mlm_loss_fn(preds, labels)
            total_loss += mlm_loss.item()

            preds_np  = preds.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            masked    = labels_np != -1
            total_correct += np.sum(preds_np[masked] == labels_np[masked])
            total_masked  += np.sum(masked)

        n   = len(self.dataloader_val_mlm)
        acc = total_correct / total_masked if total_masked > 0 else 0.0
        return total_loss / n, acc

