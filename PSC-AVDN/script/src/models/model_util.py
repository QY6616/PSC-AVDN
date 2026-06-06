import collections
import copy
import json
import logging
import os
from importlib import import_module
import numpy as np
import torch
from torch.nn import functional as F


def adjust_lr(args, epoch, schedulers):
    if epoch >= args.lr["warmup_epoch"]:
        schedulers["base"].step()
    else:
        schedulers["warmup"].step()


def create_optimizer_and_schedulers(first_epoch, args, parameters, optimizer=None):
    init_lr = args.lr["init"] * args.lr["warmup_scale"]
    if args.lr["warmup_scale"] != 1:
        assert args.lr["warmup_epoch"] > 0
    if optimizer is None:
        assert args.optimizer in ("adam", "adamw")
        OptimizerClass = (
            torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
        )
        optimizer = OptimizerClass(
            parameters, lr=init_lr, weight_decay=args.weight_decay
        )
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = init_lr
    assert args.lr["profile"] in ("linear", "cosine", "triangular", "triangular2")
    if args.lr["profile"] == "linear":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, gamma=args.lr["decay_scale"], step_size=args.lr["decay_epoch"]
        )
    elif args.lr["profile"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(args.epochs - args.lr["warmup_epoch"] - 1),
            eta_min=args.lr["final"],
        )
    else:
        assert min(args.lr["cycle_epoch_up"], args.lr["cycle_epoch_down"]) > 0
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr["init"],
            max_lr=args.lr["final"],
            step_size_up=args.lr["cycle_epoch_up"],
            step_size_down=args.lr["cycle_epoch_down"],
            mode=args.lr["profile"],
            cycle_momentum=False,
        )
    warmup_scheduler = None
    if args.lr["warmup_epoch"]:
        warmup_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=(1 / args.lr["warmup_scale"] ** (1 / args.lr["warmup_epoch"])),
        )
    for epoch in range(first_epoch):
        if epoch >= args.lr["warmup_epoch"]:
            lr_scheduler.step()
        else:
            warmup_scheduler.step()
    return optimizer, {"base": lr_scheduler, "warmup": warmup_scheduler}


def load_model(fsave, device, check_epoch=None, for_inference=False):
    logger.info("Loading from {} to {}".format(fsave, device))
    save = torch.load(fsave, map_location=device)
    LearnedModel = import_module("alfred.model.learned").LearnedModel
    save["args"]["model_dir"] = os.path.dirname(fsave)
    model = LearnedModel(
        save["args"], save["embs_ann"], save["vocab_out"], for_inference
    )
    model.load_state_dict(save["model"])
    OptimizerClass = (
        torch.optim.Adam if save["args"].optimizer == "adam" else torch.optim.AdamW
    )
    optimizer = OptimizerClass(
        model.parameters(), lr=1e-3, weight_decay=save["args"].weight_decay
    )
    optimizer.load_state_dict(save["optim"])
    if check_epoch:
        assert (
            save["metric"]["epoch"] == check_epoch
        ), "Epochs in info.json and latest.pth do not match"
    model = model.to(torch.device(device))
    optimizer_to(optimizer, torch.device(device))
    return model, optimizer


def load_model_args(fsave):
    save = torch.load(fsave, map_location=lambda storage, loc: storage)
    return save["args"]


def save_model(model, model_name, stats, optimizer=None, symlink=False):
    save_path = os.path.join(model.args.dout, model_name)
    if not symlink:
        state_dict = {
            key.replace("model.module.", "model."): value
            for key, value in model.state_dict().items()
        }
        assert optimizer is not None
        torch.save(
            {
                "metric": stats,
                "model": state_dict,
                "optim": optimizer.state_dict(),
                "args": model.args,
                "vocab_out": model.vocab_out,
                "embs_ann": model.embs_ann,
            },
            save_path,
        )
    else:
        model_path = os.path.join(
            model.args.dout, "model_{:02d}.pth".format(stats["epoch"])
        )
        if os.path.islink(save_path):
            os.unlink(save_path)
        os.symlink(model_path, save_path)


def tensorboard(writer, metrics, split, iter, frequency, batch_size):
    if (iter // batch_size) % frequency == 0:
        for metric_name, metric_value_list in metrics.items():
            metric_value = np.mean(metric_value_list[-frequency:])
            writer.add_scalar("{}/{}".format(split, metric_name), metric_value, iter)


def save_log(dout, progress, total, stage, **kwargs):
    info_path = os.path.join(dout, "info.json")
    info_dicts = []
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info_dicts = json.load(f)
    info_dict = {"stage": stage, "progress": progress, "total": total}
    info_dict.update(kwargs)
    info_dicts.append(info_dict)
    with open(info_path, "w") as f:
        json.dump(info_dicts, f)


def load_log(dout, stage):
    info_path = os.path.join(dout, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info_dicts = json.load(f)
        info_dict = [el for el in info_dicts if el["stage"] == stage][-1]
    else:
        info_dict = {"progress": 0, "best_loss": {}, "iters": {}}
    if isinstance(info_dict["best_loss"], dict):
        info_dict["best_loss"] = collections.defaultdict(
            lambda: 1e10, info_dict["best_loss"]
        )
    if isinstance(info_dict["iters"], dict):
        info_dict["iters"] = collections.defaultdict(lambda: 0, info_dict["iters"])
    return info_dict


def update_log(dout, stage, update, **kwargs):
    assert update in ("increase", "rewrite")
    info_path = os.path.join(dout, "info.json")
    assert os.path.exists(info_path)
    with open(info_path) as f:
        info_dicts = json.load(f)
    info_dict = copy.deepcopy([el for el in info_dicts if el["stage"] == stage][-1])
    for key, value in kwargs.items():
        assert key in info_dict
        new_value = value + info_dict[key] if update == "increase" else value
        info_dict[key] = new_value
    if info_dicts[-1]["stage"] == stage:
        info_dicts[-1] = info_dict
    else:
        info_dicts.append(info_dict)
    with open(info_path, "w") as f:
        json.dump(info_dicts, f)


def triangular_mask(size, device, diagonal_shift=1):
    square = torch.triu(torch.ones(size, size, device=device), diagonal=diagonal_shift)
    square = square.masked_fill(square == 1.0, float("-inf"))
    return square


def generate_attention_mask(len_lang, len_frames, device, num_input_actions=0):
    lang_to_lang = torch.zeros((len_lang, len_lang), device=device).float()
    lang_to_rest = torch.ones(
        (len_lang, len_frames * 2), device=device
    ).float() * float("-inf")
    lang_to_all = torch.cat((lang_to_lang, lang_to_rest), dim=1)
    frames_to_lang = torch.zeros((len_frames, len_lang), device=device).float()
    frames_to_frames = triangular_mask(len_frames, device)
    frames_to_directions = triangular_mask(len_frames, device)
    frames_to_all = torch.cat(
        (frames_to_lang, frames_to_frames, frames_to_directions), dim=1
    )
    actions_to_all = frames_to_all.clone()
    all_to_all = torch.cat((lang_to_all, frames_to_all, actions_to_all), dim=0)
    return all_to_all


def process_prediction(
    action, objects, pad, vocab_action, clean_special_tokens, predict_object=True
):
    if pad in action:
        pad_start_idx = action.index(pad)
        action = action[:pad_start_idx]
        objects = objects[:pad_start_idx]
    if clean_special_tokens:
        stop_token = vocab_action.word2index("Stop")
        if stop_token in action:
            stop_start_idx = action.index(stop_token)
            action = action[:stop_start_idx]
            objects = objects[:stop_start_idx]
    words = vocab_action.index2word(action)
    if predict_object:
        pred_object = objects[None].max(2)[1].cpu().numpy()
    else:
        pred_object = None
    pred_processed = {
        "action": " ".join(words),
        "object": pred_object,
    }
    return pred_processed


def extract_action_preds_list(
    m_out_list, pad, vocab_action, clean_special_tokens=True, lang_only=False
):
    m_out_action = (
        m_out_list[0]["action"] + m_out_list[1]["action"] + m_out_list[2]["action"]
    )
    m_out_object = (
        m_out_list[0]["object"] + m_out_list[1]["object"] + m_out_list[2]["object"]
    )
    zipped_data = zip(m_out_action.max(2)[1].tolist(), m_out_object)
    predict_object = not lang_only
    preds_list = [
        process_prediction(
            action, objects, pad, vocab_action, clean_special_tokens, predict_object
        )
        for action, objects in zipped_data
    ]
    m_out = {}
    m_out["action"] = m_out_action
    m_out["object"] = m_out_object
    return preds_list, m_out


def extract_action_preds(
    model_out, pad, vocab_action, clean_special_tokens=True, lang_only=False
):
    zipped_data = zip(model_out["action"].max(2)[1].tolist(), model_out["object"])
    predict_object = not lang_only
    preds_list = [
        process_prediction(
            action, objects, pad, vocab_action, clean_special_tokens, predict_object
        )
        for action, objects in zipped_data
    ]
    return preds_list


def compute_obj_class_precision(
    metrics, gt_dict, classes_out, compute_train_loss_over_history
):
    if len(gt_dict["object"]) > 0:
        if compute_train_loss_over_history:
            interact_idxs = torch.nonzero(gt_dict["obj_interaction_action"])
        else:
            interact_idxs = torch.nonzero(
                gt_dict["driver_actions_pred_mask"] * gt_dict["obj_interaction_action"]
            )
        obj_classes_prob = classes_out[tuple(interact_idxs.T)]
        obj_classes_pred = obj_classes_prob.max(1)[1]
        obj_classes_gt = torch.cat(gt_dict["object"], dim=0)
        precision = torch.sum(obj_classes_pred == obj_classes_gt) / len(obj_classes_gt)
        metrics["action/object"].append(precision.item())
    else:
        metrics["action/object"].append(0.0)


def obj_classes_loss(pred_obj_cls, gt_obj_cls, interact_idxs):
    pred_obj_cls_inter = pred_obj_cls[interact_idxs]
    assert not (gt_obj_cls == 0).any()
    obj_cls_loss = F.cross_entropy(pred_obj_cls_inter, gt_obj_cls, reduction="mean")
    return obj_cls_loss


def tokens_to_lang(tokens, vocab, skip_tokens=None, join=True):
    if skip_tokens is None:
        skip_tokens = {}

    def _tokens_to_lang(seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        lang = [vocab.index2word(t) for t in seq if t not in skip_tokens]
        lang = " ".join(lang) if join else lang
        return lang

    if isinstance(tokens[0], int):
        output = _tokens_to_lang(tokens)
    else:
        output = [_tokens_to_lang(seq) for seq in tokens]
    return output


def translate_to_vocab(tokens, vocab, vocab_translate, skip_new_tokens=False):
    if vocab_translate.contains_same_content(vocab):
        return tokens
    lang_orig = tokens_to_lang(tokens, vocab, join=False)
    tokens_new = []
    for word in lang_orig:
        if skip_new_tokens and word not in vocab_translate.counts:
            word = "<<pad>>"
        tokens_new.append(vocab_translate.word2index(word))
    if not skip_new_tokens:
        lang_new = tokens_to_lang(tokens_new, vocab_translate, join=False)
        assert lang_orig == lang_new
    return tokens_new


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
