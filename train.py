import torch
import os
import logging
import random
import argparse
import json
import glob

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset import LLVCDataset as Dataset
from model import Net
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
import fairseq

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
# check if port is available


def net_g_step(
    batch, net_g, device, fp16_run
):
    og, gt = batch
    og = og.to(device=device, non_blocking=True)
    gt = gt.to(device=device, non_blocking=True)

    with autocast(enabled=fp16_run):
        output = net_g(og)
    return output, gt, og


def training_runner(
    rank,
    world_size,
    config,
    training_dir,
):
    log_dir = os.path.join(training_dir, "logs")
    checkpoint_dir = os.path.join(training_dir, "checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_multi_process = world_size > 1
    is_main_process = rank == 0

    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    dist.init_process_group(
        backend="gloo", init_method="env://", rank=rank, world_size=world_size
    )

    if is_multi_process:
        torch.cuda.set_device(rank)

    torch.manual_seed(config['seed'])

    data_train = Dataset(
        **config['data'], dset='train')
    data_val = Dataset(
        **config['data'], dset='val')
    data_dev = Dataset(
        **config['data'], dset='dev')
    for ds in [data_train, data_val, data_dev]:
        logging.info(
            f"Loaded dataset at {ds.dset} containing {len(ds)} elements")

    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=config['batch_size'],
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=config['eval_batch_size'])
    dev_loader = torch.utils.data.DataLoader(data_dev,
                                             batch_size=config['eval_batch_size'])

    net_g = Net(**config['model_params'])
    logging.info(f"Model size: {utils.model_size(net_g)}M params")

    if is_multi_process:
        net_g = net_g.cuda(rank)
    else:
        net_g = net_g.to(device=device)

    if config['discriminator'] == 'hfg':
        from hfg_disc import ComboDisc, discriminator_loss, generator_loss, feature_loss
        net_d = ComboDisc()
    else:
        from discriminators import MultiPeriodDiscriminator, discriminator_loss, generator_loss, feature_loss
        net_d = MultiPeriodDiscriminator(periods=config['periods'])

    if is_multi_process:
        net_d = net_d.cuda(rank)
    else:
        net_d = net_d.to(device=device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        config['optim']['lr'],
        betas=config['optim']['betas'],
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay']
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        config['optim']['lr'],
        betas=config['optim']['betas'],
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay']
    )

    if is_multi_process:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    last_d_state = utils.latest_checkpoint_path(checkpoint_dir, "D_*.pth")
    last_g_state = utils.latest_checkpoint_path(checkpoint_dir, "G_*.pth")

    if last_d_state and last_g_state:
        net_d, optim_d, lr, epoch, step = utils.load_checkpoint(
            last_d_state, net_d, optim_d)
        net_g, optim_g, lr, epoch, step = utils.load_checkpoint(
            last_g_state, net_g, optim_g)
        global_step = step
        logging.info("Loaded generator checkpoint from %s" % last_g_state)
        logging.info("Loaded discriminator checkpoint from %s" % last_d_state)
        logging.info("Generator learning rates restored to:" +
                     utils.format_lr_info(optim_g))
        logging.info("Discriminator learning rates restored to:" +
                     utils.format_lr_info(optim_d))
    else:
        lr = config['optim']['lr']
        global_step = 0
        epoch = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config['lr_sched']['lr_decay']
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config['lr_sched']['lr_decay']
    )

    scaler = GradScaler(enabled=config['fp16_run'])

    # load fairseq model
    if config['aux_fairseq']['c'] > 0:
        cp_path = config['aux_fairseq']['checkpoint_path']
        fairseq_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([
            cp_path])
        fairseq_model = fairseq_model[0]
        # move model to GPU
        fairseq_model.eval().to(device)
    else:
        fairseq_model = None

    cache = []
    loss_mel_avg = utils.RunningAvg()
    loss_fairseq_avg = utils.RunningAvg()
    for epoch in range(epoch, 10000):
        # train_loader.batch_sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()

        use_cache = len(cache) == len(train_loader)
        data = cache if use_cache else enumerate(train_loader)

        if is_main_process:
            lr = optim_g.param_groups[0]["lr"]

        # count down steps to next checkpoint
        progress_bar = tqdm(range(config['checkpoint_interval']))
        progress_bar.update(global_step % config['checkpoint_interval'])

        for batch_idx, batch in data:
            output, gt, og = net_g_step(
                batch, net_g, device, config['fp16_run'])

            # take random slices of input and output wavs
            if config['segment_size'] < output.shape[-1]:
                start_idx = random.randint(
                    0, output.shape[-1] - config['segment_size'] - 1)
                gt_sliced = gt[:, :, start_idx:start_idx +
                               config['segment_size']]
                output_sliced = output.detach()[:, :,
                                                start_idx:start_idx + config['segment_size']]
            else:
                gt_sliced = gt
                output_sliced = output.detach()

            with autocast(enabled=config['fp16_run']):
                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(
                    output_sliced, gt_sliced)
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )

            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            if config['grad_clip_threshold'] is not None:
                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                    net_d.parameters(), config['grad_clip_threshold'])
            grad_norm_d = utils.clip_grad_value_(
                net_d.parameters(), config['grad_clip_value'])
            scaler.step(optim_d)

            with autocast(enabled=config['fp16_run']):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(gt, output)
                if fairseq_model is not None:
                    loss_fairseq = utils.fairseq_loss(
                        output, gt, fairseq_model) * config['aux_fairseq']['c']
                else:
                    loss_fairseq = torch.tensor(0.0)
                loss_fairseq_avg.update(loss_fairseq)
                with autocast(enabled=False):
                    if config['aux_mel']['c'] > 0:
                        loss_mel = utils.aux_mel_loss(
                            output, gt, config) * config['aux_mel']['c']
                    else:
                        loss_mel = torch.tensor(0.0)
                    loss_mel_avg.update(loss_mel)
                    loss_fm = feature_loss(
                        fmap_r, fmap_g) * config['feature_loss_c']
                    loss_gen, losses_gen = generator_loss(
                        y_d_hat_g)
                    loss_gen = loss_gen * config['disc_loss_c']
                    loss_gen_all = (loss_gen + loss_fm) + loss_mel + \
                        loss_fairseq

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            if config['grad_clip_threshold'] is not None:
                grad_norm_g = torch.nn.utils.clip_grad_norm_(
                    net_g.parameters(), config['grad_clip_threshold'])
            grad_norm_g = utils.clip_grad_value_(
                net_g.parameters(), config['grad_clip_value'])
            scaler.step(optim_g)
            scaler.update()

            global_step += 1
            progress_bar.update(1)

            if is_main_process and global_step > 0 and (global_step % config['log_interval'] == 0):
                lr = optim_g.param_groups[0]["lr"]
                # Amor For Tensorboard display
                if loss_mel > 50:
                    loss_mel = 50

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                    }
                )

                if config['aux_mel']['c'] > 0:
                    scalar_dict.update({"train_metrics/mel": loss_mel_avg()})
                    loss_mel_avg.reset()

                if fairseq_model is not None:
                    scalar_dict.update(
                        {
                            "loss/g/fairseq": loss_fairseq,
                        }
                    )
                    scalar_dict.update(
                        {"train_metrics/fairseq": loss_fairseq_avg()}
                    )
                    loss_fairseq_avg.reset()

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i,
                     v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i,
                     v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i,
                     v in enumerate(losses_disc_g)}
                )
                audio_dict = {}
                audio_dict.update(
                    {f"train_audio/gt_{i}": gt[i].data.cpu().numpy()
                     for i in range(min(3, gt.shape[0]))}
                )
                audio_dict.update(
                    {f"train_audio/in_{i}": og[i].data.cpu().numpy()
                     for i in range(min(3, og.shape[0]))}
                )
                audio_dict.update(
                    {f"train_audio/pred_{i}": output[i].data.cpu().numpy()
                     for i in range(min(3, output.shape[0]))}
                )
                net_g.eval()

                # load audio from benchmark dir
                test_wavs = [
                    (
                        os.path.basename(p),
                        utils.load_wav_to_torch(p, config['data']['sr']),
                    )
                    for p in glob.glob(config['test_dir'] + "/*.wav")
                ]

                logging.info("Testing...")
                for test_wav_name, test_wav in tqdm(test_wavs, total=len(test_wavs)):
                    test_out = net_g(test_wav.unsqueeze(
                        0).unsqueeze(0).to(device))
                    audio_dict.update(
                        {f"test_audio/{test_wav_name}":
                            test_out[0].data.cpu().numpy()}
                    )

                # don't worry about caching val dataset for now
                for loader in [dev_loader, val_loader]:
                    if loader == dev_loader:
                        loader_name = "dev"
                    else:
                        loader_name = "val"
                    v_data = enumerate(loader)
                    logging.info(f"Validating on {loader_name} dataset...")
                    v_loss_mel_avg = utils.RunningAvg()
                    v_loss_fairseq_avg = utils.RunningAvg()
                    v_mcd_avg = utils.RunningAvg()

                    with torch.no_grad():
                        for v_batch_idx, v_batch in tqdm(v_data, total=len(loader)):
                            v_output, v_gt, og = net_g_step(
                                v_batch, net_g, device, config['fp16_run'])

                        if config['aux_mel']['c'] > 0:
                            v_loss_mel = utils.aux_mel_loss(
                                output, gt, config) * config['aux_mel']['c']
                            v_loss_mel_avg.update(v_loss_mel)
                        if fairseq_model is not None:
                            with autocast(enabled=config['fp16_run']):
                                v_loss_fairseq = utils.fairseq_loss(
                                    output, gt, fairseq_model) * config['aux_fairseq']['c']
                                v_loss_fairseq_avg.update(v_loss_fairseq)
                        v_mcd = utils.mcd(
                            v_output, v_gt, config['data']['sr'])
                        v_mcd_avg.update(v_mcd)

                    if config['aux_mel']['c'] > 0:
                        scalar_dict.update(
                            {f"{loader_name}_metrics/mel": v_loss_mel_avg(),
                             f"{loader_name}_metrics/mcd": v_mcd_avg()}
                        )
                        v_loss_mel_avg.reset()
                    if fairseq_model is not None:
                        scalar_dict.update(
                            {f"{loader_name}_metrics/fairseq": v_loss_fairseq_avg()}
                        )
                        v_loss_fairseq_avg.reset()
                    v_mcd_avg.reset()
                    audio_dict.update(
                        {f"{loader_name}_audio/gt_{i}": v_gt[i].data.cpu().numpy()
                         for i in range(min(3, v_gt.shape[0]))}
                    )
                    audio_dict.update(
                        {f"{loader_name}_audio/in_{i}": og[i].data.cpu().numpy()
                         for i in range(min(3, og.shape[0]))}
                    )
                    audio_dict.update(
                        {f"{loader_name}_audio/pred_{i}": v_output[i].data.cpu().numpy()
                         for i in range(min(3, v_output.shape[0]))}
                    )

                net_g.train()

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict,
                    audios=audio_dict,
                    audio_sampling_rate=config['data']['sr'],
                )

                if global_step % config['checkpoint_interval'] == 0:
                    g_checkpoint = os.path.join(
                        checkpoint_dir, f"G_{global_step}.pth")
                    d_checkpoint = os.path.join(
                        checkpoint_dir, f"D_{global_step}.pth")
                    utils.save_state(
                        net_g,
                        optim_g,
                        lr,
                        epoch,
                        global_step,
                        g_checkpoint
                    )
                    utils.save_state(
                        net_d,
                        optim_d,
                        lr,
                        epoch,
                        global_step,
                        d_checkpoint
                    )
                    logging.info(
                        f"Saved checkpoints to {g_checkpoint} and {d_checkpoint}")
                    progress_bar.reset()
                torch.cuda.empty_cache()

        scheduler_g.step()
        scheduler_d.step()

    if is_main_process:
        logging.info("Training is done. The program is closed.")


def train_model(
    gpus,
    config,
    training_dir
):
    deterministic = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    PREV_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if PREV_CUDA_VISIBLE_DEVICES is None:
        PREV_CUDA_VISIBLE_DEVICES = None
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(gpu) for gpu in gpus])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = PREV_CUDA_VISIBLE_DEVICES

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    mp.spawn(
        training_runner,
        nprocs=len(gpus),
        args=(
            len(gpus),
            config,
            training_dir
        )
    )

    if PREV_CUDA_VISIBLE_DEVICES is None:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', "-d", type=str,
                        help="Path to save checkpoints and logs.")
    args = parser.parse_args()
    with open(os.path.join(args.dir, "config.json")) as f:
        config = json.load(f)
    # get gpus from torch
    gpus = [i for i in range(torch.cuda.device_count())]
    logging.info("Using GPUs: {}".format(gpus))
    # check to see if fairseq checkpoint exists
    if config['aux_fairseq']['c'] > 0:
        if not os.path.exists(config['aux_fairseq']['checkpoint_path']):
            print(
                f"Fairseq checkpoint not found at {config['aux_fairseq']['checkpoint_path']}")
            checkpoint_url = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
            print(f"Downloading from {checkpoint_url}")
            checkpoint_folder = os.path.dirname(
                config['aux_fairseq']['checkpoint_path'])
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            os.system(f"wget {checkpoint_url} -P {checkpoint_folder}")
    train_model(gpus, config, args.dir)


if __name__ == "__main__":
    main()
