import os
import json
import math, random
import torch
from tqdm.auto import tqdm

from .loss import loss_func
from .common_utils import touch_print
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger

logger = get_logger(__name__)


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f'{key}', value, completed_steps)


def accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir: str, completed_steps: int, args,
                                   optimizer=None, lr_scheduler=None, epoch=None,
                                   min_eval_loss=None, stall_num=None, best_step=None):
    accelerator.wait_for_everyone()

    accelerator.print("[CHECKPOINT] Saving checkpoint")

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        torch.save(accelerator.get_state_dict(model.gnn), f"{output_dir}/GNN.pth")
        torch.save(accelerator.get_state_dict(model.adapter), f"{output_dir}/adapter.pth")

        # Save training state for true resume support
        if optimizer is not None and lr_scheduler is not None and epoch is not None:
            torch.save(optimizer.state_dict(), f"{output_dir}/optimizer.pth")
            torch.save(lr_scheduler.state_dict(), f"{output_dir}/scheduler.pth")
            training_state = {
                "completed_steps": int(completed_steps),
                "epoch": int(epoch),
                "min_eval_loss": float(min_eval_loss) if min_eval_loss != float('inf') else None,
                "stall_num": int(stall_num),
                "best_step": int(best_step) if best_step is not None else None,
            }
            with open(f"{output_dir}/training_state.json", "w") as f:
                json.dump(training_state, f, indent=2)

    if args.mode != 'pt':
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.lm.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model.lm)
        )

    accelerator.print(
        f"[CHECKPOINT][complete_steps={completed_steps}], checkpoint {output_dir} saved"
    )
    accelerator.wait_for_everyone()


def accelerate_monitor(accelerator, reduce_loss, reduce_loss_ft, args, completed_steps,
                       lr_scheduler, optimizer, summary_writer, reduce_step, reduce_step_ft):
    """
    gather reduce_loss from all N devices.
    train logging and tensorboarding.
    """
    if not isinstance(reduce_loss, int):
        reduce_losses = accelerator.gather(reduce_loss)
        
        train_loss = torch.mean(reduce_losses) / reduce_step

        # logging and tensorboard
        logger.info(
            f"[TRAIN][complete_steps={completed_steps}][train_loss={train_loss:.6f}]"
            f"[gather shape={reduce_losses.shape}][lr={lr_scheduler.get_lr()[0]:.4e}, {optimizer.param_groups[0]['lr']:.4e}]",
        )

        train_log_dict = {"training_loss": train_loss, "lr": lr_scheduler.get_lr()[0]}
    else:
        train_log_dict = {"lr": lr_scheduler.get_lr()[0]}

    if args.mode == 'ft' and not isinstance(reduce_loss_ft, int):
        reduce_loss_ft = accelerator.gather(reduce_loss_ft)
        train_loss_ft = torch.mean(reduce_loss_ft) / reduce_step_ft
        train_log_dict["training_loss_ft"] = train_loss_ft
    
    if accelerator.is_main_process:
        write_tensorboard(summary_writer, train_log_dict, completed_steps)


def accelerate_evaluate(accelerator, model, valid_dataloader, valid_dataloader_ft, args, completed_steps, step, min_eval_loss, stall_num,
                        best_step, summary_writer, tokenizer):
    """
    evaluate the model at current completed_steps on valid_dataloader and gather eval_loss on all devices.
    eval logging and tensorboarding.
    """
    losses = []
    for batch in valid_dataloader:
        with torch.no_grad():
            outputs = model(batch)

            loss = loss_func(
                outputs=outputs,
                labels=batch['labels'],
                loss_mask=batch['loss_mask'],
            )

            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
            # print(losses[-1].shape)

    accelerator.wait_for_everyone()
    valid_batch_num = len(losses)
    gathered_size = losses[0].shape
    losses = torch.cat(losses)

    try:
        eval_loss = torch.mean(losses)
        if eval_loss <= min_eval_loss:
            min_eval_loss = eval_loss
            stall_num = 0
            best_step = completed_steps
            # Save best checkpoint
            best_dir = os.path.join(args.output_dir, "best_checkpoint") if args.output_dir else "best_checkpoint"
            accelerate_saving_checkpoint(accelerator, model, tokenizer, best_dir, completed_steps, args)
            accelerator.print(f"[BEST] New best eval_loss={eval_loss:.6f} at step {completed_steps} → saved to {best_dir}")
        else:
            stall_num += 1
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"[EVAL][global_steps={step + 1}][completed_steps={completed_steps}]"
                f"[valid_batch_num={valid_batch_num}], [gather_size={gathered_size}]"
                f"[perplexity={perplexity:.4f}][eval_loss={eval_loss:.6f}]")
    eval_log_dict = {"valid_loss": eval_loss.float(),
                     "perplexity": perplexity}
    
    if args.mode == 'ft' and len(valid_dataloader_ft) > 0:
        losses = []
        for batch in valid_dataloader_ft:
            with torch.no_grad():
                outputs = model(batch)

                loss = loss_func(
                    outputs=outputs,
                    labels=batch['labels'],
                    loss_mask=batch['loss_mask'],
                )

                losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        accelerator.wait_for_everyone()
        valid_batch_num_ft = len(losses)
        gathered_size_ft = losses[0].shape
        losses = torch.cat(losses)

        try:
            eval_loss_ft = torch.mean(losses)
            perplexity_ft = math.exp(eval_loss_ft)
        except OverflowError:
            perplexity_ft = float("inf")

        logger.info(f"[valid_batch_num_ft={valid_batch_num_ft}], [gather_size_ft={gathered_size_ft}]"
                    f"[perplexity_ft={perplexity_ft:.4f}][eval_loss_ft={eval_loss_ft:.6f}]")
        eval_log_dict["valid_loss_ft"] = eval_loss_ft.float()
        eval_log_dict["perplexity_ft"] = perplexity_ft

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, eval_log_dict, completed_steps)

    return eval_loss, min_eval_loss, stall_num, best_step


def accelerate_train(accelerator, model, train_dataloader, valid_dataloader, train_dataloader_ft, valid_dataloader_ft,
                     optimizer, lr_scheduler, tokenizer, total_train_dataset_size, args):
    # tensorboard writer
    summary_writer = SummaryWriter(log_dir=args.tb_dir) if accelerator.is_main_process else None
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("**************************************** Running training ****************************************")
    logger.info(f"  Num examples = {total_train_dataset_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total global train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization(update/completed) steps = {args.max_train_steps}")
    logger.info(f"  Complete/Optimization steps per Epoch = {args.max_train_steps // args.num_train_epochs}")
    logger.info("***************************************************************************************************")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )

    # set starting_epoch, completed_steps and resume_step of train_dataloader
    completed_steps = 0
    starting_epoch = 0
    resume_skip_steps = 0  # inner-loop batches to skip on first resumed epoch

    # monitor minimum eval_loss, stalling num, and best_step
    min_eval_loss = float('inf')
    stall_num = 0
    best_step = None

    # Load training state for true resume
    if args.checkpoint and os.path.exists(f"{args.checkpoint}/training_state.json"):
        with open(f"{args.checkpoint}/training_state.json") as f:
            state = json.load(f)
        completed_steps = state["completed_steps"]
        starting_epoch = state["epoch"]
        min_eval_loss = state["min_eval_loss"] if state["min_eval_loss"] is not None else float('inf')
        stall_num = state["stall_num"]
        best_step = state["best_step"]
        steps_per_epoch = args.max_train_steps // args.num_train_epochs if args.max_train_steps else 0
        resume_skip_steps = (completed_steps % steps_per_epoch) * args.gradient_accumulation_steps if steps_per_epoch else 0
        if os.path.exists(f"{args.checkpoint}/optimizer.pth"):
            optimizer.load_state_dict(torch.load(f"{args.checkpoint}/optimizer.pth", map_location="cpu"))
        if os.path.exists(f"{args.checkpoint}/scheduler.pth"):
            lr_scheduler.load_state_dict(torch.load(f"{args.checkpoint}/scheduler.pth"))
        progress_bar.update(completed_steps)
        accelerator.print(f"[RESUME] Resuming from step {completed_steps}, epoch {starting_epoch}, skipping {resume_skip_steps} batches")

    # monitor train loss
    reduce_loss, reduce_step = 0, 0
    reduce_loss_ft, reduce_step_ft = 0, 0
    last_train_loss = float('nan')
    last_eval_loss = float('nan')

    # Training Loop!
    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.early_stopping and stall_num == args.early_stopping_stall_num:
            break
        progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_train_epochs}")

        # prepare dataloaders
        train_dataloader_iter = iter(train_dataloader)
        next_idx, next_ft_idx = 0, 0
        if args.mode == 'ft':
            train_dataloader_ft_iter = iter(train_dataloader_ft)
            ratio = len(train_dataloader_ft) / (len(train_dataloader) + len(train_dataloader_ft)) if len(train_dataloader_ft) > 0 else 0
            loss_ratio = min(len(train_dataloader_ft) / len(train_dataloader), 1) if len(train_dataloader_ft) > 0 else 1
        
        def get_batch(next_idx, next_ft_idx):
            if args.mode == 'pt' or next_ft_idx == len(train_dataloader_ft):
                # pt mode, or ft mode but ft dataloader is over
                next_idx += 1
                return next(train_dataloader_iter), next_idx, next_ft_idx
            elif next_idx == len(train_dataloader):
                # graph dataloader is over
                next_ft_idx += 1
                return  next(train_dataloader_ft_iter), next_idx, next_ft_idx
            else:
                assert next_idx < len(train_dataloader) and next_ft_idx < len(train_dataloader_ft)
                if random.random() < ratio:
                    next_ft_idx += 1
                    return next(train_dataloader_ft_iter), next_idx, next_ft_idx
                else:
                    next_idx += 1
                    return next(train_dataloader_iter), next_idx, next_ft_idx

        print(f"length of dataloader: {len(train_dataloader)}")
        if args.mode == 'ft':
            print(f"length of dataloader: {len(train_dataloader_ft)}, ratio: {ratio}")

        # How many inner-loop batches to skip for this epoch (only on first resumed epoch)
        skip_remaining = resume_skip_steps if epoch == starting_epoch else 0
        resume_skip_steps = 0  # only skip once

        model.train()
        # Inner Loop!
        for step in range(len(train_dataloader) + len(train_dataloader_ft) if args.mode == 'ft' else len(train_dataloader)):

            batch, next_idx, next_ft_idx = get_batch(next_idx, next_ft_idx)

            # Skip already-processed batches when resuming
            if skip_remaining > 0:
                skip_remaining -= 1
                continue

            with accelerator.accumulate(model):
                if batch is None:
                    continue
                # forward
                outputs = model(batch)

                # loss
                loss = loss_func(
                    outputs=outputs,
                    labels=batch['labels'],
                    loss_mask=batch['loss_mask'],
                ) * (loss_ratio if 'graph_embedding' in batch.keys() and args.mode == 'ft' else 1)

                # backward
                accelerator.backward(loss)

                # clip gradients to prevent NaN loss
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                # update(sync_gradients)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # support args.min_lr
                if optimizer.param_groups[0]['lr'] <= args.min_lr:
                    optimizer.param_groups[0]['lr'] = args.min_lr

                # accumulate resuce_loss in a log_interval
                if not torch.isnan(loss):
                    if 'graph_embedding' in batch.keys():
                        reduce_loss += loss.detach().float() / (loss_ratio if 'graph_embedding' in batch.keys() and args.mode == 'ft' else 1)
                        reduce_step += 1
                    else:
                        assert args.mode == 'ft'
                        reduce_loss_ft += loss.detach().float()
                        reduce_step_ft += 1

                # If the accelerator has performed an optimization step behind the scenes, thus a completed_step done.
                if accelerator.sync_gradients:

                    completed_steps += 1
                    progress_bar.update(1)

                    # monitoring training process and logging and tensorboarding
                    if completed_steps % args.log_interval == 0:
                        accelerate_monitor(
                            accelerator, reduce_loss, reduce_loss_ft, args, completed_steps,
                            lr_scheduler, optimizer, summary_writer, reduce_step, reduce_step_ft
                        )
                        if not isinstance(reduce_loss, int) and reduce_step > 0:
                            last_train_loss = (reduce_loss / reduce_step).item()
                        reduce_loss, reduce_loss_ft = 0, 0
                        reduce_step, reduce_step_ft = 0, 0

                    progress_bar.set_postfix(
                        loss=f"{last_train_loss:.4f}",
                        eval=f"{last_eval_loss:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

                    # steps checkpointing
                    if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir, completed_steps, args,
                                                     optimizer=optimizer, lr_scheduler=lr_scheduler, epoch=epoch,
                                                     min_eval_loss=min_eval_loss, stall_num=stall_num, best_step=best_step)

                    # steps evaluation
                    if completed_steps % args.evaluation_steps == 0:
                        model.eval()
                        eval_loss, min_eval_loss, stall_num, best_step = accelerate_evaluate(
                            accelerator, model, valid_dataloader, valid_dataloader_ft, args, completed_steps, step,
                            min_eval_loss, stall_num, best_step, summary_writer, tokenizer
                        )
                        last_eval_loss = eval_loss.item()
                        model.train()

                        # early stoppin when stalling more than args.early_stopping_stall_num
                        if args.early_stopping and stall_num == args.early_stopping_stall_num:
                            accelerator.print(f"[WARNING] Early stopping at {completed_steps}")
                            break

                    if completed_steps >= args.max_train_steps:
                        break
                    accelerator.wait_for_everyone()

        # epoch checkpointing
        if args.epoch_checkpointing:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir, completed_steps, args)

    if summary_writer:
        summary_writer.close()

    # final save
    output_dir = f"final_step_{completed_steps}"
    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, output_dir)
    accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir, completed_steps, args)
