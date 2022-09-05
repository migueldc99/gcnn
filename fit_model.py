""" Defines fit_model for training a model """

from typing import List

import os
import torch
from torch_geometric.loader import DataLoader

from calculateEnergyAndForces import calculateEnergyAndForces


def fit_model(
    n_epochs: int,
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_train: int,
    n_validation: int,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    n_epoch_0: int = 0,
    calculate_forces: bool = False,
    weight: float = 1.0,
    writer: torch.utils.tensorboard.SummaryWriter = None,
    callbacks: List[object] = None,
    check_point_path: str = "chkpt.tar",
    check_point_freq: int = 10,
) -> None:

    r"""

    This function centralises the optimization of the model parameters

    Args:

        n_epochs (int): the number of epochs (steps) to perform in the
           minimization
        model (torch.nn.Module): the model being optimized
        loss_func (torch.nn.Module): the loss function that is minimized
           during the process
        optimizer (torch.optim.Optimizer): the optimization strategy used
           to minimize the loss
        n_train (int): the number of samples in the training dataset
        n_validation (int): the number of samples in the validation dataset
        train_dl (DataLoader): the data-loader for training
        valid_dl (DataLoader): the data-loader for validation
        n_epoch_0 (int): the initial epoch index (different from 0 if this
           is a continuation run)
        calculate_forces (bool): True if forces need to be calculated and
           fitted against (default false)
        weight (float): if forces are used, the weight to be given to
           forces vs energy (1 means equal weight)
        writer (torch.utils.tensorboard.SummaryWriter): tensorboard writer
           to document the run
        callbacks (List): list of callbacks to be used (e.g. early-stopping,
           user-stop, etc)
        check_point_path (str): filename path to save model information during
           checkpointing or callback-stopping.
        check_point_freq (int): model is checkpointed every check_point_freq
           epochs (steps)

    """

    factor = float(n_train) / float(n_validation)

    model.train()
    
    logFile = "/content/drive/MyDrive/Colab Notebooks/MoleculeDB/runs/"
    file = open(logFile + "loss.csv", "w")
    file.write("epoch,train-loss,validation-loss" + os.linesep)

    for epoch in range(n_epochs):

        n_epoch = n_epoch_0 + epoch

        train_running_loss = 0.0

        for sample in train_dl:

            optimizer.zero_grad()

            if calculate_forces:

                energy, forces = calculateEnergyAndForces(model, sample, \
                                      sample.batch)

                zeros = torch.zeros_like(forces)

                energy_loss = loss_func(torch.squeeze(energy, dim=1), sample.y)

                force_loss = loss_func(forces, zeros)

                train_loss = energy_loss + weight * force_loss

            else:

                energy = model(
                    sample.x, sample.edge_index, sample.edge_attr, sample.batch
                )

                train_loss = loss_func(torch.squeeze(energy, dim=1), sample.y)

            train_loss.backward()

            optimizer.step()

            train_running_loss += train_loss.item()

        val_running_loss = 0.0

        # with torch.set_grad_enabled( False ):

        for sample in valid_dl:

            if calculate_forces:

                energy, forces = calculateEnergyAndForces(model, sample, sample.batch)

                zeros = torch.zeros_like(forces)

                energy_loss = loss_func(torch.squeeze(energy, dim=1), sample.y)

                force_loss = loss_func(forces, zeros)

                val_loss = energy_loss + weight * force_loss

            else:

                torch.set_grad_enabled(False)

                energy = model(
                    sample.x, sample.edge_index, sample.edge_attr, sample.batch
                )

                val_loss = loss_func(torch.squeeze(energy, dim=1), sample.y)

                torch.set_grad_enabled(True)

            val_running_loss += val_loss.item()

        val_running_loss *= (
            factor  # to put it on the same scale as the training running loss
        )
        
        txt = repr(n_epoch) + "," + repr(train_running_loss) + "," + repr(val_running_loss)
        
        file.write(txt + os.linesep)
        print(txt)
        
        if writer is not None:

            writer.add_scalar("training loss", train_running_loss, n_epoch)
            writer.add_scalar("validation loss", val_running_loss, n_epoch)

        # if we should store the current state of mode/optimizer, do so here

        if (
            n_epoch + 1
        ) % check_point_freq == 0:  # n_epoch + 1 to ensure saving at the last iteration too

            torch.save(
                {
                    "epoch": n_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_running_loss,
                    "val_loss": val_running_loss,
                },
                check_point_path,
            )

        # if there are any callbacks, act them if needed

        for callback in callbacks:

            callback(train_running_loss)

            # check for early stopping; if true, we return to main function

            if (
                callback.early_stop
            ):  # if we are to stop, make sure we save model/optimizer

                torch.save(
                    {
                        "epoch": n_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_running_loss,
                        "val_loss": val_running_loss,
                    },
                    check_point_path,
                )

                return
            
    file.close()
