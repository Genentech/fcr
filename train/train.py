import os
import time
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from ..evaluate.evaluate import evaluate, evaluate_classic,evaluate_prediction

from ..model import load_FCR


from ..dataset.dataset import load_dataset_splits,load_dataset_train_test

from ..utils.general_utils import initialize_logger, ljson
from ..utils.data_utils import data_collate

def prepare_extract(args, state_dict=None):
    # dataset
    datasets = load_dataset_train_control(
        args["data_path"],
        sample_cf=(True if args["dist_mode"] == "match" else False),
    )

    datasets.update(
        {
            "loader_treatment": torch.utils.data.DataLoader(
                datasets["train_treatment"],
                batch_size=args["batch_size"],
                shuffle=False,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            ),
            
             "loader_control": torch.utils.data.DataLoader(
                datasets["train_control"],
                batch_size=args["batch_size"],
                shuffle=False,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            )
        }
    )

    args["num_outcomes"] = datasets["train"].num_outcomes
    args["num_treatments"] = datasets["train"].num_treatments
    args["num_covariates"] = datasets["train"].num_covariates

    # model
    model = load_FCR(args, state_dict)

    args["hparams"] = model.hparams

    return model, datasets

    

def prepare(args, state_dict=None):
    """
    Instantiates model and dataset to run an experiment.
    """

#     perturbation_key = "perturbation",
#     control_key = "control",
#         dose_key = "dose",
#         covariate_keys = "cell_type",
#         split_key = "split"
    
    
#     # dataset
    if args['covariate_keys']!=None:
        covariate_keys = args['covariate_keys']
    else:
        covariate_keys = 'covariates'

    if args['perturbation_key']!=None:
        perturbation_key = args["perturbation_key"]
    else:
        perturbation_key = "Agg_Treatment"
        
    if args['split']!=None:
        split_key = args["split"]
    
    
    # if args['split']=="split":
    #     datasets = load_dataset_splits(
    #         args["data_path"],
    #         sample_cf=(True if args["dist_mode"] == "match" else False),
    #     )
    # elif args['split']=="new_split":
    datasets = load_dataset_train_test(
    args["data_path"],
    covariate_keys = covariate_keys,
    perturbation_key = perturbation_key,
    split_key = split_key,
    sample_cf=(True if args["dist_mode"] == "match" else False))
    
       

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["train"],
                batch_size=args["batch_size"],
                shuffle=False,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            )
        }
    )

    args["num_outcomes"] = datasets["train"].num_outcomes
    args["num_treatments"] = datasets["train"].num_treatments
    args["num_covariates"] = datasets["train"].num_covariates

    # model
    model = load_FCR(args, state_dict)
    # print("load FCR model")

    args["hparams"] = model.hparams

    return model, datasets

def train(args, prepare=prepare, state_dict=None):
    """
    Trains a FCR model
    """
    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
    
    if state_dict!=None:
        model, datasets = prepare(args, state_dict)  
    else:
        model, datasets = prepare(args)

    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args["artifact_path"], "runs/" + args["name"] + "_" + dt))
    save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
    os.makedirs(save_dir, exist_ok=True)

    initialize_logger(save_dir)
    ljson({"training_args": args})
    ljson({"model_params": model.hparams})
    logging.info("")

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        
        epoch_training_stats = defaultdict(float)
        if epoch % args["adv_epoch"]==0:
            adv_training=True
        else:
            adv_training=False
        # print("Adversarial Training {}".format(adv_training))
        
        for data in datasets["loader_tr"]:
            (expriments, treatment, control, _, covariates)= \
            (data[0], data[1], data[2], data[3], data[4:])
          
            ## training without divergence
            # minibatch_training_stats = model.update(
            #     expriments, treatment, control, covariates, adv_training
            # )
            
            
            ## training with divergence
            minibatch_training_stats = model.update_divergence(
                expriments, treatment, control, covariates, adv_training)

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val
        model.update_eval_encoder()

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(epoch_training_stats[key])
        model.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: 
        # patience ran out OR max epochs reached
        stop = (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats = evaluate_prediction(model, datasets)
            for key, val in evaluation_stats.items():
                if not (key in model.history.keys()):
                    model.history[key] = []
                model.history[key].append(val)
            model.history["stats_epoch"].append(epoch)

            ljson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                    "Discrinimator Training": adv_training
                }
            )

            for key, val in epoch_training_stats.items():
                writer.add_scalar(key, val, epoch)

            torch.save(
                (model.state_dict(), args, model.history),
                os.path.join(
                    save_dir,
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

            ljson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )
            # stop = stop or model.early_stopping(np.mean(evaluation_stats["test"]))
            if stop:
                ljson({"early_stop": epoch})
                break

    writer.close()
    return model
