import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from .module import (
    CompoundEmbedding, MLP,
    Bernoulli, NegativeBinomial, ZeroInflatedNegativeBinomial
)

from ..utils.math_utils import (
    logprob_normal, kldiv_normal,
    kldiv_normal_marginal,
    logprob_bernoulli_logits,
    logprob_nb_positive,
    logprob_zinb_positive,
    aggregate_normal_distr,
    marginalize_latent_tx,
    marginalize_latent
    
)

#####################################################
#                     LOAD MODEL                    #
#####################################################

def load_FCR(args, state_dict=None):
    device = (
        "cuda:" + str(args["gpu"])
            if (not args["cpu"]) 
                and torch.cuda.is_available() 
            else 
        "cpu"
    )

    model = FCR(
        args["num_outcomes"],
        args["num_treatments"],
        args["num_covariates"],
        omega0=args["omega0"],
        omega1=args["omega1"],
        omega2=args["omega2"],
        dist_mode=args["dist_mode"],
        dist_outcomes=args["dist_outcomes"],
        patience=args["patience"],
        device=device,
        distance=args['distance'],
        hparams=args["hparams"]
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

#####################################################
#                     MAIN MODEL                    #
#####################################################

class FCR(nn.Module):
    def __init__(
        self,
        num_outcomes,
        num_treatments,
        num_covariates,
        embed_outcomes=True,
        embed_treatments=False,
        embed_covariates=True,
        omega0=1.0,
        omega1=2.0,
        omega2=2.0,
        dist_mode="match",
        dist_outcomes="normal",
        type_treatments=None,
        type_covariates=None,
        mc_sample_size=30,
        best_score=-1e3,
        patience=5,
        distance="element",
        device="cuda",
        hparams=""
    ):
        super(FCR, self).__init__()
        # generic attributes
        self.num_outcomes = num_outcomes
        self.num_treatments = num_treatments
        self.num_covariates = num_covariates
        self.embed_outcomes = embed_outcomes
        self.embed_treatments = embed_treatments
        self.embed_covariates = embed_covariates
        self.dist_outcomes = dist_outcomes
        self.type_treatments = type_treatments
        self.type_covariates = type_covariates
        self.mc_sample_size = mc_sample_size
        # fcr parameters
        self.omega0 = omega0
        self.omega1 = omega1
        self.omega2 = omega2
        self.dist_mode = dist_mode
        # early-stopping
        self.best_score = best_score
        self.patience = patience
        self.patience_trials = 0
        self.distance = distance
        # set hyperparameters
        self._set_hparams_(hparams)

        # individual-specific model
        self._init_indiv_model()

        # covariate-specific model
        self._init_covar_model()

        self.iteration = 0

        self.history = {"epoch": [], "stats_epoch": []}

        self.to_device(device)

    def _set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "latent_dim": 64,
            "latent_exp_dim":192,
            "ZX_dim": 64,
            "ZT_dim":64,
            "ZXT_dim":64,            
            "outcome_emb_dim": 256,
            "treatment_emb_dim": 64,
            "covariate_emb_dim": 16,
            "encoder_width": 128,
            "encoder_depth": 3,
            "decoder_width": 128,
            "decoder_depth": 3,
            "discriminator_width": 64,
            "discriminator_depth": 2,
            "autoencoder_lr": 3e-4,
            "discriminator_lr": 3e-4,
            "autoencoder_wd": 4e-7,
            "discriminator_wd": 4e-7,
            "discriminator_steps": 3,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                with open(hparams) as f:
                    dictionary = json.load(f)
                self.hparams.update(dictionary)
            else:
                self.hparams.update(hparams)

        self.outcome_dim = (
            self.hparams["outcome_emb_dim"] if self.embed_outcomes else self.num_outcomes)
        self.treatment_dim = (
            self.hparams["treatment_emb_dim"] if self.embed_treatments else self.num_treatments)
        self.covariate_dim = (
            self.hparams["covariate_emb_dim"]*len(self.num_covariates) 
            if self.embed_covariates else sum(self.num_covariates)
        )
        self.treatment_mixed_dim = self.covariate_dim

        return self.hparams

    def _init_indiv_model(self):
        params = []

        # embeddings
        if self.embed_outcomes:
            self.outcomes_embeddings = self.init_outcome_emb()
            self.outcomes_contr_embeddings = self.init_outcome_emb()
            
            params.extend(list(self.outcomes_embeddings.parameters()))
            params.extend(list(self.outcomes_contr_embeddings.parameters()))

        if self.embed_treatments:
            self.treatments_embeddings = self.init_treatment_emb()
            params.extend(list(self.treatments_embeddings.parameters()))

        if self.embed_covariates:
            self.covariates_embeddings = nn.Sequential(*self.init_covariates_emb())
            for emb in self.covariates_embeddings:
                params.extend(list(emb.parameters()))
                
        self.treatments_mixed_embeddings = self.init_treatment_mixed_emb()
        params.extend(list(self.treatments_mixed_embeddings.parameters()))        
        
        ## dimension parameters
        self.ZX_dim = self.hparams["ZX_dim"]
        self.ZT_dim = self.hparams["ZT_dim"]
        self.ZXT_dim = self.hparams["ZXT_dim"]
        
        # models
        ## exp_encoder and control_encoder
        # self.exp_encoder = self.init_encoder_exp()
        self.encoder_ZX = self.init_encoder_X()
        self.encoder_ZT = self.init_encoder_T()
        self.encoder_ZXT = self.init_encoder_TX()

        
        
        # self.control_encoder = self.init_encoder_control()
        params.extend(list(self.encoder_ZX.parameters()))
        params.extend(list(self.encoder_ZT.parameters()))
        params.extend(list(self.encoder_ZXT.parameters()))
        # params.extend(list(self.control_encoder.parameters()))
        
        ## initialize the prior encoders
        self.encoder_ZX_prior = self.init_encoder_X_prior()
        self.encoder_ZT_prior = self.init_encoder_T_prior()
        self.encoder_ZXT_prior = self.init_encoder_XT_prior()
        # self.control_prior = self.init_control_prior()
        params.extend(list(self.encoder_ZX_prior.parameters()))
        params.extend(list(self.encoder_ZT_prior.parameters()))
        params.extend(list(self.encoder_ZXT_prior.parameters()))
        # params.extend(list(self.control_prior.parameters()))

        ## eval models
        self.encoder_ZX_eval = copy.deepcopy(self.encoder_ZX)
        self.encoder_ZT_eval = copy.deepcopy(self.encoder_ZT)
        self.encoder_ZXT_eval = copy.deepcopy(self.encoder_ZXT)

        self.encoder_ZX_prior_eval = copy.deepcopy(self.encoder_ZX_prior)
        self.encoder_ZT_prior_eval = copy.deepcopy(self.encoder_ZT_prior)
        self.encoder_ZXT_prior_eval = copy.deepcopy(self.encoder_ZXT_prior)
        # self.control_encoder_eval = copy.deepcopy(self.control_encoder)
        # self.control_prior_eval = copy.deepcopy(self.control_prior)

        self.decoder = self.init_decoder_experiments()
        params.extend(list(self.decoder.parameters()))
        # params.extend(list(self.control_decoder.parameters()))
        
        ## covariate decoder
        self.cov_decoder = self.init_decoder_cov()
        params.extend(list(self.cov_decoder.parameters()))
                
        ##intervention decoder
        # print("intervention decoder style {}".format(self.distance))
        if self.distance == "cosine":
            self.interv_decoder = self.init_decoder_interv()
        elif self.distance == "element": 
            self.interv_decoder = self.init_decoder_interv_element()
        elif self.distance == "concat":
            self.interv_decoder = self.init_decoder_interv_concat()
        elif self.distance=="single":
            self.interv_decoder = self.init_decoder_interv_single()


        params.extend(list(self.interv_decoder.parameters()))

        # optimizer
        self.optimizer_autoencoder = torch.optim.Adam(
            params,
            lr=self.hparams["autoencoder_lr"],
            weight_decay=self.hparams["autoencoder_wd"],
        )
        self.scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            self.optimizer_autoencoder, step_size=self.hparams["step_size_lr"]
        )

        # return self.exp_encoder,self.decoder, self.control_encoder, self.control_decoder, self.cov_decoder, self.interv_decoder
        return self.exp_encoder,self.decoder, self.cov_decoder, self.interv_decoder

    def _init_covar_model(self):

        # if self.dist_mode == "discriminate":
        params = []

        # embeddings
        if self.embed_outcomes:
            self.adv_outcomes_emb = self.init_outcome_emb()
            params.extend(list(self.adv_outcomes_emb.parameters()))

        if self.embed_treatments:
            self.adv_treatments_emb = self.init_treatment_emb()
            params.extend(list(self.adv_treatments_emb.parameters()))

        if self.embed_covariates:
            self.adv_covariates_emb = nn.Sequential(*self.init_covariates_emb())
            for emb in self.adv_covariates_emb:
                params.extend(list(emb.parameters()))

        # model
        self.discriminator_X = self.init_discriminator_X()
        self.loss_discriminator_X = nn.BCEWithLogitsLoss()
        params.extend(list(self.discriminator_X.parameters()))

        self.discriminator_T = self.init_discriminator_T()
        self.loss_discriminator_T = nn.BCEWithLogitsLoss()
        params.extend(list(self.discriminator_T.parameters()))
        # print("initialized discriminator {}".format(self.discriminator_T))

        self.optimizer_discriminator = torch.optim.Adam(
            params,
            lr=self.hparams["discriminator_lr"],
            weight_decay=self.hparams["discriminator_wd"],
        )
        self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
            self.optimizer_discriminator, step_size=self.hparams["step_size_lr"]
        )

        return self.discriminator_X, self.discriminator_T

#         elif self.dist_mode == "fit":
#             raise NotImplementedError(
#                 'TODO: implement dist_mode "fit" for distribution loss')

#         elif self.dist_mode == "match":
#             return None

#         else:
#             raise ValueError("dist_mode not recognized")

    def encode_control(self, outcomes, covariates, eval=False):
        
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
            
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]

        covariates = torch.cat(covariates, -1)
        inputs = torch.cat([outcomes, covariates], -1)

        if eval:
            return self.control_encoder_eval(inputs)
        else:
            return self.control_encoder(inputs)
        
    def encode_exp(self, outcomes, covariates, treatments, eval=False):
        
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)    
            
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]

        covariates = torch.cat(covariates, -1)
        inputs = torch.cat([outcomes,treatments ,covariates], -1)

        if eval:
            return self.exp_encoder_eval(inputs)
        else:
            return self.exp_encoder(inputs)
        
        
    def encode_ZX(self, outcomes, covariates, eval=False):
        
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_covariates:
            # print("covariates[0] {}".format(covariates[0]))
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
        covariates = torch.cat(covariates, -1)

        inputs = torch.cat([outcomes, covariates], -1)

        if eval:
            return self.encoder_ZX_eval(inputs)
        else:
            return self.encoder_ZX(inputs)
        
    def encode_ZT(self, outcomes, treatments, eval=False):
        if self.embed_outcomes:
            outcomes = self.outcomes_embeddings(outcomes)
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)
    
        inputs = torch.cat([outcomes, treatments], -1)

        if eval:
            return self.encoder_ZT_eval(inputs)
        else:
            return self.encoder_ZT(inputs)
        
        
    def encode_ZXT(self, outcomes, treatments, eval=False):
         
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
            
        covariates = torch.cat(covariates, -1)
        treatments = self.treatments_mixed_embeddings(treatments)
    
        inputs = torch.cat([outcomes, treatments], -1)

        if eval:
            return self.encoder_ZXT_eval(inputs)
        else:
            return self.encoder_ZXT(inputs)
            

          
    def encode_ZT_prior(self, treatments, eval=False):
        if self.embed_treatments:
            treatments = self.treatments_embeddings(treatments)
    
        inputs = treatments

        if eval:
            return self.encoder_ZT_prior_eval(inputs)
        else:
            return self.encoder_ZT_prior(inputs)    
        
    
    def encode_ZX_prior(self, covariates, eval=False):
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
        covariates = torch.cat(covariates, -1)
        
        inputs = covariates

        if eval:
            return self.encoder_ZX_prior_eval(inputs)
        else:
            return self.encoder_ZX_prior(inputs)
        
        
        
    def encode_ZXT_prior(self, covariates, treatments, eval=False):
        
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
            
        covariates = torch.cat(covariates, -1)
        treatments = self.treatments_mixed_embeddings(treatments)
        inputs = torch.mul(covariates, treatments)

        if eval:
            return self.encoder_ZXT_prior_eval(inputs)
        else:
            return self.encoder_ZXT_prior(inputs)        
     
    
    def encode_control_prior(self, covariates, eval=False):
        
        if self.embed_covariates:
            covariates = [emb(covars) for covars, emb in 
                zip(covariates, self.covariates_embeddings)
            ]
        covariates = torch.cat(covariates, -1)
        inputs = covariates

        if eval:
            return self.control_prior_eval(inputs)
        else:
            return self.control_prior(inputs)    
        
    
    def decode(self, latents):
        inputs = latents
        return self.decoder(inputs)
    

    def control_decode(self, latents):
        
        inputs = latents
        return self.control_decoder(inputs)
    
    def covariate_decode(self, ZX):        
        inputs = ZX
        return self.cov_decoder(inputs)
    
    
    def intervention_decode(self, ZT, latent_control):
        if self.distance == "cosine":
            similarities = torch.nn.CosineSimilarity(dim=1)
            inputs = similarities(ZT, latent_control)
            inputs = torch.unsqueeze(inputs, 1)
        elif self.distance=="element":
            inputs = torch.sub(ZT, latent_control)
        elif self.distance == "concat":
            inputs = torch.cat([ZT ,latent_control], -1)
    
        return self.interv_decoder(inputs)
        
    
    def discriminate_T(self, simulation):

        inputs = simulation

        return self.discriminator_T(inputs).squeeze()
    
    def discriminate_X(self, simulation):

        inputs = simulation

        return self.discriminator_X(inputs).squeeze()
    

    def reparameterize(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param sigma: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        eps = torch.randn_like(sigma)
        return eps * sigma + mu

    def distributionize(self, constructions, dim=None, dist=None, eps=1e-3):
        if dim is None:
            dim = self.num_outcomes
        if dist is None:
            dist = self.dist_outcomes

        if dist == "nb":
            mus = F.softplus(constructions[..., 0]).add(eps)
            thetas = F.softplus(constructions[..., 1]).add(eps)
            dist = NegativeBinomial(
                mu=mus, theta=thetas
            )
        elif dist == "zinb":
            mus = F.softplus(constructions[..., 0]).add(eps)
            thetas = F.softplus(constructions[..., 1]).add(eps)
            zi_logits = constructions[..., 2].add(eps)
            dist = ZeroInflatedNegativeBinomial(
                mu=mus, theta=thetas, zi_logits=zi_logits
            )
        elif dist == "normal":
            locs = constructions[..., 0]
            scales = F.softplus(constructions[..., 1]).add(eps)
            dist = Normal(
                loc=locs, scale=scales
            )
        elif dist == "bernoulli":
            logits = constructions[..., 0]
            dist = Bernoulli(
                logits=logits
            )

        return dist

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor, treatments: torch.Tensor, 
            size=1) -> torch.Tensor:
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        treatments = treatments.repeat(size, 1)

        latents = self.reparameterize(mu, sigma)

        return self.decode(latents, treatments)
    
    
    def sample_latent(self,  mu: torch.Tensor, sigma: torch.Tensor, size=1):
        
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        latents = self.reparameterize(mu, sigma)
        return latents
    
    def sample_control(self, mu: torch.Tensor, sigma: torch.Tensor,
            size=1) -> torch.Tensor:
        
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        latents = self.reparameterize(mu, sigma)
        return self.control_decode(latents)
    
    
    def sample_expr(self, mu: torch.Tensor, sigma: torch.Tensor,size=1) -> torch.Tensor:
        
        mu = mu.repeat(size, 1)
        sigma = sigma.repeat(size, 1)
        latents = self.reparameterize(mu, sigma)        
        return self.decode(latents)
    
    ## permutation function for latent space Z_XT
    ## input: joint distribution mu and sigma, and ZXT's sigma mu
    ## output: joint sampled ZXT and ZT, and independently sampled ZXT with previous sampled ZT
    
    def permutation_distribution_X(self, mu, sigma, conditions):
        sample1 = self.sample_latent(mu, sigma, size=1)
        sample1 = marginalize_latent(sample1, conditions)
        ZX1 = sample1[:, :self.ZX_dim]
        ZXT1 = sample1[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim]
        label1 = torch.ones(sample1.shape[0], device="cuda",dtype=torch.float32).unsqueeze(1)
        original = torch.cat([ZX1, ZXT1, label1], -1)
        
        
        ## resample the ZXT
        sample2 = self.sample_latent(mu, sigma, size=1)
        sample2 = marginalize_latent(sample2, conditions)
        ZXT2 = sample2[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim]
        label2 = torch.zeros(sample2.shape[0],device="cuda", dtype=torch.float32).unsqueeze(1)
        perturbed = torch.cat([ZX1, ZXT2,label2], -1)
        # print("orignal shape {}, perturb shape {}".format(original.shape, perturbed.shape))
        
        combined =  torch.cat([original, perturbed], 0)
        index = torch.randperm(combined.shape[0])
        combined = combined[index, :]
        # print("combined shape {}".format(combined.shape))
        
        return combined
        
        
    def permutation_distribution_T(self, mu, sigma, conditions):
        sample1 = self.sample_latent(mu, sigma, size=1)
        sample1 = marginalize_latent(sample1, conditions)
        ZT1 = sample1[:, self.ZX_dim+self.ZXT_dim:]
        ZXT1 = sample1[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim]
        label1 = torch.ones(sample1.shape[0], dtype=torch.float32,device="cuda").unsqueeze(1)
        original = torch.cat([ZXT1,ZT1,label1], -1)
        
        ## resample the ZXT
        sample2 = self.sample_latent(mu, sigma, size=1)
        sample2 = marginalize_latent(sample2, conditions)
        ZXT2 = sample2[:, self.ZX_dim+self.ZXT_dim:]
        label2 = torch.zeros(sample2.shape[0], dtype=torch.float32,device="cuda").unsqueeze(1)
        perturbed = torch.cat([ZXT2,ZT1,label2], -1)
        # print("orignal shape {}, perturb shape {}".format(original.shape, perturbed.shape))
        
        ##
        combined =  torch.cat([original, perturbed], 0)
        index = torch.randperm(combined.shape[0])
        combined = combined[index, :]
        # print("combined shape {}".format(combined.shape))
        
        return combined
    
    
    
    def permutation_samples_X(self, mu: torch.Tensor, sigma: torch.Tensor, size=1):
        
        sample1 = self.sample_latent(mu, sigma, size)
        ZX1 = sample1[:, :self.ZX_dim]
        ZXT1 = sample1[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim]
        label1 = torch.ones(sample1.shape[-1], dtype=torch.float64)
        original = torch.cat([ZX1, ZXT1, label1], -1)
        
        ## resample the ZXT
        sample2 = self.sample_latent(mu, sigma, size)
        ZXT2 = sample2[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim]
        label2 = torch.zeros(sample2.shape[-1], dtype=torch.float64)
        perturbed = torch.cat([ZX1, ZXT2,label2], -1)
        print("orignal shape {}, perturb shape {}".format(original.shape, perturbed.shape))
        
        ##
        combined =  torch.cat([original, perturbed], 0)
        index = torch.randperm(combined.shape[0])
        combined = combined[index, :]
        print("combined shape {}".format(combined.shape))
        
        return combined
        
        
    def permutation_samples_T(self, mu: torch.Tensor, sigma: torch.Tensor, size=1):
        
        sample1 = self.sample_latent(mu, sigma, size)
        ZT1 = sample1[:, self.ZX_dim+self.ZXT_dim:]
        ZXT1 = sample1[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim]
        label1 = torch.ones(sample1.shape[-1], dtype=torch.float64)
        original = torch.cat([ZXT1,ZT1,label1], -1)
        
        ## resample the ZXT
        sample2 = self.sample_latent(mu, sigma, size)
        ZXT2 = sample2[:, self.ZX_dim+self.ZXT_dim:]
        label2 = torch.zeros(sample2.shape[-1], dtype=torch.float64)
        perturbed = torch.cat([ZXT2,ZT1,label2], -1)
        print("orignal shape {}, perturb shape {}".format(original.shape, perturbed.shape))
        
        ##
        combined =  torch.cat([original, perturbed], 0)
        index = torch.randperm(combined.shape[0])
        combined = combined[index, :]
        print("combined shape {}".format(combined.shape))
        
        return combined
    

    def predict(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr = self.encode_exp(outcomes, treatments, covariates)
            latents_dist = self.distributionize(
                latents_constr, dim=self.hparams["latent_dim"], dist="normal"
            )

            outcomes_constr = self.decode(latents_dist.mean)
            outcomes_dist = self.distributionize(outcomes_constr)

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean
        
        
    def predict_self(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, covariates = self.move_inputs(
            outcomes, treatments, covariates
        )
        with torch.autograd.no_grad():
            ZX_constr = self.encode_ZX(outcomes, covariates)
            ZX_dist = self.distributionize(
            ZX_constr, dim=self.hparams["ZX_dim"], dist="normal"
        )
        
            ZT_constr = self.encode_ZT(outcomes, treatments)
            ZT_dist = self.distributionize(
                ZT_constr, dim=self.hparams["ZT_dim"], dist="normal"
            )
            
            outcomes_constr = self.sample_expr(ZX_dist.mean, ZX_dist.stddev,ZT_dist.mean, ZT_dist.stddev)
            outcomes_dist = self.distributionize(outcomes_constr)
 

        if return_dist:
            return outcomes_dist
        else:
            return outcomes_dist.mean    

    def generate(
        self,
        outcomes,
        treatments,
        cf_treatments,
        covariates,
        return_dist=False
    ):
        outcomes, treatments, cf_treatments, covariates = self.move_inputs(
            outcomes, treatments, cf_treatments, covariates
        )
        if cf_treatments is None:
            cf_treatments = treatments

        with torch.autograd.no_grad():
            latents_constr = self.encode(outcomes, treatments, covariates)
            latents_dist = self.distributionize(
                latents_constr, dim=self.hparams["latent_dim"], dist="normal"
            )

            outcomes_constr_samp = self.sample(
                latents_dist.mean, latents_dist.stddev, cf_treatments
            )
            outcomes_dist_samp = self.distributionize(outcomes_constr_samp)

        if return_dist:
            return outcomes_dist_samp
        else:
            return outcomes_dist_samp.mean

    def logprob(self, outcomes, outcomes_param, dist=None):
        """
        Compute log likelihood.
        """
        if dist is None:
            dist = self.dist_outcomes

        num = len(outcomes)
        if isinstance(outcomes, list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes_param = [
                torch.repeat_interleave(out, sizes, dim=0) 
                for out in outcomes_param
            ]
            outcomes = torch.cat(outcomes, 0)
        elif isinstance(outcomes_param[0], list):
            sizes = torch.tensor(
                [out.size(0) for out in outcomes_param[0]], device=self.device
            )
            weights = torch.repeat_interleave(1./sizes, sizes, dim=0)
            outcomes = torch.repeat_interleave(outcomes, sizes, dim=0)
            outcomes_param = [
                torch.cat(out, 0)
                for out in outcomes_param
            ]
        else:
            weights = None

        if dist == "nb":
            logprob = logprob_nb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                weight=weights
            )
        elif dist == "zinb":
            logprob = logprob_zinb_positive(outcomes,
                mu=outcomes_param[0],
                theta=outcomes_param[1],
                zi_logits=outcomes_param[2],
                weight=weights
            )
        elif dist == "normal":
            logprob = logprob_normal(outcomes,
                loc=outcomes_param[0],
                scale=outcomes_param[1],
                weight=weights
            )
        elif dist == "bernoulli":
            logprob = logprob_bernoulli_logits(outcomes,
                loc=outcomes_param[0],
                weight=weights
            )

        return (logprob.sum(0)/num).mean()

    def loss(self, outcomes, outcomes_dist_samp,
            cf_outcomes, cf_outcomes_out,
            latents_dist, cf_latents_dist,
            treatments, cf_treatments,
            covariates, kde_kernel_std=1.0):
        """
        Compute losses.
        """
        # (1) individual-specific likelihood
        indiv_spec_nllh = -outcomes_dist_samp.log_prob(
            outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1))
        ).mean()

        # (2) covariate-specific likelihood
        if self.dist_mode == "discriminate":
            if self.iteration % self.hparams["discriminator_steps"]:
                self.update_discriminator(
                    outcomes, cf_outcomes_out.detach(),
                    treatments, cf_treatments, covariates
                )

            covar_spec_nllh = self.loss_discriminator(
                self.discriminate(cf_outcomes_out, cf_treatments, covariates),
                torch.ones(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
            )
        elif self.dist_mode == "fit":
            raise NotImplementedError(
                'TODO: implement dist_mode "fit" for distribution loss')
        elif self.dist_mode == "match":
            notNone = [o != None for o in cf_outcomes]
            cf_outcomes = [o for (o, n) in zip(cf_outcomes, notNone) if n]
            cf_outcomes_out = cf_outcomes_out[notNone]

            kernel_std = [kde_kernel_std * torch.ones_like(o) 
                for o in cf_outcomes]
            covar_spec_nllh = -self.logprob(
                cf_outcomes_out, (cf_outcomes, kernel_std), dist="normal"
            )

        # (3) kl divergence
        kl_divergence = kldiv_normal(
            latents_dist.mean,
            latents_dist.stddev,
            cf_latents_dist.mean,
            cf_latents_dist.stddev
        )

        return (indiv_spec_nllh, covar_spec_nllh, kl_divergence)
    
    ## control_outcomes, control_outcomes_dist_samp, expr_outcomes, expr_outcomes_dist_samp, exp_dist,
    ## ZX, ZT, ZXT, ZX_prior_dist, ZT_prior_dist,ZXT_prior_dist, control_prior_dist, control_latents_dist,
    ### cov_constr, treatment_constr, treatments,covariates
    def loss_paired(self, control_outcomes, control_outcomes_dist_samp,
            expr_outcomes, expr_outcomes_dist_samp, exp_dist, ZX_dist, ZT_dist, ZXT_dist,
            ZX_prior_dist, ZT_prior_dist, ZXT_prior_dist, control_prior_dist,control_latents_dist,
            cov_constr, treatment_constr, treatments,
            covariates, kde_kernel_std=1.0):
        """
        Compute losses.
        """
        # (1) individual-specific likelihood
        # indiv_spec_nllh = -outcomes_dist_samp.log_prob(
        #     outcomes.repeat(self.mc_sample_size, *[1]*(outcomes.dim()-1))
        # ).mean()
        
        ## control likelihood
    
        indiv_spec_nllh_control = -control_outcomes_dist_samp.log_prob(
            control_outcomes.repeat(self.mc_sample_size, *[1]*(control_outcomes.dim()-1))
        ).mean()
            
        ## experiments likelihood
        # print("expr_outcomes_dist_samp mean shape is {}".format(expr_outcomes_dist_samp.mean.shape))
        # print("expr_outcomes shape is {}".format(expr_outcomes.shape))
        indiv_spec_nllh_experiments = -expr_outcomes_dist_samp.log_prob(
            expr_outcomes.repeat(self.mc_sample_size, *[1]*(expr_outcomes.dim()-1))
        ).mean()
            
        # covariate cross-entropy loss 
        cross_entropy_loss = nn.CrossEntropyLoss()
        cov_loss = 0.0
        for i in range(len(covariates)):
            cov_loss = cov_loss + cross_entropy_loss(cov_constr[...,i],covariates[i].squeeze())
    
        treatment_loss = cross_entropy_loss(treatment_constr.squeeze(),torch.argmax(treatments, 1))

        # (3) kl divergence
        
        ## individual KL divergence with the joint distribution
        ## KL(q(zt,zxt,zx| y, x, t))
        agg_prior = aggregate_normal_distr([ZX_prior_dist.mean, ZXT_prior_dist.mean, ZT_prior_dist.mean],\
                                           [ZX_prior_dist.stddev, ZXT_prior_dist.stddev, ZT_prior_dist.stddev] )
        
        kl_divergence_ind =  kldiv_normal(
            agg_prior[0],
            agg_prior[1],
            exp_dist.mean,
            exp_dist.stddev,
        )
        
        ## marginalized distributionn KL divergence
        ## ZX divergence
        # print("ZX_prior_dist.mean shape {}, ZX mean shape {}".format(ZX_prior_dist.mean.shape, ZX_dist[0].shape))
        covariates = torch.cat(covariates, dim=1)
        conditions = torch.cat((treatments, covariates), dim=1)
        conditions_labels = torch.unique(conditions, dim=0)
        marginal_ZX_prior =  marginalize_latent_tx(ZX_prior_dist.mean, ZX_prior_dist.stddev, conditions)
        marginal_ZX = marginalize_latent_tx(ZX_dist[0], ZX_dist[1], conditions)
        kl_divergence_X = torch.tensor(0, dtype=torch.float64, device="cuda")
        # print("marginal_ZX_prior device {}, ZX device {}".format(marginal_ZX_prior.device(), marginal_ZX.device()))
        for i in range(conditions_labels.shape[0]):
            kl_divergence_X +=kldiv_normal(
            marginal_ZX_prior[0][i],
            marginal_ZX_prior[1][i],
            marginal_ZX[0][i],
            marginal_ZX[1][i]        
            )
        
      
        ## ZT divergence
        marginal_ZT_prior =  marginalize_latent_tx(ZT_prior_dist.mean, ZT_prior_dist.stddev, conditions)
        marginal_ZT = marginalize_latent_tx(ZT_dist[0], ZT_dist[1], conditions)
        kl_divergence_T = torch.tensor(0, dtype=torch.float64, device="cuda")
        for i in range(conditions_labels.shape[0]):
            kl_divergence_T +=kldiv_normal(
            marginal_ZT_prior[0][i],
            marginal_ZT_prior[1][i],
            marginal_ZT[0][i],
            marginal_ZT[1][i]        
            )
        
        ### Z_XT divergence
    
        marginal_ZXT_prior =  marginalize_latent_tx(ZXT_prior_dist.mean, ZXT_prior_dist.stddev, conditions)
        marginal_ZXT = marginalize_latent_tx(ZXT_dist[0], ZXT_dist[1], conditions)
        kl_divergence_XT = torch.tensor(0,dtype=torch.float64, device="cuda")
        for i in range(conditions_labels.shape[0]):
            kl_divergence_XT +=kldiv_normal(
            marginal_ZXT_prior[0][i],
            marginal_ZXT_prior[1][i],
            marginal_ZXT[0][i],
            marginal_ZXT[1][i]        
            )
        
        ## control divergence
        kl_divergence_control = kldiv_normal(
            control_prior_dist[0],
            control_prior_dist[1],
            control_latents_dist.mean,
            control_latents_dist.stddev
        )

        return (indiv_spec_nllh_control, indiv_spec_nllh_experiments, cov_loss, treatment_loss,
                kl_divergence_ind, kl_divergence_X, kl_divergence_T, kl_divergence_XT,kl_divergence_control, conditions, conditions_labels)


    def forward(self, outcomes, treatments, control_outcomes, covariates,
                sample_latent=True, sample_outcome=False, detach_encode=False, detach_eval=True):
        """
        Execute the workflow.
        """
        
        ## esitimation latents for controls
        # q(z_control | y_control, x)
        control_treatment = torch.zeros(treatments.shape, dtype=torch.float32, device="cuda")
        ZX_control_constr = self.encode_ZX(control_outcomes, covariates)
        ZX_control_dist = self.distributionize(
            ZX_control_constr, dim=self.hparams["ZX_dim"], dist="normal"
        )
        
        ## q(z_t| y, T)
        ZT_control_constr = self.encode_ZT(control_outcomes, control_treatment)
        ZT_control_dist = self.distributionize(
            ZT_control_constr, dim=self.hparams["ZT_dim"], dist="normal"
        )
        
        ZXT_control_constr = self.encode_ZXT(control_outcomes, covariates, control_treatment)
        ZXT_control_dist = self.distributionize(
            ZXT_control_constr, dim=self.hparams["ZT_dim"], dist="normal"
        )
        
        ## estimation latents for experiments
        
        
        ZX_constr = self.encode_ZX(outcomes, covariates)
        ZX_dist = self.distributionize(
            ZX_constr, dim=self.hparams["ZX_dim"], dist="normal"
        )
        
        ## q(z_t| y, T)
        ZT_constr = self.encode_ZT(outcomes, treatments)
        ZT_dist = self.distributionize(
            ZT_constr, dim=self.hparams["ZT_dim"], dist="normal"
        )
        
        ZXT_constr = self.encode_ZXT(outcomes, covariates, treatments)
        ZXT_dist = self.distributionize(
            ZXT_constr, dim=self.hparams["ZT_dim"], dist="normal"
        )
        
        
        ZT_control_prior = self.encode_ZT_prior(control_treatment)
        ZT_control_prior_dist = self.distributionize(
            ZT_control_prior, dim=self.hparams["ZT_dim"], dist="normal"
        )
        
        ## p(z_xt|x,t) prior
        ZXT_control_prior = self.encode_ZXT_prior(covariates, control_treatment)
        ZXT_control_prior_dist = self.distributionize(
            ZXT_control_prior, dim=self.hparams["ZXT_dim"], dist="normal"
        )
        
        
        ## p(z_x|x) prior
        ZX_constr_prior = self.encode_ZX_prior(covariates)
        ZX_prior_dist = self.distributionize(
            ZX_constr_prior, dim=self.hparams["ZX_dim"], dist="normal"
        )

        ## p(z_t|t) prior
        ZT_constr_prior = self.encode_ZT_prior(treatments)
        ZT_prior_dist = self.distributionize(
            ZT_constr_prior, dim=self.hparams["ZT_dim"], dist="normal"
        )
        
        ## p(z_xt|x,t) prior
        ZXT_constr_prior = self.encode_ZXT_prior(covariates, treatments)
        ZXT_prior_dist = self.distributionize(
            ZXT_constr_prior, dim=self.hparams["ZXT_dim"], dist="normal"
        )
        
        control_prior_dist = aggregate_normal_distr([ZX_prior_dist.mean, ZXT_control_prior_dist.mean, ZT_control_prior_dist.mean],\
                                                   [ZX_prior_dist.stddev, ZXT_control_prior_dist.stddev, ZT_control_prior_dist.stddev])

      
        
        ## p(x|ZX)
        ZX_resample = self.sample_latent(ZX_constr[0], ZX_constr[1])
        cov_inputs =ZX_resample
        cov_constr = self.covariate_decode(cov_inputs)

        
        
        ZT_resample = self.sample_latent(ZT_constr[0], ZT_constr[1])
        ZXT_resample = self.sample_latent(ZXT_constr[0], ZXT_constr[1])
        ZTs = torch.cat([ZT_resample, ZXT_resample], dim=1)
        ZT_control_resample = self.sample_latent(ZT_control_constr[0], ZT_control_constr[1])
        ZXT_control_resample = self.sample_latent(ZXT_control_constr[0], ZXT_control_constr[1])
        ZTs_control = torch.cat([ZT_control_resample, ZXT_control_resample], dim=1)
        treatment_constr = self.intervention_decode(ZTs, ZTs_control)

        control_latents_dist_mean = torch.cat([ZT_control_dist.mean, ZXT_control_dist.mean, ZT_control_dist.mean], dim=1)
        control_latents_dist_stddev = torch.cat([ZT_control_dist.stddev, ZXT_control_dist.stddev, ZT_control_dist.stddev], dim=1)
        control_latents_dist = self.distributionize(control_latents_dist_mean, control_latents_dist_stddev)
        control_outcomes_constr_samp = self.sample_expr(control_latents_dist_mean, control_latents_dist_stddev, size=self.mc_sample_size
        )
        control_outcomes_dist_samp = self.distributionize(control_outcomes_constr_samp)

        exp_latents_dist_mean = torch.cat([ZX_dist.mean, ZXT_dist.mean, ZT_dist.mean], dim=1)
        exp_latents_dist_stddev = torch.cat([ZX_dist.stddev, ZXT_dist.stddev, ZT_dist.stddev], dim=1)
        exp_dist = self.distributionize(exp_latents_dist_mean, exp_latents_dist_stddev)
        
        expr_outcomes_constr_samp = self.sample_expr(exp_latents_dist_mean, exp_latents_dist_stddev, size=self.mc_sample_size)
        expr_outcomes_dist_samp = self.distributionize(expr_outcomes_constr_samp)

        results = [control_outcomes_dist_samp,expr_outcomes_dist_samp, exp_dist, ZX_constr, ZT_constr, ZXT_constr, ZX_prior_dist, ZT_prior_dist,ZXT_prior_dist,\
                  control_latents_dist, control_prior_dist,cov_constr, treatment_constr]      
            
        return results

    
    def update(self, expr_outcomes, treatments, contol_outcomes, covariates, adv_training=False):
        """
        Update model's parameters given a minibatch of outcomes, treatments, and covariates.
        """
        expr_outcomes, treatments, control_outcomes, covariates = self.move_inputs(
            expr_outcomes, treatments, contol_outcomes, covariates
        )
        
        if not adv_training: 
            control_outcomes_dist_samp, expr_outcomes_dist_samp, exp_dist, ZX, ZT, ZXT, \
            ZX_prior_dist, ZT_prior_dist,ZXT_prior_dist, control_latents_dist, control_prior_dist, \
            cov_constr, treatment_constr = self.forward(
                expr_outcomes, treatments, control_outcomes, covariates
            )

            indiv_spec_nllh_control, indiv_spec_nllh_experiments, cov_loss, treatment_loss,\
            kl_divergence_ind, kl_divergence_X, kl_divergence_T, kl_divergence_XT,kl_divergence_control, conditions, conditions_labels= \
            self.loss_paired(control_outcomes, control_outcomes_dist_samp, expr_outcomes, expr_outcomes_dist_samp, exp_dist,
                           ZX, ZT, ZXT, ZX_prior_dist, ZT_prior_dist,ZXT_prior_dist, control_prior_dist, control_latents_dist,
                           cov_constr, treatment_constr, treatments,covariates)

        
            perturb_X = self.permutation_distribution_X(exp_dist.mean, exp_dist.stddev,conditions)
            perturb_T = self.permutation_distribution_T(exp_dist.mean, exp_dist.stddev,conditions)
            with torch.no_grad():
                permute_T_pred = self.discriminate_T(perturb_T[:, :-1])
                permute_X_pred = self.discriminate_X(perturb_X[:, :-1])
                
            permute_T_loss = self.loss_discriminator_T(permute_T_pred, perturb_T[:, -1]) 
            permute_X_loss = self.loss_discriminator_X(permute_X_pred, perturb_X[:, -1])
            permute_loss =  0.5 * (permute_T_loss + permute_X_loss)

            indiv_spec_nllh = indiv_spec_nllh_control + indiv_spec_nllh_experiments
            covar_spec_nllh = treatment_loss + cov_loss
            kl_divergence_factored = kl_divergence_X+kl_divergence_T+kl_divergence_XT
            kl_divergence_samples= kl_divergence_control + kl_divergence_ind
            kl_divergence = kl_divergence_factored + kl_divergence_samples

            loss = (self.omega0 * indiv_spec_nllh
                + self.omega1 * covar_spec_nllh
                + self.omega2 * kl_divergence_samples
                + self.omega2 * kl_divergence_factored
                -  permute_loss   
            )

            self.optimizer_autoencoder.zero_grad()
            loss.backward()
            self.optimizer_autoencoder.step()
            self.iteration += 1
            
            return {
                "Indiv-spec NLLH": indiv_spec_nllh.item(),
                "Covar-spec NLLH": covar_spec_nllh.item(),
                "KL Divergence": kl_divergence.item(),
                "Discriminator": permute_loss.item()
            }
            
        else:
            with torch.no_grad():
                control_outcomes_dist_samp, expr_outcomes_dist_samp, exp_dist, ZX, ZT, ZXT, \
                ZX_prior_dist, ZT_prior_dist,ZXT_prior_dist, control_latents_dist, control_prior_dist, \
                cov_constr, treatment_constr = self.forward(expr_outcomes, treatments, control_outcomes, covariates)
                
                indiv_spec_nllh_control, indiv_spec_nllh_experiments, cov_loss, treatment_loss,\
                kl_divergence_ind, kl_divergence_X, kl_divergence_T, kl_divergence_XT,kl_divergence_control ,conditions, conditions_labels= \
                self.loss_paired(control_outcomes, control_outcomes_dist_samp, expr_outcomes, expr_outcomes_dist_samp, exp_dist,
                           ZX, ZT, ZXT, ZX_prior_dist, ZT_prior_dist,ZXT_prior_dist, control_prior_dist, control_latents_dist,
                           cov_constr, treatment_constr, treatments,covariates)

                
            perturb_X = self.permutation_distribution_X(exp_dist.mean, exp_dist.stddev, conditions)
            perturb_T = self.permutation_distribution_T(exp_dist.mean, exp_dist.stddev, conditions)
            # print("perturb_T shape is {}".format(perturb_T[:,:-1].shape))
            permute_T_pred = self.discriminate_T(perturb_T[:, :-1])
            permute_X_pred = self.discriminate_X(perturb_X[:, :-1])
            
            
            indiv_spec_nllh = indiv_spec_nllh_control + indiv_spec_nllh_experiments
            covar_spec_nllh = treatment_loss + cov_loss
            kl_divergence_factored = kl_divergence_X+kl_divergence_T+kl_divergence_XT
            kl_divergence_samples= kl_divergence_control + kl_divergence_ind
            kl_divergence = kl_divergence_factored + kl_divergence_samples
                
            permute_T_loss = self.loss_discriminator_T(permute_T_pred, perturb_T[:, -1]) 
            permute_X_loss = self.loss_discriminator_X(permute_X_pred, perturb_X[:, -1])
            permute_loss =  0.5 * (permute_T_loss + permute_X_loss)
            self.optimizer_discriminator.zero_grad()
            permute_loss.backward()
            self.optimizer_discriminator.step()


            return {
                "Indiv-spec NLLH": indiv_spec_nllh.item(),
                "Covar-spec NLLH": covar_spec_nllh.item(),
                "KL Divergence": kl_divergence.item(),
                "Discriminator": permute_loss.item()
            }

    def update_discriminator(self, outcomes, cf_outcomes_out,
                                treatments, cf_treatments, covariates):
        loss_tru = self.loss_discriminator(
            self.discriminate(outcomes, treatments, covariates),
            torch.ones(outcomes.size(0), device=outcomes.device)
        )

        loss_fls = self.loss_discriminator(
            self.discriminate(cf_outcomes_out, cf_treatments, covariates),
            torch.zeros(cf_outcomes_out.size(0), device=cf_outcomes_out.device)
        )

        loss = (loss_tru+loss_fls)/2.
        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item()
    
    

    def update_eval_encoder(self):
        for target_param, param in zip(
            self.exp_encoder_eval.parameters(), self.exp_encoder.parameters()
        ):
            target_param.data.copy_(param.data)
                    

    def early_stopping(self, score):
        """
        Decays the learning rate, and possibly early-stops training.
        """
        self.scheduler_autoencoder.step()

        if score > self.best_score:
            self.best_score = score
            self.patience_trials = 0
        else:
            self.patience_trials += 1

        return self.patience_trials > self.patience

    def init_outcome_emb(self):
        return MLP(
            [self.num_outcomes, self.hparams["outcome_emb_dim"]], final_act="relu"
        )

    def init_treatment_emb(self):
        
        if self.type_treatments in ("object", "bool", "category", None):
            # print(self.num_treatments, self.hparams["treatment_emb_dim"])
            return CompoundEmbedding(
                self.num_treatments, self.hparams["treatment_emb_dim"]
            )
        else:
            return MLP(
                [self.num_treatments] + [self.hparams["treatment_emb_dim"]] * 2
            )
        
        
    def init_treatment_mixed_emb(self):
        

        return MLP(
                [self.num_treatments] + [self.treatment_mixed_dim] * 2
            )
    

    def init_covariates_emb(self):
        type_covariates = self.type_covariates
        if type_covariates is None or isinstance(type_covariates, str):
            type_covariates = [type_covariates] * len(self.num_covariates)

        covariates_emb = []
        for num_cov, type_cov in zip(self.num_covariates, type_covariates):
            if type_cov in ("object", "bool", "category", None):
                covariates_emb.append(CompoundEmbedding(
                        num_cov, self.hparams["covariate_emb_dim"]
                    ))
            else:
                covariates_emb.append(MLP(
                        [num_cov] + [self.hparams["covariate_emb_dim"]] * 2
                    ))
        return covariates_emb

    def init_encoder(self):
        return MLP([self.outcome_dim+self.treatment_dim+self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["latent_dim"]],
            heads=2, final_act="relu"
        )
    
   
      
    def init_encoder_exp(self):
        return MLP([self.outcome_dim+self.covariate_dim + self.treatment_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["latent_exp_dim"]],
            heads=2, final_act="relu"
        )
    
    def init_encoder_X(self):
        return MLP([self.outcome_dim+self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZX_dim"]],
            heads=2, final_act="relu"
        )
    
    
    def init_encoder_X_prior(self):
        return MLP([self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZX_dim"]],
            heads=2, final_act="relu"
        )
    
    def init_encoder_T(self):
        return MLP([self.outcome_dim+self.treatment_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZT_dim"]],
            heads=2, final_act="relu"
        )
    
    
    
    def init_encoder_T_prior(self):
        return MLP([self.treatment_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZT_dim"]],
            heads=2, final_act="relu"
        )
    
    
    def init_encoder_XT(self):
        return MLP([self.outcome_dim+self.covariate_dim]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZT_dim"]],
            heads=2, final_act="relu"
        )
    
    
    def init_encoder_XT_prior(self):
        return MLP([self.covariate_dim ]
            + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
            + [self.hparams["ZXT_dim"]],
            heads=2, final_act="relu"
        )
          
    # def init_control_prior(self):
    #     return MLP([self.covariate_dim]
    #         + [self.hparams["encoder_width"]] * (self.hparams["encoder_depth"] - 1)
    #         + [self.hparams["ZT_dim"]],
    #         heads=2, final_act="relu"
    #     )

    def init_decoder(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return MLP([self.hparams["latent_dim"]+self.treatment_dim]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_outcomes],
            heads=heads
        )
    
#     def init_decoder_control(self):
#         if self.dist_outcomes == "nb":
#             heads = 2
#         elif self.dist_outcomes == "zinb":
#             heads = 3
#         elif self.dist_outcomes == "normal":
#             heads = 2
#         elif self.dist_outcomes == "bernoulli":
#             heads = 1
#         else:
#             raise ValueError("dist_outcomes not recognized")

#         return MLP([self.hparams["latent_dim"]]
#             + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
#             + [self.num_outcomes],
#             heads=heads
#         )
    
    
    def init_decoder_experiments(self):
        if self.dist_outcomes == "nb":
            heads = 2
        elif self.dist_outcomes == "zinb":
            heads = 3
        elif self.dist_outcomes == "normal":
            heads = 2
        elif self.dist_outcomes == "bernoulli":
            heads = 1
        else:
            raise ValueError("dist_outcomes not recognized")

        return MLP([self.hparams["latent_exp_dim"]]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [self.num_outcomes],
            heads=heads
        )
    
    def init_decoder_cov(self):
        return MLP([self.hparams["ZX_dim"]]
            + [self.hparams["decoder_width"]] * (self.hparams["decoder_depth"] - 1)
            + [np.max(self.num_covariates)],
            heads=len(self.num_covariates), final_act= "softmax"
        )
    
    
    def init_decoder_interv(self):
        return MLP([1]
            + [self.num_treatments],
            heads=1, final_act= "softmax"
        )
    
#     def init_decoder_interv_element(self):
#         return MLP([self.hparams["ZT_dim"]]
#             + [self.num_treatments],
#             heads=1, final_act= "softmax"
#         )
    
#     def init_decoder_interv_concat(self):
#         return MLP([self.hparams["ZT_dim"] + self.hparams["latent_dim"]]
#             + [self.num_treatments],
#             heads=1, final_act= "softmax"
#         )

    def init_discriminator_T(self):
        return MLP([self.hparams["ZT_dim"]+ self.hparams["ZXT_dim"]]
            + [self.hparams["discriminator_width"]] * (self.hparams["discriminator_depth"] - 1)
            + [1]
        )


    def init_decoder_interv(self):
        return MLP([1]
            + [self.num_treatments],
            heads=1, final_act= "softmax"
        )
    
    def init_decoder_interv_element(self):
        return MLP([self.hparams["ZT_dim"] +self.hparams["ZXT_dim"]]
            + [self.num_treatments],
            heads=1, final_act= "softmax"
        )
    
    def init_decoder_interv_concat(self):
        return MLP([(self.hparams["ZT_dim"] + self.hparams["ZXT_dim"])*2]
            + [self.num_treatments],
            heads=1, final_act= "softmax"
        )
    
    
    def init_decoder_interv_single(self):
        return MLP([(self.hparams["ZT_dim"] + self.hparams["ZXT_dim"])*2]
            + [self.num_treatments],
            heads=1, final_act= "softmax"
        )
    
    
    
    def init_discriminator_X(self):
        return MLP([self.hparams["ZX_dim"]+ self.hparams["ZXT_dim"]]
            + [self.hparams["discriminator_width"]] * (self.hparams["discriminator_depth"] - 1)
            + [1]
        )

    def move_input(self, input):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if isinstance(input, list):
            return [i.to(self.device) if i is not None else None for i in input]
        else:
            return input.to(self.device)

    def move_inputs(self, *inputs: torch.Tensor):
        """
        Move minibatch tensors to CPU/GPU.
        """
        return [self.move_input(i) if i is not None else None for i in inputs]

    def to_device(self, device):
        self.device = device
        self.to(self.device)

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for FCR
        """

        return self._set_hparams_(self, "")

    
    @torch.no_grad()
    def get_latent(self, outcomes, treatments, covariates):
        
        exp_constr = self.encode_exp(outcomes, covariates, treatments)
        exp_dist = self.distributionize(
            exp_constr, dim=self.hparams["latent_exp_dim"], dist="normal"
        )
        
        ZX = (exp_dist.mean[:, :self.ZX_dim], exp_dist.stddev[:, :self.ZX_dim])
        ZXT = (exp_dist.mean[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim], exp_dist.stddev[:, self.ZX_dim:self.ZX_dim+self.ZXT_dim])
        ZT = (exp_dist.mean[:, self.ZX_dim+self.ZXT_dim:], exp_dist.stddev[:, self.ZX_dim+self.ZXT_dim:])
        
        ZX_resample = self.sample_latent(ZX[0], ZX[1])
        ZXT_resample = self.sample_latent(ZXT[0], ZXT[1])
        ZT_resample = self.sample_latent(ZT[0], ZT[1])
        return ZX_resample,ZXT_resample, ZT_resample
    
    
    @torch.no_grad()
    def get_latent_presentation(self, outcomes, treatments, covariates, sample=False):
        outcomes, treatments, covariates = self.move_inputs(
            outcomes, treatments, covariates
        )
        if not sample:
            ZX, ZXT, ZT = self.get_latent(
                outcomes, treatments, covariates
            )
            
            return ZX, ZXT, ZT
        else:
            ZX, ZXT, ZT = self.get_latent(
                outcomes, treatments, covariates
            )
            ZX_resample = self.sample_latent(ZX[0], torch.ones(ZX[1].shape), device="cuda")
            ZXT_resample = self.sample_latent(ZXT[0], torch.ones(ZXT[1].shape), device="cuda")
            ZT_resample = self.sample_latent(ZT[0], torch.ones(ZT[1].shape), device="cuda")
            return ZX_resample,ZXT_resample, ZT_resample
            
            
            
    
    
    
    @torch.no_grad()
    def get_latent_prior_presentation(self, outcomes, treatments, covariates):
        outcomes, treatments, covariates = self.move_inputs(
            outcomes, treatments, covariates
        )
        ZX_constr_prior = self.encode_ZX_prior(covariates)
        ZX_prior_dist = self.distributionize(
            ZX_constr_prior, dim=self.hparams["ZX_dim"], dist="normal"
        )

        ## p(z_t|t) prior
        ZT_constr_prior = self.encode_ZT_prior(treatments)
        ZT_prior_dist = self.distributionize(
            ZT_constr_prior, dim=self.hparams["ZT_dim"], dist="normal"
        )
        
        ## p(z_xt|x,t) prior
        ZXT_constr_prior = self.encode_ZXT_prior(covariates, treatments)
        ZXT_prior_dist = self.distributionize(
            ZXT_constr_prior, dim=self.hparams["ZXT_dim"], dist="normal"
        )
        ZX_resample = self.sample_latent(ZX_prior_dist.mean, ZX_prior_dist.stddev)
        ZT_resample = self.sample_latent(ZT_prior_dist.mean, ZT_prior_dist.stddev)
        ZXT_resample = self.sample_latent(ZXT_prior_dist.mean, ZXT_prior_dist.stddev)
        
        return ZX_resample, ZT_resample, ZXT_resample
        
        

    @torch.no_grad()
    def get_control_latent(self, outcomes, covariates):
        
        control_latent_constr = self.encode_control(outcomes, covariates)
        control_latents_dist = self.distributionize(
            control_latent_constr, dim=self.hparams["latent_dim"], dist="normal"
        )
        return control_latents_dist.rsample()
    
    
    @torch.no_grad()
    def get_control_representation(self, outcomes, covariates):
        outcomes, covariates = self.move_inputs(
            outcomes, covariates
        )
        latents = self.get_control_latent(
            outcomes, covariates
        )
        return latents
