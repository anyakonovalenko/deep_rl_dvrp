def get_experiment_config(self):
    return {
        "training": {
            "env": "GridWorldEnv-v0",
            "run": "PPO",
            "config": {
                "gamma": 0.995,
                "lambda": 0.95,
                "clip_param": 0.2,
                "kl_coeff": 1.0,
                "num_sgd_iter": 10,
                "lr": 0.0001,
                "sample_batch_size": 1000,
                "sgd_minibatch_size": 1024,
                "train_batch_size": 35000,
                "use_gae": False,
                "num_workers": (self.num_cpus - 1),
                "num_gpus": self.num_gpus,
                "batch_mode": "complete_episodes",
                "observation_filter": "NoFilter",
                "model": {
                    "custom_model": "action_mask",
                    "fcnet_hiddens": [512, 512],

                    #!!!! Setup of the feature net (used to encode observations into feature (latent) vectors).
                    # "feature_net_config": {
                    #     "fcnet_hiddens": [],
                    #     "fcnet_activation": "relu",
                    # },
                },
                "vf_share_layers": False,
                "entropy_coeff": 0.01,
            }
        }
    }

# actor_hiddens
# critic_hiddens