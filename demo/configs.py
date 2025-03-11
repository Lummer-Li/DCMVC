def get_config(args):
    """Configure hyperparameters for different datasets
    
    Args:
        args (argparse.Namespace): Command line arguments or base config
        
    Returns:
        Updated args with dataset-specific parameters
    
    Usage:
        >>> args = get_config(args)
    """
    
    # Configuration templates for supported datasets
    dataset_configs = {
        'synthetic3d': {
            'input_dim': [3, 3, 3],
            'embedding_dims': [1024, 1024, 16],
            'cluster_dims': 256,
            'temperature': 0.5,
            'batch_size': 180,
            'lr': 1e-4,
            'alpha': 0.1,
            'beta': 1.0,
            'gamma': 1.0
        },
        # Add new dataset templates here
        # 'mnist': {...}
    }

    try:
        # Update args with dataset-specific config
        config = dataset_configs[args.dataset]
        for key, value in config.items():
            setattr(args, key, value)
            
    except KeyError:
        raise ValueError(f"Unsupported dataset: {args.dataset}. "
                         f"Supported options: {list(dataset_configs.keys())}")

    return args