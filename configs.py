def get_config(args):
    if args.dataset == 'synthetic3d':
        args.input_dim = [3, 3, 3]
        args.embedding_dims = [1024, 1024, 16]         # [1024, 1024, 128]
        args.cluster_dims = 256                         # 512
        args.temperature = 0.5
        args.batch_size = 180                            # 128
        args.lr = 0.0001
        args.alpha = 0.1
        args.beta = 1.0
        args.gamma = 1.0

    return args