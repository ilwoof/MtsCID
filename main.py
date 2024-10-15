import copy
import argparse
from torch.backends import cudnn
from utils.utils import *

# ours
from solver import Solver

def str2bool(str_v):
    return str_v.lower() in ['true']

def main(config_setting):
    # set_seed(42)
    cudnn.benchmark = True
    if not os.path.exists(config_setting.model_save_path):
        mkdir(config_setting.model_save_path)

    result_list = {key: [] for key in metric_list}
    solver = Solver(vars(config_setting))

    for i in range(config_setting.run_times):
        # To ensure that the model parameters are re-initialized before each round.
        solver.model_init(vars(config_setting))
        print(f"--------------------------- Round {i+1} -----------------------------")
        if not config_setting.test_only:
            solver.train()
        eval_results = solver.test(vars(config_setting))

        for key, value in eval_results.items():
            if key in result_list:
                result_list[key].append(value)

    final_eval_results = {key: '-' for key in metric_list}

    for key, value in result_list.items():
        if key in final_eval_results:
            final_eval_results[key] = f"{np.mean(result_list[key]):.4f}±{np.std(result_list[key]):.4f}"
    dump_final_results(vars(config_setting), final_eval_results)

    print(f"----------------------{config_setting.dataset} Evaluation Results-----------------------")
    print(f"{Color.CYAN}pc-a: {np.mean(result_list['pc_adjust']):.4f}{result_list['pc_adjust']}{Color.RESET}")
    print(f"{Color.CYAN}rc-a: {np.mean(result_list['rc_adjust']):.4f}{result_list['rc_adjust']}{Color.RESET}")
    print(f"{Color.CYAN}f1-a: {np.mean(result_list['f1_adjust']):.4f}{result_list['f1_adjust']}{Color.RESET}")

    del solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--framework', nargs="+", type=str, default=['MtsCID'])
    parser.add_argument('--test_only', default=False, action="store_true")

    # Data setting
    parser.add_argument('--dataset', type=str, default='NIPS_TS_Water')
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='./data/NIPS_TS_GECCO/')

    # Model setting
    parser.add_argument('--input_c', type=int, default=9)
    parser.add_argument('--output_c', type=int, default=9)
    parser.add_argument('--d_model', type=int, default=9)
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--encoder_layers', type=int, default=1, help="The number of encoder layers")
    parser.add_argument('--branches_group_embedding', type=str, default='False_False', choices=['True_True', 'True_False', 'False_True', 'False_False'], help="The parameter is used only when conv1d is employed in the encoder layer")
    parser.add_argument('--multiscale_kernel_size', nargs="+", type=int, default=[5], help="The parameter is used when conv1d is employed in the encoder layer")
    parser.add_argument('--multiscale_patch_size', nargs="+", type=int, default=[10, 20], help="The parameter is used when multi-attention is employed in the encoder layer")
    parser.add_argument('--branch1_networks', nargs="+", type=str, default=['fc_linear', 'intra_fc_transformer', 'multiscale_ts_attention'])
    parser.add_argument('--branch1_match_dimension', type=str, default='first', choices=['none', 'first', 'middle', 'last'])
    parser.add_argument('--branch2_networks', nargs="+", type=str, default=['multiscale_conv1d', 'inter_fc_transformer'])
    parser.add_argument('--branch2_match_dimension', type=str, default='first', choices=['none', 'first', 'middle', 'last'])

    parser.add_argument('--decoder_networks', nargs="+", type=str, default=['linear'])
    parser.add_argument('--decoder_layers', type=int, default=1, help="The number of encoder layers")
    parser.add_argument('--decoder_group_embedding', type=str, default='False', choices=['True', 'False'])

    parser.add_argument('--embedding_init', type=str, default='normal')
    parser.add_argument('--memory_guided', type=str, default='sinusoid')

    parser.add_argument('--aggregation', type=str, default='normal_mean', choices=['normal_mean', 'mean', 'max', 'harmonic_mean', 'harmonic_max'])  # for recon loss, max is better than mean

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    # Training setting
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--peak_lr', type=float, default=2e-3)
    parser.add_argument('--end_lr', type=float, default=5e-5)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)

    # Device parameter
    parser.add_argument('--device', type=str, default="cuda:0")

    # Loss weight hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0)

    # Parameters setting for evaluation
    parser.add_argument('--threshold_setting', type=str, default='optimal', choices=['preset', 'optimal'])
    parser.add_argument('--anomaly_ratio', type=float, default=1.0, help="The parameter is used when threshold_setting is set as 'preset'")
    parser.add_argument('--run_times', type=int, default=5, help="The number of times to run for evaluating the result")

    # debug parameters
    parser.add_argument('--plot_data', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--anomaly_only', type=str, default='False', choices=['True', 'False'])

    default_args = parser.parse_args()

    for frame_work in default_args.framework:

        config = copy.copy(default_args)

        config.framework = frame_work

        print(f'--------------------- Framework: {frame_work} -------------------')

        if ('conv1d' not in config.branch1_networks) or (len(config.branch1_networks) < 2):
            updated_group_embedding = 'False'
        else:
            updated_group_embedding = config.branches_group_embedding.split('_')[0]

        if ('conv1d' not in config.branch2_networks) or (len(config.branch2_networks) < 2):
            config.branches_group_embedding = updated_group_embedding + '_False'
        else:
            config.branches_group_embedding = updated_group_embedding + '_' + config.branches_group_embedding.split('_')[1]

        config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        args = vars(config)
        # print('--------------------------- Parameters Setting-------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        # print('--------------------------- End -------------------------------')
        main(config)
