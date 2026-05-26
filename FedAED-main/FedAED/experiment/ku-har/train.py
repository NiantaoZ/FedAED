import torch
import random
import numpy as np
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path

# 添加热力图相关的导入
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from constants import constants
from trainers.server_trainer import Server
from model.mm_models import HARClassifier
from dataloader.dataload_manager import DataloadManager

from trainers.fed_rs_trainer import ClientFedRS
from trainers.fed_avg_trainer import ClientFedAvg
from trainers.scaffold_trainer import ClientScaffold
from trainers.fed_FAED_trainer import ClientFAED
from trainers.pFedMe_trainer import ClientpFedMe
from trainers.peravg_trainer import Clientperavg
from trainers.APFL_trainer import ClientAPFL
from trainers.FedPHP_trainer import ClientFedPHP
from trainers.FedBN_trainer import ClientFedBN
from trainers.DMML_KD_trainer import ClientDMML_KD
from trainers.MASA_trainer import ClientMASA
from trainers.FedSKD_trainer import ClientFedSKD
from trainers.FedMKD_trainer import ClientFedMKD

# Define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def plot_final_confusion_matrix_heatmap(y_true, y_pred, class_names, save_path, dataset_name="KU-HAR", best_f1=0.0):
    """
    从真实标签和预测标签生成混淆矩阵热力图
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 简化标签名称
    simplified_class_names = [
        'Sit', 'Stand', 'Lie', 'Walk', 'Run',
        'UpStair', 'DownStair', 'Elevator'
    ]

    # 创建图形 - 调整为更紧凑的尺寸
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图 - 设置颜色条范围，让0.00也有轻微颜色
    sns.heatmap(
        cm_percentage, 
        annot=True, 
        fmt='.2f',
        cmap='Blues', 
        xticklabels=simplified_class_names, 
        yticklabels=simplified_class_names, 
        cbar=True,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 8},
        vmin=0.0,   # 设置最小值
        vmax=1.0,   # 设置最大值
        center=0.5  # 添加中心点，让颜色分布更均匀
    )
    
    # 简洁的标签设置
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    
    # 紧凑的刻度设置
    plt.xticks(rotation=0, fontsize=8, ha='center')
    plt.yticks(rotation=0, fontsize=8, va='center')
    
    # 紧凑布局
    plt.tight_layout(pad=1.0)

    # 先保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Heatmap saved to: {save_path}")

    # 再显示图像
    plt.show()
    
    plt.close()
    
    
def save_best_samples(best_true, best_pred, best_f1, best_fold_idx, args, class_names, save_dir):
    """
    保存最佳样本的预测结果
    """
    samples_data = {
        'true_labels': best_true,
        'pred_labels': best_pred,
        'f1_score': best_f1,
        'fold_idx': best_fold_idx,
        'dataset': args.dataset,
        'fed_alg': args.fed_alg,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'class_names': class_names,
        'model_setting': {
            'hid_size': args.hid_size,
            'local_epochs': args.local_epochs,
            'learning_rate': args.learning_rate,
            'sample_rate': args.sample_rate
        }
    }
    
    # 创建保存目录
    samples_dir = Path(save_dir).joinpath('saved_samples')
    Path.mkdir(samples_dir, parents=True, exist_ok=True)
    
    # 生成文件名
    filename = f"{args.fed_alg}_{args.dataset}_fold{best_fold_idx}_f1_{best_f1:.4f}.pkl"
    filepath = samples_dir.joinpath(filename)
    
    # 保存数据
    with open(filepath, 'wb') as f:
        pickle.dump(samples_data, f)
    
    logging.info(f'Saved best samples to: {filepath}')
    return filepath
    
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():

    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))

    parser = argparse.ArgumentParser(description='FedMultimoda experiments')
    parser.add_argument(
        '--data_dir', 
        default=path_conf["output_dir"],
        type=str, 
        help='output feature directory'
    )
    
    parser.add_argument(
        '--acc_feat', 
        default='acc',
        type=str,
        help="acc feature name",
    )
    
    parser.add_argument(
        '--gyro_feat', 
        default='gyro',
        type=str,
        help="gyro feature name",
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--sample_rate', 
        default=0.1,
        type=float,
        help="client sample rate",
    )

    parser.add_argument(
        '--global_learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--num_epochs', 
        default=300,
        type=int,
        help="total training rounds",
    )

    parser.add_argument(
        '--test_frequency', 
        default=1,
        type=int,
        help="perform test frequency",
    )
    
    parser.add_argument(
        '--local_epochs', 
        default=1,
        type=int,
        help="local epochs",
    )
    
    parser.add_argument(
        '--optimizer', 
        default='sgd',
        type=str,
        help="optimizer",
    )
    
    parser.add_argument(
        '--hid_size',
        type=int, 
        default=64,
        help='RNN hidden size dim'
    )

    parser.add_argument(
        '--mu',
        type=float, 
        default=0.01,
        help='Fed prox term'
    )
    
    parser.add_argument(
        '--fed_alg', 
        default='fed_avg',
        type=str,
        help="federated learning aggregation algorithm",
    )
    
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help="training batch size",
    )
    
    parser.add_argument(
        "--missing_modality",
        type=bool, 
        default=False,
        help="missing modality simulation",
    )
    
    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )
    
    parser.add_argument(
        "--missing_modailty_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        "--missing_label",
        type=bool, 
        default=False,
        help="missing label simulation",
    )
    
    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )
    
    parser.add_argument(
        "--missing_label_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        '--label_nosiy', 
        type=bool, 
        default=False,
        help='clean label or nosiy label')
    
    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )
    
    parser.add_argument(
        '--att', 
        type=bool, 
        default=False,
        help='self attention applied or not')
    
    parser.add_argument(
        "--en_att",
        dest='att',
        action='store_true',
        help="enable self-attention"
    )

    parser.add_argument(
        '--att_name',
        type=str, 
        default='multihead',
        help='attention name'
    )

    parser.add_argument(
        '--label_nosiy_level', 
        type=float, 
        default=0.1,
        help='nosiy level for labels; 0.9 means 90% wrong'
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="ku-har",
        help='data set name'
    )

    parser.add_argument(
        '--modality', 
        type=str, 
        default='multimodal',
        help='modality type'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # argument parser
    args = parse_args()

    # find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    save_result_dict = dict()
    
    # KU-HAR 数据集的类别名称
    ku_har_class_names = [
        'Sitting', 'Standing', 'Lying', 'Walking', 'Running', 
        'Stairs up', 'Stairs down', 'Elevator up', 'Elevator down'
    ]
    
    # 根据您的数据集调整类别名称
    if args.dataset == "ku-har":
        class_names = ku_har_class_names
    else:
        # 如果是其他数据集，使用通用名称或根据实际情况调整
        class_names = [f'Class_{i}' for i in range(constants.num_class_dict[args.dataset])]
    
    # pdb.set_trace()
    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS
    elif args.fed_alg in ['FAED']:
    	Client = ClientFAED
    elif args.fed_alg in ['pFedMe']:
    	Client = ClientpFedMe
    elif args.fed_alg in ['peravg']:
    	Client = Clientperavg
    elif args.fed_alg in ['FedPHP']:
    	Client = ClientFedPHP
    elif args.fed_alg in ['FedBN']:
    	Client = ClientFedBN
    elif args.fed_alg in ['DMML_KD']:
    	Client = ClientDMML_KD
    elif args.fed_alg in ['MASA']:
    	Client = ClientMASA
    elif args.fed_alg in ['FedSKD']:
    	Client = ClientFedSKD
    elif args.fed_alg in ['APFL']:
    	Client = ClientAPFL
    elif args.fed_alg in ['FedMKD']:
    	Client = ClientFedMKD
    
    # 用于存储最佳结果
    best_f1_score = 0.0
    best_fold_idx = 0
    best_test_true = None
    best_test_pred = None
    
    # We perform 5 fold experiments with 5 seeds
    for fold_idx in range(1, 6):
        # data manager
        dm = DataloadManager(args)
        dm.get_simulation_setting()
        # load simulation feature
        dm.load_sim_dict(fold_idx=fold_idx)
        # load client ids
        dm.get_client_ids(fold_idx=fold_idx)
        
        # set dataloaders
        dataloader_dict = dict()
        logging.info('Reading Data')
        for client_id in tqdm(dm.client_ids):
            acc_dict = dm.load_acc_feat(
                fold_idx=fold_idx,
                client_id=client_id
            )
            gyro_dict = dm.load_gyro_feat(
                fold_idx=fold_idx,
                client_id=client_id
            )

            dm.get_label_dist(
                gyro_dict, 
                client_id
            )
            shuffle = False if client_id in ['dev', 'test'] else True
            client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
            dataloader_dict[client_id] = dm.set_dataloader(
                acc_dict, 
                gyro_dict, 
                shuffle=shuffle,
                client_sim_dict=client_sim_dict,
                default_feat_shape_a=np.array([256, constants.feature_len_dict[args.acc_feat]]),
                default_feat_shape_b=np.array([256, constants.feature_len_dict[args.gyro_feat]]),
            )
        
        # number of clients, removing dev and test
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        num_of_clients = len(client_ids)
        
        # set seeds
        set_seed(8)
        # loss function
        criterion = nn.NLLLoss().to(device)
        # Define the model
        global_model = HARClassifier(
            num_classes=constants.num_class_dict[args.dataset],         # Number of classes 
            acc_input_dim=constants.feature_len_dict[args.acc_feat],    # Acc data input dim
            gyro_input_dim=constants.feature_len_dict[args.gyro_feat],  # Gyro data input dim
            en_att=args.att,                                            # Enable self attention or not
            d_hid=args.hid_size,
            att_name=args.att_name
        )
        global_model = global_model.to(device)

        # initialize server
        server = Server(
            args, 
            global_model, 
            device=device, 
            criterion=criterion,
            client_ids=client_ids
        )
        server.initialize_log(fold_idx)
        server.sample_clients(
            num_of_clients, 
            sample_rate=args.sample_rate
        )
        
        # set seeds again
        set_seed(8)

        # save json path
        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result', 
            args.fed_alg,
            args.dataset, 
            server.feature,
            server.att,
            server.model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)

        server.save_json_file(
            dm.label_dist_dict, 
            save_json_path.joinpath(f'fold{fold_idx}_label.json')
        )

        # 用于存储当前fold的最佳结果
        current_best_f1 = 0.0
        current_best_epoch = 0
        current_best_true = None
        current_best_pred = None

        # Training steps
        for epoch in range(int(args.num_epochs)):
            # define list varibles that saves the weights, loss, num_sample, etc.
            server.initialize_epoch_updates(epoch)
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                # Local training
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue
                # initialize client object
                client = Client(
                    args, 
                    device, 
                    criterion, 
                    dataloader, 
                    model=copy.deepcopy(server.global_model),
                    label_dict=dm.label_dist_dict[client_id],
                    num_class=constants.num_class_dict[args.dataset]
                )

                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control), 
                        client_control=copy.deepcopy(server.client_controls[client_id])
                    )
                    client.update_weights()

                    # server append updates
                    server.set_client_control(client_id, copy.deepcopy(client.client_control))
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result,
                        delta_control=copy.deepcopy(client.delta_control)
                    )
                else:
                    client.update_weights()
                    # server append updates
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result
                    )
                del client
            
            # logging skip client
            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')
            
            # 2. aggregate, load new global weights
            if len(server.num_samples_list) == 0: continue
            server.average_weights()
            logging.info('---------------------------------------------------------')
            server.log_classification_result(
                data_split='train',
                metric='f1'
            )
            if epoch % args.test_frequency == 0:
                with torch.no_grad():
                    # 3. Perform the validation on dev set
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_classification_result(
                        data_split='dev',
                        metric='f1'
                    )

                    # 4. Perform the test on holdout set
                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    
                    # 检查是否为当前fold的最佳结果
                    current_f1 = server.result['f1']
                    if current_f1 > current_best_f1:
                        current_best_f1 = current_f1
                        current_best_epoch = epoch
                        # 保存真实标签和预测标签（如果server有这些属性）
                        if hasattr(server, 'test_true') and hasattr(server, 'test_pred'):
                            current_best_true = server.test_true.copy() if server.test_true is not None else None
                            current_best_pred = server.test_pred.copy() if server.test_pred is not None else None
                    
                    server.log_classification_result(
                        data_split='test',
                        metric='f1'
                    )
                
                logging.info('---------------------------------------------------------')
                server.log_epoch_result(metric='f1')
            logging.info('---------------------------------------------------------')

        # 更新全局最佳结果
        if current_best_f1 > best_f1_score and current_best_true is not None and current_best_pred is not None:
            best_f1_score = current_best_f1
            best_fold_idx = fold_idx
            best_test_true = current_best_true
            best_test_pred = current_best_pred

        # Performance save code
        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()
        
        # output to results
        server.save_json_file(
            save_result_dict, 
            save_json_path.joinpath('result.json')
        )

    # 生成最终的热力图（使用所有fold中最佳的结果）
    if best_test_true is not None and best_test_pred is not None:
        heatmap_dir = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result', 
            args.fed_alg,
            args.dataset, 
            'heatmaps'
        )
        Path.mkdir(heatmap_dir, parents=True, exist_ok=True)
        
        heatmap_path = heatmap_dir.joinpath(f'best_confusion_matrix_fold{best_fold_idx}.png')
        
        plot_final_confusion_matrix_heatmap(
            y_true=best_test_true,
            y_pred=best_test_pred,
            class_names=class_names,
            save_path=heatmap_path,
            dataset_name=args.dataset,
            best_f1=best_f1_score
        )
        
        logging.info(f'Best performance: F1-score = {best_f1_score:.4f} (Fold {best_fold_idx})')
        logging.info(f'Final confusion matrix heatmap saved to: {heatmap_path}')
    else:
        logging.warning('Could not generate heatmap: test_true or test_pred is None')


# 生成最终的热力图（使用所有fold中最佳的结果）
if best_test_true is not None and best_test_pred is not None:
    heatmap_dir = Path(os.path.realpath(__file__)).parents[2].joinpath(
        'result', 
        args.fed_alg,
        args.dataset, 
        'heatmaps'
    )
    Path.mkdir(heatmap_dir, parents=True, exist_ok=True)
    
    heatmap_path = heatmap_dir.joinpath(f'best_confusion_matrix_fold{best_fold_idx}.png')
    
    # ==== 在这里添加保存样本的代码 ====
    samples_path = save_best_samples(
        best_test_true, best_test_pred, best_f1_score, best_fold_idx,
        args, class_names, heatmap_dir
    )
    # ================================
    
    # 显示并保存热力图
    plot_final_confusion_matrix_heatmap(
        y_true=best_test_true,
        y_pred=best_test_pred,
        class_names=class_names,
        save_path=heatmap_path,
        dataset_name=args.dataset,
        best_f1=best_f1_score
    )
    
    logging.info(f'Best performance: F1-score = {best_f1_score:.4f} (Fold {best_fold_idx})')
    logging.info(f'Samples saved to: {samples_path}')  # 添加这行
    logging.info(f'Heatmap saved to: {heatmap_path}')

    # Calculate the average of the 5-fold experiments
    save_result_dict['average'] = dict()
    for metric in ['f1', 'acc', 'top5_acc']:
        result_list = list()
        for key in save_result_dict:
            if metric not in save_result_dict[key]: continue
            result_list.append(save_result_dict[key][metric])
        save_result_dict['average'][metric] = np.nanmean(result_list)
    
    # dump the dictionary
    server.save_json_file(
        save_result_dict, 
        save_json_path.joinpath('result.json')
    )
