import os
import json
import random
import itertools
from datetime import datetime
from test_MLP import mlp_train


class HyperParamSearch:
    def __init__(self, search_space, save_dir, dataset_dir):
        self.search_space = search_space
        self.results = {}
        self.best_model = None
        self.best_params = None
        self.best_val_acc = 0.0
        self.dataset_dir = dataset_dir
        self.save_dir = os.path.join(save_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)

    def generate_combinations(self):
        keys = self.search_space.keys()
        values = (self.search_space[key] for key in keys)
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def save_results(self):
        best_path = os.path.join(self.save_dir, 'best_results.json')
        all_path = os.path.join(self.save_dir, 'all_results.json')
        with open(best_path, 'w') as f:
            json.dump({
                'best_size_list': self.best_size_list,
                'best_params': self.best_params,
                'best_val_acc': self.best_val_acc,
                'best_model': self.best_model
            }, f, indent=2)
        with open(all_path, 'w') as f:
            json.dump({'all_results': self.results}, f, indent=2)

    def run_search(self, num_trials=50):
        all_combos = self.generate_combinations()
        sampled_combos = random.sample(all_combos, min(num_trials, len(all_combos)))

        for i, params in enumerate(sampled_combos):
            print(f"\n=== Trial {i+1}/{len(sampled_combos)} ===")
            print("Params:", json.dumps(params, indent=2))

            trial_dir = os.path.join(self.save_dir, f"trial_{i+1}")
            os.makedirs(trial_dir, exist_ok=True)

            with open(os.path.join(trial_dir, 'hparams.json'), 'w') as f:
                json.dump(params, f, indent=2)

            # 调用mlp_train并获取验证集最佳准确率
            model, test_acc, train_scores, dev_scores, train_loss, dev_loss = mlp_train(
                size_list=[784, params['hidden_layer_num'], 10],
                act_func_list=params['act_func'],
                lambda_list=params['lambda_list'],
                initialize_method=params['initialize_method'],
                save_dir=trial_dir,
                dataset_dir=self.dataset_dir,
                Loss_function={
                    'method': 'MultiCrossEntropyLoss'},
                Optimizer={
                    'method': params['optimizer'],
                    'init_lr': params['init_lr'],
                    'init_beta': params['init_beta']},
                Scheduler={
                    'method': params['scheduler'],
                    'step_size': params['step_size'],
                    'gamma': params['gamma_Step'] if params['scheduler'] == 'StepLR' else params['gamma_Exp'],
                    'milestones': None},
                Runner={
                    'batch_size': params['batch_size'],
                    'num_epochs': 10,
                    'log_iters': params['log_iters']},
                Early_Stopping={
                    'applying': True,
                    'patience': 15,
                    'min_delta': 0.001,
                    'verbose': True}
            )
            val_acc = max(dev_scores)

            result = {
                'size_list': size_list,
                'params': params,
                'val_acc': float(val_acc),
                'model_path': trial_dir
            }
            self.results[f"trial_{i+1}"] = result

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = params
                self.best_model = trial_dir
                self.best_size_list = size_list

        self.save_results()


# 搜索空间配置

search_config = {
    'hidden_layer_size': [256],
    'act_func': ['Tanh'],
    'lambda_list': [0],
    'initialize_method': ['HeInit'],

    'scheduler': ['StepLR'],
    'step_size': [4],
    'gamma_Step': [0.5],
    'gamma_Exp': [0.5],

    'optimizer': ['MomentumGD'],
    'init_lr': [0.3],
    'init_beta': [0.9],

    'batch_size': [16],
    'log_iters': [200],
    'num_epochs': [10]
}

if __name__ == "__main__":

    save_directory = ".\\result\\test_hyperparameter_search"
    dataset_directory = '.\\dataset\\MNIST'

    searcher = HyperParamSearch(search_config, save_directory, dataset_directory)
    searcher.run_search(num_trials=30)
    print(f"\nBest Params: {json.dumps(searcher.best_params, indent=2)}")
    print(f"Best Val Acc: {searcher.best_val_acc:.2%}")
