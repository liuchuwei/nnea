import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Explainer(object):

    def __init__(self, config, model, loader):

        self.config = config
        self.model = model
        self.loader = loader.torch_loader
        self.gene = loader.gene

    def indicate_gene_importance(self):
        indicators = self.model.gene_set_layer.get_set_indicators()
        indicators = indicators.detach().cpu().numpy()
        indicators = np.vstack([self.gene, indicators])
        np.savetxt(self.config['indicator'], indicators, delimiter=",", fmt="%s")

    def indicate_gene_set_importance(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 按类别存储重要性分数
        class_ig_scores = {}

        # 添加进度条
        total_samples = len(self.loader.dataset)
        progress_bar = tqdm(self.loader.dataset, total=total_samples, desc="Calculating IG scores")

        for batch in progress_bar:

            R_batch = batch[0].to(device)
            S_batch = batch[1].to(device)
            N_batch = batch[2].to(device)

            # 获取类别标签
            if self.config['task'] in ['classification']:
                y_batch = batch[2]
            elif self.config['task'] in ['regression']:
                y_batch = batch[2]
            elif self.config['task'] in ['cox']:
                event_batch = batch[3]  # 生存分析的事件标签
                # time_batch = batch[2]  # 如果需要时间信息可以启用


            R_sample = R_batch.unsqueeze(0)
            S_sample = S_batch.unsqueeze(0)
            N_sample = N_batch.unsqueeze(0)

            # 确定存储键名
            if self.config['task'] in ['classification']:
                class_label = y_batch.item()
            elif self.config['task'] in ['regression']:
                class_label = 'regression'
            elif self.config['task'] in ['cox']:
                class_label = f"event_{event_batch.item()}"  # 按事件状态分组
            elif self.config['task'] in ['autoencoder']:
                class_label = 'autoencoder'


            if self.config['task'] in ['autoencoder']:
                ig_score = integrated_gradients_for_genesets(
                    model=self.model,
                    R=R_sample,
                    S=S_sample,
                    N=N_sample,
                    task_type=self.config['task'],  # 传递任务类型
                    steps=50
                )
            else:
                ig_score = integrated_gradients_for_genesets(
                    model=self.model,
                    R=R_sample,
                    S=S_sample,
                    N=N_sample,
                    task_type=self.config['task'],  # 传递任务类型
                    steps=50
                )
            # 按类别/任务类型存储结果
            if class_label not in class_ig_scores:
                class_ig_scores[class_label] = []
            class_ig_scores[class_label].append(ig_score.cpu().detach().numpy())

        # 计算每个类别的平均重要性分数
        class_avg_ig = {}
        for class_label, scores in class_ig_scores.items():
            class_avg_ig[class_label] = np.mean(scores, axis=0)

        # 保存结果（这里需要根据你的实际需求实现）
        # 例如：np.save(self.config['class_avg_ig'], class_avg_ig)
        print("Class average IG scores calculated:", class_avg_ig.keys())
        geneset_importance = pd.DataFrame(class_avg_ig)
        geneset_importance.to_csv(self.config['geneset_importance'], index=False)

    def explain(self):

        self.model.load_state_dict(torch.load(self.config['check_point']))
        # self.model.load_state_dict(torch.load(self.config['model_pt']))

        # explain gene importance
        self.indicate_gene_importance()

        # explain geneset
        self.indicate_gene_set_importance()

def integrated_gradients_for_genesets(model, R, S, N=None, task_type=None, target_class=None, baseline=None, steps=50):
    """
    使用积分梯度解释基因集重要性

    参数:
    model -- 训练好的NNEA模型
    R -- 基因表达数据 (1, num_genes)
    S -- 基因排序索引 (1, num_genes)
    target_class -- 要解释的目标类别 (默认使用模型预测类别)
    baseline -- 基因集的基线值 (默认全零向量)
    steps -- 积分路径的插值步数

    返回:
    ig -- 基因集重要性分数 (num_sets,)
    """
    # 确保输入为单样本
    assert R.shape[0] == 1 and S.shape[0] == 1, "只支持单样本解释"

    model.eval()  # 设置为评估模式

    # 计算样本的富集分数 (es_scores)
    with torch.no_grad():
        es_scores = model.gene_set_layer(R, S)  # (1, num_sets)

    # 确定目标类别
    if task_type in ['classification']:
        if target_class is None:
            with torch.no_grad():
                output = model(R, S)
                target_class = torch.argmax(output, dim=1).item()
    else:  # 回归或生存分析
        target_class = None  # 不使用特定类别

    # 设置基线值
    if baseline is None:
        baseline = torch.zeros_like(es_scores)

    # 生成插值路径 (steps个点)
    scaled_es_scores = []
    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = baseline + alpha * (es_scores - baseline)
        scaled_es_scores.append(interpolated)

    # 存储梯度
    gradients = []

    # 计算插值点梯度
    for interp_es in scaled_es_scores:
        interp_es = interp_es.clone().requires_grad_(True)

        # 根据分类器类型处理输出
        if task_type in ['autoencoder']:
            reconstructed = model.decoder(interp_es)
            loss = torch.nn.functional.mse_loss(reconstructed, N)
            # 使用负损失表示改善重构效果的影响
            target_logit = -loss

        elif task_type in ['classification']:
            logits = model.classifier(interp_es)
            target_logit = logits[0, target_class]
        else:  # 回归或生存分析
            target_logit = logits.squeeze()  # 使用整个输出

        # 计算梯度
        grad = torch.autograd.grad(outputs=target_logit, inputs=interp_es)[0]
        gradients.append(grad.detach())

    # 整合梯度计算积分梯度
    gradients = torch.stack(gradients)  # (steps, 1, num_sets)
    avg_gradients = torch.mean(gradients, dim=0)  # (1, num_sets)
    ig = (es_scores - baseline) * avg_gradients  # (1, num_sets)

    return ig.squeeze(0)  # (num_sets,)