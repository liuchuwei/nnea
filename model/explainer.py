import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Explainer(object):

    def __init__(self, config, model, loader):

        self.config = config
        self.model = model
        self.loader = loader.torch_loader

    def indicate_gene_importance(self):
        indicators = self.model.gene_set_layer.get_set_indicators()
        indicators = indicators.detach().cpu().numpy()
        np.savetxt(self.config['indicator'], indicators, delimiter=",")

    def indicate_gene_set_importance(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 按类别存储重要性分数
        class_ig_scores = {}

        # 添加进度条
        total_samples = len(self.loader.dataset)
        progress_bar = tqdm(self.loader.dataset, total=total_samples, desc="Calculating IG scores")

        for R_sample, S_sample, y_sample in progress_bar:
            R_sample = R_sample.clone().detach().to(device)
            S_sample = S_sample.clone().detach().to(device)

            # 获取类别标签
            class_label = y_sample.item() if isinstance(y_sample, torch.Tensor) else y_sample

            ig_score = integrated_gradients_for_genesets(
                model=self.model,
                R=R_sample.unsqueeze(0),
                S=S_sample.unsqueeze(0),
                steps=50
            )

            # 按类别存储结果
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

def integrated_gradients_for_genesets(model, R, S, target_class=None, baseline=None, steps=50):
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
    if target_class is None:
        with torch.no_grad():
            output = model(R, S)
            target_class = torch.argmax(output, dim=1).item()

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
        if model.config['classifier'] == "attention":
            logits, _ = model.classifier(interp_es)
        else:
            logits = model.classifier(interp_es)

        # 获取目标类别的logit
        target_logit = logits[0, target_class]

        # 计算梯度
        grad = torch.autograd.grad(outputs=target_logit, inputs=interp_es)[0]
        gradients.append(grad.detach())

    # 整合梯度计算积分梯度
    gradients = torch.stack(gradients)  # (steps, 1, num_sets)
    avg_gradients = torch.mean(gradients, dim=0)  # (1, num_sets)
    ig = (es_scores - baseline) * avg_gradients  # (1, num_sets)

    return ig.squeeze(0)  # (num_sets,)