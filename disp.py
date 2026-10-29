


import os
import matplotlib
import matplotlib.pyplot as plt


def plot_and_save(train_result, val_result, save_path):
    """
    画训练/验证的 loss 和 acc 曲线，并保存为图片。

    参数:
        train_result: dict, 形如 {'loss': [...], 'acc': [...]}
        val_result:   dict, 形如 {'loss': [...], 'acc': [...]}
        save_path:    str,  图片保存路径，如 'result/curve.png'
    """
    # 1. 检查数据
    train_loss = train_result.get('loss', [])
    train_acc = train_result.get('acc', [])
    val_loss = val_result.get('loss', [])
    val_acc = val_result.get('acc', [])

    if len(train_loss) == 0:
        print("[plot_and_save] train_result 为空，跳过画图")
        return

    epochs = range(1, len(train_loss) + 1)

    # 2. 创建画布：1 行 2 列
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- 左图：loss ----
    ax = axes[0]
    ax.plot(epochs, train_loss, 'o-', label='train loss', color='tab:blue')
    if len(val_loss) == len(train_loss):
        ax.plot(epochs, val_loss, 's-', label='val loss', color='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- 右图：acc ----
    ax = axes[1]
    ax.plot(epochs, train_acc, 'o-', label='train acc', color='tab:blue')
    if len(val_acc) == len(train_acc):
        ax.plot(epochs, val_acc, 's-', label='val acc', color='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. 标注最佳结果（可选，对精度曲线有用）
    if len(val_acc) > 0 and max(val_acc) == val_acc[-1]:
        best_ep = val_acc.index(max(val_acc)) + 1
        axes[1].annotate(
            f"best={max(val_acc):.4f}",
            xy=(best_ep, max(val_acc)),
            xytext=(best_ep, max(val_acc) - 0.05),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red'
        )

    fig.tight_layout()

    # 4. 保存
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)          # 防止多次调用累积 figure，占内存
    print(f"[plot_and_save] 图片已保存到: {save_path}")







