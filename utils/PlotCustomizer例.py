import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

def plot_histogram_with_scale_toggle(data, bins=30):
    """
    ヒストグラムをプロットし、ボタンでスケールをlinearとlogで切り替えるインタラクティブなプロットを作成します。

    Parameters:
    ----------
    data : array-like
        ヒストグラムをプロットするデータ
    bins : int, optional
        ヒストグラムのビン数（デフォルトは30）
    """
    # プロット作成
    fig, ax = plt.subplots(figsize=(8, 6))

    # 初期のヒストグラムをプロット
    n, bins, patches = ax.hist(data, bins=bins, histtype='step', color='blue', lw=2)
    ax.set_title("Histogram (Linear Scale)", fontsize=14)

    # Linear と Log スケールを切り替える関数
    def toggle_scale(event):
        if ax.get_yscale() == 'linear':
            ax.set_yscale('log')
            ax.set_title("Histogram (Log Scale)", fontsize=14)
        else:
            ax.set_yscale('linear')
            ax.set_title("Histogram (Linear Scale)", fontsize=14)
        
        # 再描画
        fig.canvas.draw()

    # ボタンの作成
    ax_button = plt.axes([0.75, 0.01, 0.2, 0.075])  # ボタンの位置
    button = Button(ax_button, "Toggle Scale")
    button.on_clicked(toggle_scale)  # ボタンが押されたときに切り替え関数を実行

    plt.show()

# 使用例
data = np.random.gamma(2, 2, size=1000)  # ヒストグラム用のデータ
plot_histogram_with_scale_toggle(data)


def interactive_plot_function(func, param_range, param_name, initial_value=1.0, *args, **kwargs):
    """
    任意の関数をスライダーで動的に制御してプロットする汎用ツール。

    Parameters:
    -----------
    func : function
        動的にプロットを更新したい関数。
        この関数は `ax`, `param_value`, その他の引数を受け取る必要がある。
    param_range : tuple
        スライダーの最小値と最大値の範囲 (min, max)。
    param_name : str
        スライダーのラベルとして使用するパラメータ名。
    initial_value : float, optional
        初期値。デフォルトは1.0。
    *args : tuple
        `func` に渡す追加の引数。
    **kwargs : dict
        `func` に渡す追加のキーワード引数。
    """
    # プロット設定
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)  # スライダー用のスペースを作る

    # 初期のプロットを描画
    func(ax, initial_value, *args, **kwargs)

    # スライダーの設定
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, param_name, param_range[0], param_range[1], valinit=initial_value)

    # スライダーの値が変更されたときにプロットを更新する関数
    def update(val):
        ax.clear()  # 以前のプロットをクリア
        func(ax, slider.val, *args, **kwargs)  # 関数を再実行してプロットを更新
        plt.draw()

    # スライダーにイベントをバインド
    slider.on_changed(update)

    # プロットの表示
    plt.show()

# 任意の関数: 例えば、sin(x)のプロットを動的に更新する関数
def plot_sine(ax, scale_factor, *args, **kwargs):
    """
    スケーリングファクターで変化する sin(x) のプロットを描画。
    """
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * scale_factor  # scale_factor に応じて変化
    ax.plot(x, y, label=f'Sine curve scaled by {scale_factor}')
    ax.set_title(f"Scale Factor: {scale_factor}")
    ax.legend()

# 使用例: `interactive_plot_function` を使って、sin(x)のスケールをスライダーで変更
interactive_plot_function(plot_sine, param_range=(0.1, 10.0), param_name='Scale Factor', initial_value=1.0)
