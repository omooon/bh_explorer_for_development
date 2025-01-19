import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

def interactive_plot_function(func, ax, param_range, param_name, initial_value=1.0, **kwargs):
    """
    任意の関数をスライダーで動的に制御してプロットする汎用ツール。

    Parameters:
    -----------
    func : function
        動的にプロットを更新したい関数。
        この関数は `ax`, `param_value`, その他の引数を受け取る必要がある。
    ax : Axes
        プロットを描画するためのmatplotlibの軸オブジェクト。
    param_range : tuple
        スライダーの最小値と最大値の範囲 (min, max)。
    param_name : str
        スライダーのラベルとして使用するパラメータ名。
    initial_value : float, optional
        初期値。デフォルトは1.0。
    **kwargs : dict
        `func` に渡す追加のキーワード引数。
    """
    # Create space for sliders
    plt.subplots_adjust(bottom=0.2)

    # 初期のプロットを描画
    func(ax, initial_value, **kwargs)

    # スライダーの設定
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, param_name, param_range[0], param_range[1], valinit=initial_value)

    # スライダーの値が変更されたときにプロットを更新する関数
    def update(val):
        ax.clear()  # 以前のプロットをクリア
        func(ax, slider.val, **kwargs)  # 関数を再実行してプロットを更新
        plt.draw()

    # スライダーにイベントをバインド
    slider.on_changed(update)

    # プロットの表示
    plt.show()

# 使用例:
def plot_function(ax, param_value, **kwargs):
    """
    param_value に応じてプロットを更新する例。
    """
    x = range(0, 100)
    y = [xi**param_value for xi in x]
    ax.plot(x, y)
    ax.set_title(f"Plot with parameter value {param_value}")

# プロットの設定
fig, ax = plt.subplots(figsize=(8, 6))
interactive_plot_function(plot_function, ax, param_range=(0.1, 3.0), param_name="Exponent", initial_value=1.0)
