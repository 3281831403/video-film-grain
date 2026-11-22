# video-film-grain

## 题面

给数字素材加胶片颗粒后，暗部全是噪点、高光还冒出黑点：噪声用 random，每帧不可复现；亮度权重固定 1，暗部噪点过重；颗粒尺寸不生效，永远单像素；叠加不钳制，高光回绕成黑点；时域混合用固定权重，闪烁；高频能量用均值，黑帧也有值；尺寸不符不报错，越界；强度越界不校验，画面发灰；批量共用噪声层，帧之间串；seed 类型不校验，哈希报错。grain/noise.py 是核心，tests/test_noise.py 15个红。要修的是：用显式 seed 的确定性哈希替代 random、按中间调计算亮度权重、让 size 真正控制颗粒尺寸、叠加后钳制到 0..1、按时域权重混合、用高频差分度量颗粒、校验尺寸/强度/NaN/seed、为每帧独立生成噪声层并支持批量原子回滚。

## 运行方式

```powershell
cd D:\gsb-20261119\video-film-grain
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败全部来自 `grain/noise.py` 中 8 个桩的 `NotImplementedError`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_noise.py` 是评测基准，**不可修改**；只允许修改 `grain/noise.py`。实现只使用 Python 标准库（`math`）与 pytest，离线运行，不使用随机数；帧是二维灰度 `list[list[float]]`，灰度与强度均使用 `0..1`，数值比较容差为 `1e-6`，异常类为 `GrainError`。
