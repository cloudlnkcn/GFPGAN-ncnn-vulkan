# GFPGAN v1.4 → ncnn 权重转换（实验性）

`src/gfpgan.cpp` 按固定布局手工加载权重。转换 = 把 `GFPGANv1.4.pth`
（`params_ema`，GFPGANv1Clean 架构）拆成三份：

1. `encoder.param` + `encoder.bin` — 标准 ncnn 模型：`input.1` 进，输出
   9 个 blob：`420`（styles latent，512 维）+ 8 组跳连特征
   （`440/443`、`463/466`、`486/489`、`509/512`、`532/535`）。
   做法：torch 里把解码器之前的全图子图 trace 成 TorchScript → pnnx 转
   ncnn，并用 `ncnn2table`/重命名把输出 blob 改成上述名字。
2. `style.bin` — 纯 float32 小端顺序流（C++ `load_weights` 按此顺序读）：
   - 15 组 style conv（通道表见 `src/include/gfpgan.h` 的
     `style_conv_sizes` / `style_conv_channels`），每组依次：
     `modulated_conv.weight`(out·in·3·3) → `modulated_conv.modulation.weight`(512·512)
     → `modulated_conv.modulation.bias`(512) → `conv.weight`(out·in·3·3) → `conv.bias`(out)
   - 8 组 to_rgb（`to_rgb_sizes` / `to_rgb_channels`），每组依次：
     `modulated_conv.weight`(3·in·1·1) → `modulation.weight`(512·512) →
     `modulation.bias`(512) → `bias`(3)
   - `const_input`：4·4·512（StyleGAN 常量输入）
3. `yolov5-blazeface.param/bin`、`real_esrgan.param/bin` — 人脸检测与背景
   放大模型，v1.4 不影响，沿用现有包内文件。

## dump_layout.py

在装有 torch + gfpgan 的机器上运行（本仓库 CI 不含这些依赖）：

```bash
pip install torch gfpgan
python tools/convert/dump_layout.py --pth GFPGANv1.4.pth
```

输出按 15 组 style conv / 8 组 to_rgb / const_input 分桶打印
`params_ema` 的键与形状，并做尺寸校验（总浮点数与 C++ 表格逐段对账）。
**校验通过后**再用 `--export out_dir` 导出 style.bin；encoder 子图的
pnnx 导出随后补充。
