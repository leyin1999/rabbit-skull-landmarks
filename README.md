# 方案 B · 颅骨点云标志点识别

输入结构光相机导出的颅顶点云、带有人工标注的参考模型。程序用 Open3D 进行 FPFH + RANSAC 粗配准和两级点到面 ICP 精配准，再把参考标志点转换到扫描点云坐标系。参考模板可重复使用。当前实现用于离线检查打印件的扫描数据。

## 仓库内容

本仓库包含识别程序、模拟基准脚本和绘图源码。主程序用法见“准备输入”和“运行”。已有报告中的原始 CT、模型文件、模拟点云及实验输出存放在本地项目中。下载对应 STL 后，`benchmark_rabbit.py` 可独立生成基础模拟数据。重跑针对已标注候选点的裁剪、遮挡和叠加图实验时，需要从本地项目复制对应的标注和历史结果到 `rabbit_benchmark/`。

绘图和相关复测脚本需额外安装 `requirements-benchmark.txt`。

## 公开兔颅模型测试

本项目使用汇报中同编号的公开兔颅模型进行模拟验证。来源是 SMART Neuro at Penn Vet 的 [NIH 3D 模型 3DPX-004799 v3](https://3d.nih.gov/entries/3DPX-004799?version=3)，许可为 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。下载该版本的 STL 后，将其放在 `rabbit_benchmark/krakoski_1-nih3d.stl`。参考点云和模拟扫描是从该 STL 派生的，沿用同一许可。

程序从网格模拟不同视野的深度采样，施加已知刚体位姿和随机噪声，并从预设位姿计算三个几何测试点的预期坐标。测试点名为 `test_A`、`test_B`、`test_C`，三者均为选取的表面点，未赋予解剖名称。源 STL 未提供明确单位，测试将一个模型单位按一毫米传给识别脚本。

| 模拟场景 | 输出 | 已输出点的最大位置误差（模型单位） |
| --- | --- | ---: |
| 完整颅顶视野 | 3/3 | 0.006 |
| 缩小视野 | 3/3 | 0.017 |
| 一个测试点落在视野外 | 2/3，视野外点为 `null` | 0.007 |

在安装依赖后，可从同一 STL 重新生成模拟点云和结果。每组的预设位姿、配准指标和逐点误差见输出目录中的 `summary.json`。

```powershell
python benchmark_rabbit.py rabbit_benchmark/krakoski_1-nih3d.stl rabbit_benchmark
```

接入 Gemini 215 时，将模拟扫描换成打印件的实扫点云，使用打印文件的实际比例，并在参考模型上确认和标注真实 Bregma、Lambda。上述测试误差衡量同源网格在模拟扫描下的配准与坐标映射。

### Bregma 和 Lambda 候选点

在 [NIH 3D 下载页](https://3d.nih.gov/entries/download/4799/3) 的 Supplemental Files 中取得同一标本的 `KrakoskiBean_1.zip`。将连续 CT 切片的颅顶骨面与 STL 配准，再沿 STL 外表面向内约 0.8 个模型单位查看 CT 灰度。根据冠状缝与矢状缝的疑似交会处标记 Bregma 候选点，根据后部两侧疑似人字缝向中线汇合的位置标记 Lambda 候选点。Lambda 附近的缝线较模糊，两个坐标均待独立复核。参照 [兔颅骨标注图](https://pmc.ncbi.nlm.nih.gov/articles/PMC9804214/#jez23171-fig-0002) 和 [兔颅骨 Bregma 与 Lambda 的定义](https://kopfinstruments.com/app/uploads/2016/06/CarrierV4.pdf)。

标注图、输入和输出分别保存在 `rabbit_benchmark/bregma_lambda_candidates.png`、`rabbit_benchmark/bregma_lambda_candidates.json` 和 `rabbit_benchmark/bregma_lambda_full_result.json`。两点在完整视野模拟扫描中均得到坐标。候选点的映射误差可用预设位姿计算，该误差衡量已选坐标的转换精度。STL 原始比例和标注的解剖准确性仍需核实。

```powershell
python recognize_landmarks.py --scan rabbit_benchmark/full_scan.ply --scan-unit mm `
  --reference rabbit_benchmark/reference_dorsal.ply --reference-unit mm `
  --landmarks rabbit_benchmark/bregma_lambda_candidates.json `
  --output rabbit_benchmark/bregma_lambda_full_result.json
```

## 准备输入

1. 导出 Gemini 215 的三维点云（PLY、PCD 或 XYZ），保留颅顶有效区域，裁掉桌面、夹具和离群点。建议让扫描覆盖前后两个标志点及其周围的曲率变化。
2. 准备与扫描视野对应的参考颅顶区域。可使用打印件的原始模型，接受 STL / OBJ 网格或 PLY / PCD / XYZ 点云。裁剪时保留原坐标系和物理尺寸。整头网格与单面扫描的重叠会偏低，优先裁剪参考区域。
3. 在同一参考模型坐标系内人工标注 Bregma、Lambda 或其他需要的点，将点名及 XYZ 保存为 UTF-8 JSON。坐标使用参考文件自身单位。下面的数字仅演示格式，需替换为实测标注。

```json
{
  "Bregma": [10.0, 20.0, 30.0],
  "Lambda": [15.0, 20.0, 27.0]
}
```

## 运行

使用 Python 3.12。以下命令以 Windows PowerShell 为例，点云单位由导出设置决定。

```powershell
python -m pip install -r requirements.txt
python recognize_landmarks.py --scan scan.ply --scan-unit mm `
  --reference dorsal_reference.stl --reference-unit mm `
  --landmarks reference_landmarks.json --output result.json
```

以米保存的点云传入 `--scan-unit m`。初始位姿已知时，可传入 `--initial-pose pose.json`。文件内容是将扫描坐标转换到参考坐标的 4×4 刚体矩阵，平移量使用毫米，采用列向量约定。该模式从输入位姿直接开始 ICP。

## 读取结果

- `accepted` 表示配准通过质量筛查，且所有标志点均通过局部视野覆盖检查。进程返回 0。
- `partial` 表示仅部分标志点通过检查。未通过的点的 `xyz_mm` 为 `null`。进程返回 2。
- `rejected` 表示配准失败、重叠或残差未达阈值、候选结果有歧义，或没有标志点通过局部覆盖检查。进程返回 2。输入或写入错误返回 1，请检查终端错误信息。
- `reference_to_scan_mm` 将毫米制参考坐标转换到毫米制扫描坐标。每点的 `nearest_scan_distance_mm` 是到最近实扫点的距离。
- 单点状态 `no_nearby_scan_point` 表示搜索半径内缺少实扫点。`outside_scan_coverage` 表示候选点未通过边界或局部包围检查。`local_scan_hole` 表示候选点位于邻近扫描点围住的采样缺口中。

默认用 1 mm 和 0.5 mm 体素做配准。质量筛查在 1.5 mm 匹配距离下计算双向重叠率与扫描至参考的内点 RMSE。默认扫描覆盖率至少 65%，参考覆盖率至少 35%，内点 RMSE 至多 0.9 mm。局部覆盖检查将标志点 2 mm 内的扫描点投影到参考表面的局部切平面。投影点必须围住标志点，等价于相邻方向之间的最大角度小于 180°。这些值是调试起点。颅顶平滑或近似对称时，相近的残差可能对应不同位姿，应在打印件上用独立人工标注计算各点的三维欧氏误差。配准残差只衡量表面拟合。

### 局部采样孔洞

对通过局部包围检查的候选点，程序将两倍搜索半径内的扫描点投影到同一切平面，再构建 Delaunay 三角网。局部采样尺度取各投影点到第二近邻距离的中位数。最长边超过该尺度 2.25 倍的三角形表示缺少表面采样支持。若包住候选点的这类三角形组成封闭缺口，则返回 `local_scan_hole`。通过三角网边缘与外部相连的缺口交给外边界检查。该方法的检测尺度随点云密度变化。

运行 `python benchmark_occlusion_sweep.py` 可复测 11 个圆形遮挡半径，并与保存的修改前结果比较。识别程序只接收挖空后的点云、参考模型和标注。孔洞参数用于构造测试数据及作图。

### 自动估计视野边界

默认从扫描点云的三维凸包估计外边界，保留面积至少占凸包表面积 1% 且有足够邻近采样点支持的面。边界容差取点云最近邻距离中位数的四分之一。估计边界用于剔除外侧候选点，局部切平面检查用于评估候选点周围的采样覆盖。点云近似共面或缺少足够支持的凸包面时，使用局部检查。凹形边界和不同扫描密度下的遮挡检测仍需实扫验证。

运行 `python benchmark_crop_sweep.py` 可重跑 21 个裁剪位置。基准脚本用真实平面生成输入点云和评估结果，识别程序只接收裁剪后的点云、参考模型和参考标注。`coverage_boundary_source: point_cloud_hull` 表示使用了自动估计的边界。

### 使用已知裁剪边界

有裁剪平面或已标定的视野平面时，传入 `--coverage-planes coverage.json`。每个平面用 `[nx, ny, nz, d]` 表示，法向朝向保留区域。坐标系为扫描坐标系，距离单位为毫米。候选点须满足所有平面的 `nx*x + ny*y + nz*z + d >= 0`，也须通过局部覆盖检查。程序将平面系数按法向长度归一化。`tolerance_mm` 默认为 0，可给边界处的配准偏差留出小量容差。启用任一种边界检查时，结果中的 `coverage_boundary_distance_mm` 是到限制最严的平面的有符号距离，保留侧为正。

例如，扫描坐标中保留 `Z >= 41.5` 时，文件可写为：

```json
{"planes": [[0, 0, 1, -41.5]], "tolerance_mm": 0.01}
```

模拟扫线复测可运行 `python benchmark_crop_sweep.py --with-boundary`。每一步的平面来自生成该裁剪的几何条件，以扫描坐标表达。识别程序仍从点云估计配准位姿。复测使用 0.01 个模型单位的边界容差。

实现依据为进度汇报第 10 页方案 B，粗配准与精配准接口参照 [Open3D 配准文档](https://open3d.org/docs/release/tutorial/pipelines/global_registration.html)。
