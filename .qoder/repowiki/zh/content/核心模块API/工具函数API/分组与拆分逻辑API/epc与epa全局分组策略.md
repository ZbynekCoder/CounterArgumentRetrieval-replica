# epc与epa全局分组策略

<cite>
**本文引用的文件**
- [utils.py](file://utils.py)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py)
- [dataloader.py](file://dataloader.py)
- [README.md](file://README.md)
</cite>

## 目录
1. [引言](#引言)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构总览](#架构总览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考量](#性能考量)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)

## 引言
本文件系统性阐述epc与epa两类全局评估级别在分组与拆分流程中的特殊处理机制，重点说明：
- group_method()在epc/epa下不执行分组，直接返回完整数据集；
- split_method()在epc/epa下分别调用针对“全门户”场景的计数拆分与论点拆分函数；
- 如何通过domain与argumentation_title双重过滤条件模拟“全库检索”场景，从而评估模型在大规模候选集中的检索能力；
- 双重过滤条件在保证数据有效性方面的关键作用以及其计算复杂度与评估价值。

## 项目结构
该项目围绕辩论性论证文本的数据组织与评估展开，核心流程由数据加载、分组与拆分、嵌入提取与检索评估构成。epc与epa属于评估任务集合中的两类全局级别，它们通过统一的分组与拆分接口参与评估流水线。

```mermaid
graph TB
subgraph "数据层"
DL["dataloader.py<br/>读取原始文本并构造DataFrame"]
end
subgraph "评估工具层"
U["utils.py<br/>group_method()<br/>split_method()<br/>split_point_*_for_entire_portal()"]
TE["tasks_evaluator.py<br/>构建8类任务数据"]
TEU["tasks_evaluator_util.py<br/>构建任务数据与BallTree评估"]
end
subgraph "模型层"
M["BERT Biencoder 模型"]
end
DL --> TE
TE --> U
TEU --> U
U --> M
```

图示来源
- [dataloader.py](file://dataloader.py#L30-L75)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)
- [utils.py](file://utils.py#L259-L290)

章节来源
- [README.md](file://README.md#L1-L7)
- [dataloader.py](file://dataloader.py#L30-L75)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)
- [utils.py](file://utils.py#L259-L290)

## 核心组件
- 分组与拆分接口
  - group_method(data, group_level)：根据group_level选择是否按字段分组，epc/epa返回整表不分组。
  - split_method(data, group_level)：根据group_level选择拆分策略，epc/epa分别调用“全门户”计数拆分与“全门户”论点拆分。
- “全门户”拆分函数
  - split_point_counter_for_entire_portal(data)：按立场(pro/con)拆分point与counter，使用domain与argumentation_title双重过滤。
  - split_point_argument_for_entire_portal(data)：按立场(pro/con)拆分point与argument，同样使用domain与argumentation_title双重过滤。
- 评估入口
  - tasks_evaluator.py与tasks_evaluator_util.py中均调用group_and_split(df, level)构建8类任务数据，其中包含epc与epa。

章节来源
- [utils.py](file://utils.py#L259-L290)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)

## 架构总览
epc与epa的评估流程遵循“先不分组，再按全门户规则拆分”的设计，确保评估覆盖整个语料库规模，避免因分组导致的样本泄露或过拟合。

```mermaid
sequenceDiagram
participant Loader as "数据加载器"
participant Evaluator as "评估入口"
participant Utils as "utils.py"
participant Model as "模型/BallTree评估"
Loader->>Evaluator : 提供DataFrame
Evaluator->>Utils : group_and_split(df, "epc"/"epa")
Utils->>Utils : group_method(df, level)
alt level为epc/epa
Utils-->>Evaluator : 返回整表(不分组)
else 其他level
Utils-->>Evaluator : 按字段分组后返回列表
end
Utils->>Utils : split_method(df, level)
alt level为epc
Utils->>Utils : 调用 split_point_counter_for_entire_portal(df)
else level为epa
Utils->>Utils : 调用 split_point_argument_for_entire_portal(df)
end
Utils-->>Evaluator : 返回拆分后的数据组
Evaluator->>Model : 进行检索/分类评估
```

图示来源
- [utils.py](file://utils.py#L259-L290)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)

## 详细组件分析

### 分组策略：group_method()在epc/epa下的行为
- 设计要点
  - 对于epc/epa，group_method()在映射表中对应的分组字段为空数组，因此不分组，直接返回包含整张表的一个元素列表。
  - 这一设计使得后续拆分阶段面对的是“全库”视角，而非按领域/主题等维度切分后的子集。
- 评估意义
  - 避免因分组导致的样本分布偏差；
  - 使检索评估更接近真实生产环境中的全库检索场景。

```mermaid
flowchart TD
Start(["进入 group_method"]) --> Lookup["查找 level 映射"]
Lookup --> IsEmpty{"映射字段为空？"}
IsEmpty --> |是| ReturnFull["返回整表列表"]
IsEmpty --> |否| GroupBy["按映射字段分组"]
GroupBy --> ReturnGrouped["返回分组结果"]
ReturnFull --> End(["结束"])
ReturnGrouped --> End
```

图示来源
- [utils.py](file://utils.py#L259-L274)

章节来源
- [utils.py](file://utils.py#L259-L274)

### 拆分策略：split_method()在epc/epa下的行为
- 设计要点
  - 对于epc，split_method()调用split_point_counter_for_entire_portal()，按立场(pro/con)拆分point与counter，并使用domain与argumentation_title双重过滤。
  - 对于epa，split_method()调用split_point_argument_for_entire_portal()，按立场(pro/con)拆分point与argument，并使用同样的双重过滤。
- 评估意义
  - 通过“全门户”规则，确保每个point都与同一argumentation_title下的counter/argument进行匹配，从而在更大候选空间中评估检索质量。
  - 双重过滤有效控制了跨主题/跨立场的错误匹配，提升评估的可靠性。

```mermaid
flowchart TD
Start(["进入 split_method"]) --> Lookup["查找 level 映射函数"]
Lookup --> Level{"level 是 epc 还是 epa？"}
Level --> |epc| CallC["调用 split_point_counter_for_entire_portal"]
Level --> |epa| CallA["调用 split_point_argument_for_entire_portal"]
CallC --> ReturnC["返回计数拆分结果"]
CallA --> ReturnA["返回论点拆分结果"]
ReturnC --> End(["结束"])
ReturnA --> End
```

图示来源
- [utils.py](file://utils.py#L277-L289)

章节来源
- [utils.py](file://utils.py#L277-L289)

### “全门户”拆分函数：计数与论点拆分
- split_point_counter_for_entire_portal(data)
  - 功能：按立场(pro/con)拆分point与counter，使用domain与argumentation_title双重过滤，确保匹配在同一argumentation_title内且立场一致。
  - 复杂度：对每条记录扫描并筛选，时间复杂度约为O(N^2)（在单个argumentation内部）；整体复杂度取决于数据结构与索引策略。
- split_point_argument_for_entire_portal(data)
  - 功能：按立场(pro/con)拆分point与argument，同样使用domain与argumentation_title双重过滤。
  - 复杂度：同上，受argumentation内部记录数量影响。

```mermaid
flowchart TD
S(["输入：DataFrame 列包含 domain/argumentation_title/stance/utterence_type 等"]) --> Loop["逐条遍历记录"]
Loop --> Filter["按 stance 与 domain/argumentation_title 过滤"]
Filter --> Match{"是否存在匹配的 counter/argument？"}
Match --> |是| Append["加入对应列表"]
Match --> |否| Drop["丢弃该 point"]
Append --> Next["继续遍历"]
Drop --> Next
Next --> Done["输出：point 与 counter/argument 列表"]
```

图示来源
- [utils.py](file://utils.py#L195-L231)
- [utils.py](file://utils.py#L235-L255)

章节来源
- [utils.py](file://utils.py#L195-L231)
- [utils.py](file://utils.py#L235-L255)

### 评估入口与调用链
- tasks_evaluator.py与tasks_evaluator_util.py均调用group_and_split(df, level)，其中level包含epc与epa。
- 该调用链确保epc/epa与其他level（如sdoc/sdoa/stc等）共享相同的分组与拆分接口，便于统一评估与对比。

```mermaid
sequenceDiagram
participant DF as "DataFrame"
participant TE as "tasks_evaluator.py"
participant TEU as "tasks_evaluator_util.py"
participant U as "utils.py"
TE->>U : group_and_split(df, "epc")
TEU->>U : group_and_split(df, "epa")
U->>U : group_method(...)
U->>U : split_method(...)
U-->>TE : 返回拆分结果
U-->>TEU : 返回拆分结果
```

图示来源
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)
- [utils.py](file://utils.py#L259-L290)

章节来源
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)
- [utils.py](file://utils.py#L259-L290)

## 依赖关系分析
- 组件耦合
  - 评估入口依赖utils.py提供的分组与拆分接口；
  - utils.py内部通过映射表将level与具体拆分函数绑定，降低调用方与实现细节的耦合。
- 关键依赖链
  - dataloader.py提供DataFrame；
  - tasks_evaluator*.py调用group_and_split；
  - utils.py实现group_method与split_method；
  - 模型/BallTree评估模块消费拆分后的数据组。

```mermaid
graph LR
D["dataloader.py"] --> T1["tasks_evaluator.py"]
D --> T2["tasks_evaluator_util.py"]
T1 --> U["utils.py"]
T2 --> U
U --> M["模型/BallTree评估"]
```

图示来源
- [dataloader.py](file://dataloader.py#L30-L75)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)
- [utils.py](file://utils.py#L259-L290)

章节来源
- [dataloader.py](file://dataloader.py#L30-L75)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)
- [utils.py](file://utils.py#L259-L290)

## 性能考量
- 时间复杂度
  - epc/epa下的拆分函数在单个argumentation内部进行point与counter/argument的配对，整体复杂度约O(N^2)（N为单argumentation内的记录数）。
  - 若argumentation数量较多或单个argumentation规模较大，建议考虑：
    - 使用索引或预处理加速匹配；
    - 将数据按argumentation_title分桶，减少不必要的全表扫描。
- 空间复杂度
  - 拆分过程主要产生point/counter/argument列表，空间开销与记录总数线性相关。
- 实际部署建议
  - 在大规模候选集上评估时，优先采用“全门户”策略以贴近真实检索场景；
  - 结合BallTree等高效检索方法，平衡召回与效率。

## 故障排查指南
- 常见问题
  - 数据缺失domain/argumentation_title/stance/utterence_type字段：会导致拆分失败或结果异常。
  - 记录重复或缺失counter：可能导致point被丢弃，影响评估样本数量。
- 排查步骤
  - 确认DataFrame列名与拆分函数期望一致；
  - 检查每个argumentation_title内的point与counter配对是否满足立场与标题一致性；
  - 对epc/epa结果进行抽样核验，确认无跨主题/跨立场误配。
- 相关定位路径
  - 分组与拆分逻辑：[utils.py](file://utils.py#L259-L290)
  - “全门户”计数拆分：[utils.py](file://utils.py#L195-L231)
  - “全门户”论点拆分：[utils.py](file://utils.py#L235-L255)
  - 评估入口调用：[tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37), [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)

章节来源
- [utils.py](file://utils.py#L195-L231)
- [utils.py](file://utils.py#L235-L255)
- [utils.py](file://utils.py#L259-L290)
- [tasks_evaluator.py](file://bert/tasks_evaluator.py#L24-L37)
- [tasks_evaluator_util.py](file://bert/tasks_evaluator_util.py#L13-L24)

## 结论
epc与epa通过“不分组、全门户”的策略，将评估范围扩展至整个语料库，从而更真实地反映模型在大规模候选集中的检索与匹配能力。domain与argumentation_title的双重过滤在保证数据有效性的同时，也提升了评估的稳定性与可比性。结合BallTree等高效检索方法，可在保证评估价值的前提下优化整体性能。