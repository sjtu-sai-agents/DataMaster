### First Principle (MUST OBEY)

- The solution sketch should be a brief natural language description of how the previous solution can be improved.
- You should be very specific and should only propose **a single actionable improvement**. This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.
- This means that you **should NOT make multiple changes** at once.
    - For instance, do not simultaneously change the feature engineering and the model architecture.
- The improvement should be based on the previous solution's execution output provided below.
- When proposing the design, take the Memory section into account.
- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.
- Don't propose the same modelling solution but keep the evaluation the same.
- The solution sketch should be 3-5 sentences.

### Abailable Improvement Methods

在脚本成功运行的前提下，提升最终模型性能的表现往往有两种通用的渠道：

- **算法增强**：选择新的模型、更换新的训练方法在原始的数据集上进行训练
- **数据增强**：增强现有训练集 & 到网络上寻找更多公开的类似数据集进行数据增强，从源头增强训练的效果

在选择优化方向的时候，你可以**选择一个可行的优化方向**进行深度挖掘（either 算法增强 or 数据增强），下面是一些可供指导的意见参考！你可以大胆地尝试各种不同的优化方向！(but ONLY ONCE at a time!)

{data_enhancement_prompt}

{algo_enhancement_prompt}
