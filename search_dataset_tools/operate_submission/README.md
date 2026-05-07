# Operation Interface with different levels

## 相同部分

- `run_code`: 运行代码，得到控制台结果
- `validate_submission`: 验证代码的提交是否合法
- `grade_code`: 打分评价获得测试集分数
- `read_code`: 输出 `code_{node_id}_dataloader.py` 和 `code_{node_id}_template.py` 的文件内容 + Base Data Loader 的定义（不包含在文件中，但是硬编码，需要读取 ${PROJECT_ROOT}/search_dataset_tools/operate_submission/base_dataloader.py 文件路径进行读取）

## For red & black node exp

- `write_code` & `fix_code`: 只能操作 `code_{node_id}_dataloader.py`

## For initial exp

- `write_code` & `fix_code`: 可以操作 `code_{node_id}_dataloader.py` 和 `code_{node_id}_template.py` 两个文件


每个代码文件存储的内容：

- `code_{node_id}_dataloader.py`: 存储 MyDataLoader 的类定义
- `code_{node_id}_template.py`：存储训练脚本和训练代码
    - 对于 initial exp，其核心任务就是生成这个 template
    - 后续节点目前不可以修改这个 template，只能修改数据的格式

然后和 agent 的提示词需要强调下 DataLoader 不需要导入语句，会自动拼接成一个完整的代码 `code_{node_id}_overall.py` (调用 `run_code` 工具的时候会自动生成，agent 结束如果没有这个文件也会自动拼接生成)

然后和 agent 强调下（尤其是 InitialExp 强调下，在 template 的代码里面必须使用 MyDataLoader 类，并且读取数据符合规范和一定的契约）

拼接的顺序是

base_data_loader 的代码 + \n\n + MyDataLoader 的类定义 + template 的运行脚本