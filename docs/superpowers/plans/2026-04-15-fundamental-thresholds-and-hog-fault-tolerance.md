# 分类阈值与生猪容错实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现品种分类基本面阈值，并为生猪增加基于价格区间位的动态保底分机制，确保系统在高敏感品种和 API 故障时的稳定性。

**Architecture:** 
1. 在配置文件中定义品种分类及其对应的基本面筛选阈值。
2. 筛选逻辑根据品种动态查找阈值。
3. 生猪数据抓取增加局部失败标记，筛选逻辑根据标记及 `range_pct` 进行分数补偿。

**Tech Stack:** Python, YAML, pandas, numpy

---

### Task 1: 配置层更新

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: 添加 `fundamental_screening` 配置段**
- [ ] **Step 2: 提交配置修改**

---

### Task 2: 生猪数据层容错优化

**Files:**
- Modify: `scripts/data_cache.py`

- [ ] **Step 1: 修改 `get_hog_fundamentals` 以支持部分数据返回并标记状态**

---

### Task 3: 筛选逻辑与动态阈值实现

**Files:**
- Modify: `scripts/daily_workflow.py`

- [ ] **Step 1: 在 `phase_1_screen` 中实现动态阈值查找**
- [ ] **Step 2: 在 `_score_symbol` 中实现生猪保底逻辑 (Option B)**

---

### Task 4: 报告展示与验证

**Files:**
- Modify: `scripts/daily_workflow.py`

- [ ] **Step 1: 验证报告说明文案与表格对齐**
- [ ] **Step 2: 模拟测试与运行验证**
