#!/usr/bin/env python3
"""按 UCT 树节点可视化 trajectory - 支持 Grade 分数

python vis_node_by_tree_with_grade.py --run-dir ${PROJECT_ROOT}/runs/ml_master_datatree_20260326_165235 --port 8789 --dataset leaf-classification

新增功能：
- 加载 grade_results.json 中的 test_score
- HTML 中添加按钮切换显示 val_score 和 test_score
- --auto-grade: 自动对 submission 进行批量打分（如果 grade_results.json 不存在）
- --force-grade: 强制重新打分
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"


def _now() -> float:
    return time.time()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_run_dir(explicit: Optional[str]) -> Path:
    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
        if not run_dir.exists():
            raise SystemExit(f"run_dir not found: {run_dir}")
        return run_dir

    if not RUNS_DIR.exists():
        raise SystemExit(f"runs dir not found: {RUNS_DIR}")

    candidates = []
    for d in RUNS_DIR.glob("ml_master_*"):
        if d.is_dir() and (d / "logs" / "uct_nodes" / "node.json").exists():
            candidates.append(d)

    if not candidates:
        raise SystemExit("no run with node.json found under runs/")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def collect_node_ids(
    node: Dict[str, Any], ids: Optional[List[str]] = None
) -> List[str]:
    """递归收集树中所有节点的 ID"""
    if ids is None:
        ids = []

    node_id = node.get("id", "")
    if node_id and node_id != "__virtual_root__":
        ids.append(node_id)

    for child in node.get("children", []):
        collect_node_ids(child, ids)
    return ids


def load_node_tree(run_dir: Path) -> Dict[str, Any]:
    """加载 UCT 节点树"""
    nodes_dir = run_dir / "logs" / "uct_nodes"

    node_file = nodes_dir / "node.json"
    if node_file.exists():
        try:
            with open(node_file, "r", encoding="utf-8") as f:
                tree = json.load(f)
            if tree:
                if isinstance(tree, dict) and "root" in tree:
                    return tree["root"]
                return tree
        except Exception as e:
            print(
                f"[WARNING] Failed to load node.json: {e}, building from individual files..."
            )

    print(f"[INFO] Building tree from individual node files...")

    nodes_by_id = {}
    for node_file in sorted(nodes_dir.glob("*.json")):
        if node_file.name == "node.json":
            continue
        try:
            with open(node_file, "r", encoding="utf-8") as f:
                node = json.load(f)
                if node and node.get("id"):
                    nodes_by_id[node["id"]] = node
        except Exception:
            continue

    print(f"[INFO] Loaded {len(nodes_by_id)} node files")

    if not nodes_by_id:
        raise SystemExit(f"No valid node files found in {nodes_dir}")

    roots = []
    for node_id, node in nodes_by_id.items():
        parent_id = node.get("parent")
        if not parent_id or parent_id not in nodes_by_id:
            roots.append(node)

    def build_children(node):
        node["children"] = [
            build_children(nodes_by_id[cid])
            for cid, child in nodes_by_id.items()
            if child.get("parent") == node["id"]
        ]
        return node

    tree_roots = [build_children(root) for root in roots]

    if len(tree_roots) == 1:
        return tree_roots[0]

    return {
        "id": "__virtual_root__",
        "stage": "root",
        "children": tree_roots,
    }


def load_grade_results(run_dir: Path) -> Dict[str, float]:
    """加载 grade 结果

    返回: {submission_id: score}
    """
    grade_paths = [
        run_dir / "trajectories" / "task_0" / "grade_results.json",
        run_dir / "test" / "grade_results.json",
    ]

    for grade_path in grade_paths:
        if grade_path.exists():
            try:
                with open(grade_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[INFO] Loaded grade results from {grade_path}")
                return {sid: entry.get("score") for sid, entry in data.items()}
            except Exception as e:
                print(f"[WARNING] Failed to load grade results from {grade_path}: {e}")

    print("[WARNING] No grade results found")
    return {}


def run_auto_grading(
    run_dir: Path, dataset: str = "leaf-classification", force: bool = False
) -> Optional[Dict[str, float]]:
    """自动对 submission 进行批量打分

    Args:
        run_dir: 运行目录
        dataset: 数据集名称，对应 ${DATA_ROOT}/{dataset}/prepared/grade.py
        force: 是否强制重新打分（忽略已有结果）

    Returns:
        {submission_id: score} 或 None（如果不需要打分）
    """
    grade_results_dir = run_dir / "trajectories" / "task_0"
    grade_results_path = grade_results_dir / "grade_results.json"

    if not force and grade_results_path.exists():
        return None

    submission_dirs = [
        run_dir / "workspaces" / "task_0" / "submission",
        run_dir / "workspaces" / "submission",
    ]

    submission_dir = None
    for d in submission_dirs:
        if d.exists() and d.is_dir():
            submission_dir = d
            break

    if not submission_dir:
        print("[INFO] No submission directory found, skipping auto grading")
        return None

    base_dir = Path("${DATA_ROOT}")
    grade_script_paths = [
        base_dir / dataset / "prepared" / "grade.py",
        REPO_ROOT / "test" / "grade.py",
    ]

    grade_script = None
    for p in grade_script_paths:
        if p.exists():
            grade_script = p
            break

    if not grade_script:
        print("[WARNING] grade.py not found, skipping auto grading")
        return None

    submission_files = list(submission_dir.glob("submission_*.csv"))
    if not submission_files:
        print(f"[INFO] No submission files found in {submission_dir}")
        return None

    print(f"[INFO] Running auto grading on {len(submission_files)} submissions...")
    print(f"[INFO] Grade script: {grade_script}")

    def extract_score(output: str) -> Optional[float]:
        """从 grade.py 输出中提取分数"""
        match = re.search(r"metric\s*=.*?(\d+\.\d+)", output)
        if match:
            return float(match.group(1))
        return None

    def grade_submission(submission_path: str) -> Tuple[str, Optional[float]]:
        """对单个 submission 进行评测"""
        submission_name = os.path.basename(submission_path)
        try:
            result = subprocess.run(
                [sys.executable, str(grade_script), "-s", submission_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            score = extract_score(output)
            return submission_name, score
        except subprocess.TimeoutExpired:
            return submission_name, None
        except Exception:
            return submission_name, None

    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(grade_submission, str(f)): str(f) for f in submission_files
        }

        for future in as_completed(futures):
            submission_name, score = future.result()
            submission_id = submission_name.replace("submission_", "").replace(
                ".csv", ""
            )
            results[submission_id] = {"score": score}
            if score is not None:
                print(f"  {submission_id[:8]}...: {score:.5f}")
            else:
                print(f"  {submission_id[:8]}...: FAILED")

    grade_results_dir.mkdir(parents=True, exist_ok=True)
    with open(grade_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    scores = [r["score"] for r in results.values() if r["score"] is not None]
    if scores:
        print(
            f"[INFO] Grading complete: {len(scores)}/{len(submission_files)} succeeded"
        )
        print(
            f"[INFO] Score stats - Max: {max(scores):.5f}, Min: {min(scores):.5f}, Mean: {sum(scores)/len(scores):.5f}"
        )
    else:
        print(f"[WARNING] No submissions were successfully graded")

    print(f"[INFO] Grade results saved to: {grade_results_path}")
    return {sid: entry.get("score") for sid, entry in results.items()}


def match_trajectories_to_nodes(
    run_dir: Path, node_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """匹配 trajectory entries 到节点"""
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        return {}

    trajectories_by_node: Dict[str, List[Dict[str, Any]]] = {
        nid: [] for nid in node_ids
    }

    for task_dir in sorted(traj_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue
        traj_file = task_dir / "trajectory.json"
        if not traj_file.exists():
            continue

        try:
            with open(traj_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for entry in data:
                if not isinstance(entry, dict):
                    continue

                agent_id = entry.get("agent_id", "")
                parts = agent_id.split("_")
                if not parts:
                    continue

                potential_node_id = parts[0]

                for node_id in node_ids:
                    if node_id.startswith(
                        potential_node_id
                    ) or potential_node_id.startswith(node_id[:8]):
                        trajectories_by_node[node_id].append(entry)
                        break

        except Exception as e:
            print(f"[WARNING] Failed to load trajectory from {traj_file}: {e}")
            continue

    result: Dict[str, Dict[str, Any]] = {}

    for node_id, entries in trajectories_by_node.items():
        if not entries:
            continue

        max_entry = max(entries, key=lambda e: e.get("steps", 0))
        traj = max_entry.get("trajectory", {})

        result[node_id] = {
            "node_id": node_id,
            "task_id": max_entry.get("task_id"),
            "agent_id": max_entry.get("agent_id"),
            "exp_name": max_entry.get("exp_name"),
            "exp_index": max_entry.get("exp_index"),
            "steps": max_entry.get("steps"),
            "status": max_entry.get("status"),
            "agent_name": max_entry.get("agent_name"),
            "trajectory": {
                "messages": traj.get("messages", []),
                "meta": traj.get("meta", {}),
            },
        }

    return result


def add_grade_scores_to_tree(
    tree: Dict[str, Any], grade_results: Dict[str, float]
) -> None:
    """将 grade 分数添加到树节点

    node_id 和 submission_id 应该匹配（或前8位匹配）
    """

    def add_scores_recursive(node: Dict[str, Any]) -> None:
        node_id = node.get("id", "")
        if not node_id or node_id == "__virtual_root__":
            for child in node.get("children", []):
                add_scores_recursive(child)
            return

        grade_score = None
        if node_id in grade_results:
            grade_score = grade_results[node_id]
        else:
            node_prefix = node_id[:8]
            for sid, score in grade_results.items():
                if sid.startswith(node_prefix) or node_prefix == sid[:8]:
                    grade_score = score
                    break

        if grade_score is not None:
            node["test_score"] = grade_score

        for child in node.get("children", []):
            add_scores_recursive(child)

    add_scores_recursive(tree)


def load_payload(run_dir: Path) -> Dict[str, Any]:
    """加载可视化数据"""
    tree = load_node_tree(run_dir)
    node_ids = collect_node_ids(tree)
    grade_results = load_grade_results(run_dir)
    add_grade_scores_to_tree(tree, grade_results)
    node_trajectories = match_trajectories_to_nodes(run_dir, node_ids)

    return {
        "generated_at": _now(),
        "run_dir": str(run_dir),
        "node_count": len(node_ids),
        "tree": tree,
        "node_trajectories": node_trajectories,
        "matched_nodes": len(node_trajectories),
        "has_grade_data": len(grade_results) > 0,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>UCT Node + Trajectory Viewer (By Tree) + Grade Scores</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    :root { --bg:#0b1020; --panel:#131a2f; --text:#e7ebff; --muted:#95a0c7; --edge:#5f6b98; --accent:#4dabf7; }
    * { box-sizing:border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system; background:radial-gradient(1200px 600px at 10% 0%, #1f2d58 0%, var(--bg) 55%); color:var(--text); }
    .top { display:flex; justify-content:space-between; align-items:center; padding:12px 16px; background:rgba(8,11,24,.75); backdrop-filter: blur(6px); position:sticky; top:0; z-index:5; border-bottom:1px solid #2a355f; }
    .title { font-weight:700; letter-spacing:.3px; }
    .meta { color:var(--muted); font-size:13px; }
    .layout { display:flex; height:calc(100vh - 56px); }
    .left { flex:1; position:relative; }
    .right { width:500px; border-left:1px solid #2a355f; background:rgba(12,18,37,.7); overflow:auto; }
    #svg { width:100%; height:100%; display:block; }
    .hint { position:absolute; left:12px; bottom:12px; color:var(--muted); font-size:12px; background:rgba(8,11,24,.7); padding:6px 8px; border-radius:8px; }

    .score-toggle { display:flex; gap:8px; align-items:center; }
    .toggle-btn { padding:6px 12px; border:1px solid #4dabf7; background:transparent; color:#4dabf7; border-radius:6px; cursor:pointer; font-size:12px; transition:all 0.2s; }
    .toggle-btn:hover { background:rgba(77,171,247,0.1); }
    .toggle-btn.active { background:#4dabf7; color:#0b1020; font-weight:600; }

    .panel { padding:12px; }
    .card { background:var(--panel); border:1px solid #2c3760; border-radius:10px; padding:10px; margin-bottom:10px; }
    .k { color:#8ea3ff; font-size:12px; margin-bottom:4px; }
    .v { white-space:pre-wrap; word-break:break-word; font-size:13px; }
    .code { max-height:400px; overflow:auto; background:#0a1124; border:1px solid #33406f; border-radius:8px; padding:8px; font-size:11px; }

    .node text { fill:#fff; font-size:10px; pointer-events:none; }
    .node .node-shape { stroke:#d9e0ff; stroke-width:1.1; }
    .node.has-traj .node-shape { stroke-width:2.5; stroke:#4dabf7; }
    .node.no-traj .node-shape { stroke-dasharray: 3,3; opacity: 0.7; }
    .node.sel .node-shape { stroke:#ffd166; stroke-width:2.4; }
    .link { fill:none; stroke:var(--edge); stroke-opacity:.55; stroke-width:1.1; }

    .message { background:rgba(10,17,36,.8); border:1px solid #2c3760; border-radius:8px; padding:10px; margin-bottom:8px; }
    .message-role { font-size:11px; font-weight:600; text-transform:uppercase; margin-bottom:4px; }
    .message-role.system { color:#ff6b6b; }
    .message-role.user { color:#4dabf7; }
    .message-role.assistant { color:#51cf66; }
    .message-role.tool { color:#ffd166; }
    .message-content { font-size:12px; white-space:pre-wrap; word-break:break-word; max-height:300px; overflow:auto; }
  </style>
</head>
<body>
  <div class="top">
    <div class="title">UCT Node + Trajectory + Grade Scores</div>
    <div style="display:flex; gap:16px; align-items:center;">
      <div class="score-toggle">
        <button class="toggle-btn active" id="btn-val" onclick="setScoreMode('val')">Val Score</button>
        <button class="toggle-btn" id="btn-test" onclick="setScoreMode('test')">Test Score (Grade)</button>
      </div>
      <div class="meta" id="meta"></div>
    </div>
  </div>
  <div class="layout">
    <div class="left">
      <svg id="svg"></svg>
      <div class="hint">Wheel: zoom · Drag: pan · Click node: trajectory | Toggle: Val/Test score | Solid: has data · Dashed: no data</div>
    </div>
    <div class="right"><div class="panel" id="panel"></div></div>
  </div>
  <script>
  const DATA = __DATA__;
  let scoreMode = 'val';

  const panel = document.getElementById('panel');
  const metaEl = document.getElementById('meta');
  const svg = d3.select('#svg');
  const g = svg.append('g');
  const zoom = d3.zoom().scaleExtent([0.08, 4]).on('zoom', (e)=>g.attr('transform', e.transform));
  svg.call(zoom);

  let selectedNodeId = null;

  function setScoreMode(mode) {
    scoreMode = mode;
    document.getElementById('btn-val').classList.toggle('active', mode === 'val');
    document.getElementById('btn-test').classList.toggle('active', mode === 'test');
    drawTree();
    if (selectedNodeId) {
      renderNodeDetail(selectedNodeId);
    }
  }

  function updateMeta(){
    metaEl.textContent = `run=${DATA.run_dir} | nodes=${DATA.node_count} | with_traj=${DATA.matched_nodes} | grade_data=${DATA.has_grade_data}`;
  }

  function getScore(node) {
    if (scoreMode === 'test') {
      return node.test_score;
    }
    return node.metric;
  }

  function nodeColor(node, trajData) {
    if (!trajData[node.id]) return '#8893bd';
    if (node.is_buggy) return '#ff4d6d';
    if (node.stage === 'root') return '#ffd166';
    const stage = node.stage || '';
    if (stage === 'initial') return '#4dabf7';
    if (stage === 'black') return '#51cf66';
    if (stage === 'red') return '#cc5de8';
    if (stage === 'terminal') return '#868e96';
    return '#8893bd';
  }

  function nodeShapeType(node) {
    const stage = node.stage || '';
    if (stage === 'red') return d3.symbolTriangle;
    if (stage === 'black') return d3.symbolSquare;
    return d3.symbolCircle;
  }

  function renderNodeDetail(nodeId) {
    const trajData = DATA.node_trajectories || {};
    const traj = trajData[nodeId];

    panel.innerHTML = '';

    const nodeCard = document.createElement('div');
    nodeCard.className = 'card';
    nodeCard.innerHTML = '<div class="k">Node Info</div>';

    const findNode = (n, id) => {
      if (n.id === id) return n;
      for (let c of n.children || []) {
        const found = findNode(c, id);
        if (found) return found;
      }
      return null;
    };

    const node = findNode(DATA.tree, nodeId);
    if (node) {
      nodeCard.innerHTML += `<div class="v">ID: ${node.id.substring(0,16)}...</div>`;
      nodeCard.innerHTML += `<div class="v">Stage: ${node.stage || 'unknown'}</div>`;
      nodeCard.innerHTML += `<div class="v">Action: ${node.action_type || 'N/A'}</div>`;
      nodeCard.innerHTML += `<div class="v">Visits: ${node.visits || 0}</div>`;
      nodeCard.innerHTML += `<div class="v">Reward: ${node.reward !== null ? node.reward : 'N/A'}</div>`;
      nodeCard.innerHTML += `<div class="v">Total Reward: ${node.total_reward !== null ? node.total_reward : 'N/A'}</div>`;
      const uctVal = Number(node.uct_value);
      nodeCard.innerHTML += `<div class="v">UCT Value: ${Number.isFinite(uctVal) ? uctVal.toFixed(4) : String(node.uct_value ?? 'N/A')}</div>`;

      const valScore = node.metric !== null ? node.metric : 'N/A';
      const testScore = node.test_score !== null && node.test_score !== undefined ? node.test_score : 'N/A';
      const modeLabel = scoreMode === 'val' ? 'Val Score (metric)' : 'Test Score (grade)';
      nodeCard.innerHTML += `<div class="v" style="color:${scoreMode === 'test' ? '#4dabf7' : '#51cf66'}">${modeLabel}: ${scoreMode === 'val' ? valScore : testScore}</div>`;
      nodeCard.innerHTML += `<div class="v" style="font-size:11px; color:#95a0c7">Val Score: ${valScore} | Test Score: ${testScore}</div>`;

      nodeCard.innerHTML += `<div class="v">Has Submission: ${node.has_submission ? 'Yes' : 'No'}</div>`;
      nodeCard.innerHTML += `<div class="v">Buggy: ${node.is_buggy ? 'Yes' : 'No'}</div>`;
      nodeCard.innerHTML += `<div class="v">Valid: ${node.is_valid !== null ? (node.is_valid ? 'Yes' : 'No') : 'N/A'}</div>`;
    }
    panel.appendChild(nodeCard);

    if (traj) {
      const trajCard = document.createElement('div');
      trajCard.className = 'card';
      trajCard.innerHTML = '<div class="k">Trajectory</div>';
      trajCard.innerHTML += `<div class="v">Task: ${traj.task_id || 'N/A'}</div>`;
      trajCard.innerHTML += `<div class="v">Exp: ${traj.exp_name || 'N/A'}</div>`;
      trajCard.innerHTML += `<div class="v">Steps: ${traj.steps || 0}</div>`;
      panel.appendChild(trajCard);

      if (traj.trajectory && traj.trajectory.messages && traj.trajectory.messages.length > 0) {
        const messages = traj.trajectory.messages;

        const messagesCard = document.createElement('div');
        messagesCard.className = 'card';
        messagesCard.innerHTML = '<div class="k">对话历史</div>';

        messages.forEach(msg => {
          messagesCard.appendChild(renderMessage(msg));
        });

        panel.appendChild(messagesCard);
      }
    } else {
      const noTraj = document.createElement('div');
      noTraj.className = 'card';
      noTraj.innerHTML = '<div class="v">No trajectory data for this node</div>';
      panel.appendChild(noTraj);
    }
  }

  function drawTree(){
    g.selectAll('*').remove();

    const root = d3.hierarchy(DATA.tree);
    d3.tree().nodeSize([100, 170])(root);

    g.selectAll('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class','link')
      .attr('d', d3.linkVertical().x(d=>d.x).y(d=>d.y));

    const nodes = g.selectAll('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', d => {
        let classes = 'node';
        if (DATA.node_trajectories[d.data.id]) {
          classes += ' has-traj';
        } else {
          classes += ' no-traj';
        }
        if (d.data.id === selectedNodeId) classes += ' sel';
        return classes;
      })
      .attr('transform', d=>`translate(${d.x},${d.y})`)
      .on('click', function(_, d){
        selectedNodeId = d.data.id;
        g.selectAll('.node').classed('sel', n => n.data.id === selectedNodeId);
        renderNodeDetail(selectedNodeId);
      });

    nodes.append('path')
      .attr('class', 'node-shape')
      .attr('d', d => {
        const v = Number(d.data.visits || 0);
        const r = Math.max(8, Math.min(22, 9 + Math.log10(v + 1) * 6));
        return d3.symbol()
          .type(nodeShapeType(d.data))
          .size(Math.PI * r * r)();
      })
      .attr('fill', d=>nodeColor(d.data, DATA.node_trajectories));

    nodes.append('text')
      .attr('text-anchor','middle')
      .attr('dominant-baseline','middle')
      .text(d=>{
        if (d.data.stage === 'root') {
          return 'root';
        }
        const score = getScore(d.data);
        const m = Number(score);
        return Number.isFinite(m) ? m.toFixed(3) : '';
      });

    if (!selectedNodeId && DATA.tree && DATA.tree.id) {
      selectedNodeId = DATA.tree.id;
      renderNodeDetail(selectedNodeId);
    }
  }

  function init(){
    updateMeta();
    drawTree();

    if (!selectedNodeId && DATA.tree && DATA.tree.id) {
      const firstWithTraj = findFirstNodeWithTraj(DATA.tree);
      if (firstWithTraj) {
        selectedNodeId = firstWithTraj.id;
        renderNodeDetail(selectedNodeId);
        g.selectAll('.node').classed('sel', n => n.data.id === selectedNodeId);
      }
    }

    svg.call(zoom.transform, d3.zoomIdentity.translate(70,40).scale(1));
  }

  function findFirstNodeWithTraj(node) {
    if (node.id && DATA.node_trajectories[node.id]) {
      return node;
    }
    for (let child of node.children || []) {
      const found = findFirstNodeWithTraj(child);
      if (found) return found;
    }
    return null;
  }

  function formatToolArgs(rawArgs) {
    if (rawArgs === null || rawArgs === undefined) return '';
    let text = rawArgs;
    if (typeof text !== 'string') {
      try {
        text = JSON.stringify(text, null, 2);
      } catch (_) {
        text = String(text);
      }
    }
    const trimmed = text.trim();
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        return JSON.stringify(JSON.parse(trimmed), null, 2);
      } catch (_) {
        return text;
      }
    }
    return text;
  }

  function parseToolCall(tc) {
    if (!tc) return { name: 'unknown', args: '' };

    if (typeof tc === 'object') {
      const fn = tc.function || {};
      const name = fn.name || tc.name || 'unknown';
      const args = fn.arguments !== undefined ? fn.arguments : (tc.arguments !== undefined ? tc.arguments : '');
      return { name, args };
    }

    if (typeof tc === 'string') {
      let name = 'unknown';
      let args = '';

      const nameMatch = tc.match(/FunctionCall\\(name='([^']+)'/);
      if (nameMatch && nameMatch[1]) {
        name = nameMatch[1];
      } else {
        const fallbackName = tc.match(/name='([^']+)'/);
        if (fallbackName && fallbackName[1]) name = fallbackName[1];
      }

      const argsMatch = tc.match(/arguments='([\\s\\S]*?)'\\)/);
      if (argsMatch && argsMatch[1]) {
        args = argsMatch[1].replace(/\\\\'/g, "'");
      }

      return { name, args };
    }

    return { name: 'unknown', args: String(tc) };
  }

  function renderMessage(msg) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message';

    const role = msg.role || 'unknown';
    msgDiv.innerHTML = `<div class="message-role ${role}">${role}</div>`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
    if (toolCalls.length > 0) {
      toolCalls.forEach(tc => {
        const parsed = parseToolCall(tc);
        const funcName = parsed.name || 'unknown';
        const funcArgs = formatToolArgs(parsed.args);

        const toolDiv = document.createElement('div');
        toolDiv.style.background = 'rgba(255, 209, 102, 0.1)';
        toolDiv.style.border = '1px solid rgba(255, 209, 102, 0.3)';
        toolDiv.style.borderRadius = '6px';
        toolDiv.style.padding = '8px';
        toolDiv.style.marginBottom = '8px';

        toolDiv.innerHTML = `<strong>Tool: ${funcName}</strong>`;
        const argsDiv = document.createElement('div');
        argsDiv.style.fontSize = '11px';
        argsDiv.style.color = '#95a0c7';
        argsDiv.style.marginTop = '4px';
        argsDiv.style.whiteSpace = 'pre-wrap';
        argsDiv.style.wordBreak = 'break-word';
        argsDiv.textContent = funcArgs;
        toolDiv.appendChild(argsDiv);

        contentDiv.appendChild(toolDiv);
      });
    }

    if (msg.content) {
      if (typeof msg.content === 'string') {
        const textDiv = document.createElement('div');
        textDiv.textContent = msg.content || '(empty)';
        textDiv.style.marginTop = toolCalls.length > 0 ? '8px' : '0';
        contentDiv.appendChild(textDiv);
      } else if (Array.isArray(msg.content)) {
        msg.content.forEach(item => {
          const itemDiv = document.createElement('div');
          itemDiv.style.marginTop = '4px';
          if (item.type === 'text') {
            itemDiv.textContent = item.text || '';
          } else if (item.type === 'image_url') {
            itemDiv.innerHTML = `<em>[Image: ${item.image_url?.url?.substring(0, 50)}...]</em>`;
          }
          contentDiv.appendChild(itemDiv);
        });
      }
    }

    if (role === 'tool' && msg.name) {
      const nameDiv = document.createElement('div');
      nameDiv.style.fontSize = '11px';
      nameDiv.style.color = '#ffd166';
      nameDiv.style.marginBottom = '4px';
      nameDiv.textContent = `Tool: ${msg.name}`;
      contentDiv.insertBefore(nameDiv, contentDiv.firstChild);
    }

    msgDiv.appendChild(contentDiv);
    return msgDiv;
  }

  init();
  </script>
</body>
</html>
"""


def write_payload(payload: Dict[str, Any], site_dir: Path) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    payload_path = site_dir / "payload.json"
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload_path


def write_site(payload: Dict[str, Any], site_dir: Path) -> Path:
    write_payload(payload, site_dir)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    index_path = site_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def _pick_free_port(host: str, port: int) -> int:
    p = max(1, int(port))
    for cand in range(p, p + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, cand))
                return cand
            except OSError:
                continue
    raise SystemExit(f"no free port near {port}")


def _stop_old_process(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, 0)
    except OSError:
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    pid_file.unlink(missing_ok=True)


def _start_http_server(site_dir: Path, host: str, port: int) -> Dict[str, Any]:
    server_pid_file = site_dir / ".vis_server.pid"
    server_log_file = site_dir / "vis_server.log"

    _stop_old_process(server_pid_file)

    bind_port = _pick_free_port(host, port)
    cmd = [
        sys.executable,
        "-m",
        "http.server",
        str(bind_port),
        "--bind",
        host,
        "--directory",
        str(site_dir),
    ]

    with server_log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start: {' '.join(cmd)}\\n")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, start_new_session=True)

    time.sleep(0.4)
    if proc.poll() is not None:
        raise SystemExit(f"deploy failed, see log: {server_log_file}")

    server_pid_file.write_text(str(proc.pid), encoding="utf-8")
    return {
        "pid": proc.pid,
        "host": host,
        "port": bind_port,
        "url": f"http://{host}:{bind_port}/index.html",
        "log": str(server_log_file),
        "pid_file": str(server_pid_file),
    }


def load_exp_id_from_config(run_dir: Path) -> Optional[str]:
    """从 run_dir/config.yaml 中读取 exp_id

    Args:
        run_dir: 运行目录

    Returns:
        exp_id 或 None
    """
    config_paths = [
        run_dir / "config.yaml",
        run_dir / "config.yml",
        run_dir / ".config" / "config.yaml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # 尝试多种可能的路径获取 exp_id
                exp_id = (
                    config.get("exp_id")
                    or (config.get("competition", {}).get("exp_id") if config.get("competition") else None)
                    or (config.get("config", {}).get("exp_id") if config.get("config") else None)
                )

                if exp_id:
                    print(f"[INFO] Loaded exp_id from config: {exp_id}")
                    return exp_id
                else:
                    print(f"[WARNING] config file found but no exp_id in {config_path}")

            except ImportError:
                print("[WARNING] pyyaml not installed, cannot read config.yaml")
                break
            except Exception as e:
                print(f"[WARNING] Failed to load config from {config_path}: {e}")
                continue

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize UCT nodes with trajectory data and grade scores"
    )
    p.add_argument("--run-dir", type=str, default=None, help="target run dir")
    p.add_argument("--site-dir", type=str, default=None, help="output site dir")
    p.add_argument("--host", type=str, default="127.0.0.1", help="http bind host")
    p.add_argument("--port", type=int, default=8789, help="http port")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name under ${DATA_ROOT}/ (auto-detected from config if not provided)",
    )
    p.add_argument("--no-deploy", action="store_true", help="only generate html")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = discover_run_dir(args.run_dir)
    site_dir = (
        Path(args.site_dir).expanduser().resolve()
        if args.site_dir
        else (run_dir / "logs" / "uct_nodes" / "vis_site")
    )

    # 自动从 config 中读取 dataset/exp_id
    dataset = args.dataset
    if not dataset:
        print("[INFO] --dataset not provided, attempting to auto-detect from config...")
        dataset = load_exp_id_from_config(run_dir)
        if not dataset:
            print("[WARNING] Could not auto-detect dataset/exp_id from config")
            print("[INFO] Auto grading will be skipped")
        else:
            print(f"[INFO] Auto-detected dataset: {dataset}")

    print("[INFO] Checking grade results...")
    run_auto_grading(run_dir, dataset=dataset or "leaf-classification")

    payload = load_payload(run_dir)
    index_path = write_site(payload, site_dir)

    print(f"[vis_node_by_tree_with_grade] run_dir: {run_dir}")
    print(f"[vis_node_by_tree_with_grade] nodes: {payload['node_count']}")
    print(f"[vis_node_by_tree_with_grade] matched: {payload['matched_nodes']}")
    print(f"[vis_node_by_tree_with_grade] has_grade_data: {payload['has_grade_data']}")
    print(f"[vis_node_by_tree_with_grade] html: {index_path}")

    if args.no_deploy:
        return 0

    server = _start_http_server(site_dir=site_dir, host=args.host, port=args.port)
    print(f"[vis_node_by_tree_with_grade] deployed: {server['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
