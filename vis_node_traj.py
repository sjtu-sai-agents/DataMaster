#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        if d.is_dir() and (d / "logs" / "uct_nodes").exists():
            candidates.append(d)

    if not candidates:
        raise SystemExit("no run with logs/uct_nodes found under runs/")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_trajectories(run_dir: Path) -> Dict[str, Any]:
    """加载 trajectory 文件，按 node 分组，每个 node 只保留最后一次执行的 trajectory"""
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        return {"trajectories": [], "by_exp": {}, "total_steps": 0, "exp_names": []}

    # 按 node 分组（从 exp_name 提取 node_id）
    # exp_name 格式: {stage}_{node_id[:8]}，例如 draft_c4639ce3
    trajectories_by_node: Dict[str, List[Dict[str, Any]]] = {}
    exp_names_set = set()

    # 遍历所有 trajectory 文件
    for task_dir in sorted(traj_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue
        traj_file = task_dir / "trajectory.json"
        if not traj_file.exists():
            continue

        try:
            traj_data = json.loads(traj_file.read_text(encoding="utf-8"))
            if isinstance(traj_data, list):
                for entry in traj_data:
                    if isinstance(entry, dict):
                        # 从 exp_name 提取 node_id
                        exp_name = entry.get("exp_name", "unknown")
                        # exp_name 格式: {stage}_{node_id[:8]}
                        # 例如: draft_c4639ce3, debug_64e0e08b
                        parts = exp_name.split("_")
                        if len(parts) >= 2:
                            node_id = parts[1]  # 提取 node_id 部分
                            if node_id not in trajectories_by_node:
                                trajectories_by_node[node_id] = []
                            trajectories_by_node[node_id].append(entry)

                        exp_names_set.add(exp_name)
        except Exception as e:
            print(f"[vis_node_traj] Failed to load trajectory {traj_file}: {e}")
            continue

    # 对每个 node，只保留最后一次执行（exp_index 最大的）
    all_trajectories: List[Dict[str, Any]] = []
    by_exp: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    for node_id, entries in trajectories_by_node.items():
        if not entries:
            continue

        # 按 exp_index 排序，取最后一次执行
        latest = max(entries, key=lambda e: e.get("exp_index", 0))
        all_trajectories.append(latest)

        # 按 exp_name 和 exp_index 组织
        exp_name = latest.get("exp_name", "unknown")
        exp_index = latest.get("exp_index", 0)

        if exp_name not in by_exp:
            by_exp[exp_name] = {}
        if exp_index not in by_exp[exp_name]:
            by_exp[exp_name][exp_index] = []
        by_exp[exp_name][exp_index].append(latest)

    total_steps = len(all_trajectories)
    exp_names = sorted(exp_names_set)

    return {
        "trajectories": all_trajectories,
        "by_exp": by_exp,
        "total_steps": total_steps,
        "exp_names": exp_names,
        "trajectory_dir": str(traj_dir),
    }


def _walk_tree_collect(node: Dict[str, Any], out: Dict[str, Dict[str, Any]]) -> None:
    node_id = node.get("id")
    if isinstance(node_id, str) and node_id:
        out[node_id] = node
    for c in node.get("children", []) or []:
        if isinstance(c, dict):
            _walk_tree_collect(c, out)


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _build_tree_from_nodes(nodes_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    shells: Dict[str, Dict[str, Any]] = {}
    for nid, raw in nodes_by_id.items():
        # Support both 'metric' and 'submission_score' field names
        metric_value = raw.get("metric")
        if metric_value is None:
            metric_value = raw.get("submission_score")

        shells[nid] = {
            "id": nid,
            "stage": raw.get("stage", "unknown"),
            "parent": raw.get("parent"),
            "metric": _as_float(metric_value),
            "maximize": raw.get("maximize", True),
            "uct_value": _as_float(raw.get("uct_value")),
            "visits": _as_int(raw.get("visits")),
            "reward": _as_float(raw.get("reward")),
            "total_reward": _as_float(raw.get("total_reward")),
            "is_buggy": raw.get("is_buggy"),
            "has_submission": raw.get("has_submission"),
            "snapshot_event": raw.get("snapshot_event"),
            "snapshot_ts": _as_float(raw.get("snapshot_ts")),
            "children": [],
        }

    roots = []
    for _nid, n in shells.items():
        pid = n.get("parent")
        if isinstance(pid, str) and pid in shells:
            shells[pid]["children"].append(n)
        else:
            roots.append(n)

    if len(roots) == 1:
        return roots[0]

    return {
        "id": "__root__",
        "stage": "root",
        "parent": None,
        "metric": None,
        "maximize": True,
        "uct_value": None,
        "visits": None,
        "reward": None,
        "total_reward": None,
        "is_buggy": False,
        "has_submission": False,
        "snapshot_event": "virtual_root",
        "snapshot_ts": None,
        "children": roots,
    }


def load_payload(run_dir: Path) -> Dict[str, Any]:
    nodes_dir = run_dir / "logs" / "uct_nodes"
    if not nodes_dir.exists():
        raise SystemExit(f"uct nodes dir not found: {nodes_dir}")

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    tree: Optional[Dict[str, Any]] = None

    tree_file = nodes_dir / "node.json"
    tree_obj = _read_json(tree_file) if tree_file.exists() else None
    if isinstance(tree_obj, dict):
        if isinstance(tree_obj.get("root"), dict):
            tree = tree_obj["root"]
        elif isinstance(tree_obj.get("tree"), dict):
            tree = tree_obj["tree"]

    for fp in sorted(nodes_dir.glob("*.json")):
        if fp.name == "node.json":
            continue
        obj = _read_json(fp)
        if not isinstance(obj, dict):
            continue
        node_id = obj.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue

        # Support both 'metric' and 'submission_score' field names
        metric_value = obj.get("metric")
        if metric_value is None:
            metric_value = obj.get("submission_score")
        obj["metric"] = _as_float(metric_value)

        obj["uct_value"] = _as_float(obj.get("uct_value"))
        obj["reward"] = _as_float(obj.get("reward"))
        obj["total_reward"] = _as_float(obj.get("total_reward"))
        obj["visits"] = _as_int(obj.get("visits"))
        obj["snapshot_ts"] = _as_float(obj.get("snapshot_ts"))
        obj["_file"] = str(fp)
        obj["_mtime"] = fp.stat().st_mtime
        nodes_by_id[node_id] = obj

    if tree is not None:
        tree_nodes: Dict[str, Dict[str, Any]] = {}
        _walk_tree_collect(tree, tree_nodes)
        for nid, n in tree_nodes.items():
            merged = dict(n)
            merged.update(nodes_by_id.get(nid, {}))
            nodes_by_id[nid] = merged
        if isinstance(tree.get("id"), str) and tree["id"] in nodes_by_id:
            tree = nodes_by_id[tree["id"]]
    else:
        tree = _build_tree_from_nodes(nodes_by_id)

    # 加载 trajectories
    trajectory_data = load_trajectories(run_dir)

    return {
        "generated_at": _now(),
        "run_dir": str(run_dir),
        "nodes_dir": str(nodes_dir),
        "node_count": len(nodes_by_id),
        "tree": tree,
        "nodes_by_id": nodes_by_id,
        "trajectories": trajectory_data,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>UCT Node + Trajectory Viewer</title>
  <script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>
  <style>
    :root { --bg:#0b1020; --panel:#131a2f; --text:#e7ebff; --muted:#95a0c7; --edge:#5f6b98; --accent:#4dabf7; }
    * { box-sizing:border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system; background:radial-gradient(1200px 600px at 10% 0%, #1f2d58 0%, var(--bg) 55%); color:var(--text); }
    .top { display:flex; justify-content:space-between; align-items:center; padding:12px 16px; background:rgba(8,11,24,.75); backdrop-filter: blur(6px); position:sticky; top:0; z-index:5; border-bottom:1px solid #2a355f; }
    .title { font-weight:700; letter-spacing:.3px; }
    .meta { color:var(--muted); font-size:13px; }
    .legend { color:var(--muted); font-size:12px; }
    .layout { display:flex; height:calc(100vh - 56px); }
    .left { flex:1; position:relative; }
    .right { width:480px; border-left:1px solid #2a355f; background:rgba(12,18,37,.7); overflow:auto; display:flex; flex-direction:column; }
    #svg { width:100%; height:100%; display:block; }
    .hint { position:absolute; left:12px; bottom:12px; color:var(--muted); font-size:12px; background:rgba(8,11,24,.7); padding:6px 8px; border-radius:8px; }

    .tabs { display:flex; background:rgba(8,11,24,.5); border-bottom:1px solid #2a355f; flex-shrink:0; }
    .tab { padding:10px 16px; cursor:pointer; font-size:13px; color:var(--muted); border:none; background:transparent; transition:all .2s; }
    .tab:hover { color:var(--text); background:rgba(77,171,247,.1); }
    .tab.active { color:var(--accent); background:rgba(77,171,247,.15); border-bottom:2px solid var(--accent); }

    .panel { padding:12px; overflow:auto; flex:1; }
    .card { background:var(--panel); border:1px solid #2c3760; border-radius:10px; padding:10px; margin-bottom:10px; }
    .k { color:#8ea3ff; font-size:12px; margin-bottom:4px; }
    .v { white-space:pre-wrap; word-break:break-word; font-size:13px; }
    .code { max-height:360px; overflow:auto; background:#0a1124; border:1px solid #33406f; border-radius:8px; padding:8px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }

    .node text { fill:#fff; font-size:10px; pointer-events:none; }
    .node circle { stroke:#d9e0ff; stroke-width:1.1; }
    .node.sel circle { stroke:#ffd166; stroke-width:2.4; }
    .link { fill:none; stroke:var(--edge); stroke-opacity:.55; stroke-width:1.1; }

    /* Trajectory styles */
    .traj-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding:8px 12px; background:rgba(77,171,247,.1); border-radius:8px; }
    .traj-title { font-weight:600; font-size:14px; color:var(--accent); }
    .traj-stats { font-size:12px; color:var(--muted); }
    .exp-group { margin-bottom:16px; }
    .exp-header { font-size:13px; font-weight:600; color:#8ea3ff; padding:6px 10px; background:rgba(142,163,255,.1); border-radius:6px; margin-bottom:8px; cursor:pointer; display:flex; justify-content:space-between; align-items:center; }
    .exp-header:hover { background:rgba(142,163,255,.15); }
    .exp-content { margin-left:8px; }
    .traj-entry { padding:8px 10px; background:rgba(19,26,47,.5); border-left:2px solid #4dabf7; border-radius:4px; margin-bottom:6px; font-size:12px; cursor:pointer; transition:all .2s; }
    .traj-entry:hover { background:rgba(77,171,247,.1); border-left-color:#ffd166; }
    .traj-entry.selected { background:rgba(77,171,247,.15); border-left-color:#ffd166; }
    .traj-step { font-weight:600; color:var(--accent); }
    .traj-meta { color:var(--muted); font-size:11px; margin-top:2px; }

    .message { background:rgba(10,17,36,.8); border:1px solid #2c3760; border-radius:8px; padding:10px; margin-bottom:8px; }
    .message-role { font-size:11px; font-weight:600; text-transform:uppercase; margin-bottom:4px; }
    .message-role.system { color:#ff6b6b; }
    .message-role.user { color:#4dabf7; }
    .message-role.assistant { color:#51cf66; }
    .message-role.tool { color:#ffd166; }
    .message-content { font-size:12px; white-space:pre-wrap; word-break:break-word; max-height:400px; overflow:auto; }

    .tool-call { background:rgba(255,209,102,.05); border-left:3px solid #ffd166; padding:8px; margin:8px 0; border-radius:4px; }
    .tool-name { font-weight:600; color:#ffd166; font-size:12px; }
    .tool-args { font-size:11px; color:var(--muted); margin-top:2px; }
    .tool-result { font-size:12px; margin-top:6px; padding:6px; background:rgba(0,0,0,.2); border-radius:4px; max-height:200px; overflow:auto; }
  </style>
</head>
<body>
  <div class=\"top\">
    <div class=\"title\">ML-Master Node + Trajectory Viewer</div>
    <div>
      <div class=\"meta\" id=\"meta\"></div>
      <div class=\"legend\">每 60 秒自动刷新 | 颜色: Bug(红) / 进行中(蓝) / 完成(绿) / 最高分(金)</div>
    </div>
  </div>
  <div class=\"layout\">
    <div class=\"left\">
      <svg id=\"svg\"></svg>
      <div class=\"hint\">Wheel: zoom · Drag: pan · Click node: details</div>
    </div>
    <div class=\"right\">
      <div class=\"tabs\">
        <button class=\"tab active\" onclick=\"switchTab('nodes')\">节点</button>
        <button class=\"tab\" onclick=\"switchTab('traj')\">轨迹</button>
      </div>
      <div class=\"panel\" id=\"panel-nodes\"></div>
      <div class=\"panel\" id=\"panel-traj\" style=\"display:none;\"></div>
    </div>
  </div>
  <script>
  const INITIAL_DATA = __DATA__;
  const REFRESH_MS = 60 * 1000;

  const panelNodes = document.getElementById('panel-nodes');
  const panelTraj = document.getElementById('panel-traj');
  const metaEl = document.getElementById('meta');
  const svg = d3.select('#svg');
  const g = svg.append('g');
  const zoom = d3.zoom().scaleExtent([0.08, 4]).on('zoom', (e)=>g.attr('transform', e.transform));
  svg.call(zoom);

  let DATA = INITIAL_DATA;
  let selectedNodeId = null;
  let currentTab = 'nodes';

  function fmt(v){
    if(v===null||v===undefined) return 'N/A';
    if(typeof v==='number') return Number.isFinite(v)? (String(v).includes('.')?v.toFixed(4):String(v)) : 'N/A';
    if(typeof v==='object') return JSON.stringify(v, null, 2);
    return String(v);
  }

  function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');

    if (tab === 'nodes') {
      panelNodes.style.display = 'block';
      panelTraj.style.display = 'none';
    } else {
      panelNodes.style.display = 'none';
      panelTraj.style.display = 'block';
      if (panelTraj.children.length === 0) {
        renderTrajectoryList();
      }
    }
  }

  function updateMeta(){
    const trajCount = DATA.trajectories?.total_steps || 0;
    metaEl.textContent = `run=${DATA.run_dir} | nodes=${DATA.node_count} | traj_steps=${trajCount} | generated_at=${new Date((DATA.generated_at||0)*1000).toLocaleString()}`;
  }

  function scoreForBest(node){
    const m = Number(node.metric);
    if(!Number.isFinite(m)) return null;
    const maximize = node.maximize !== false;
    return maximize ? m : -m;
  }

  function findBestNodeId(){
    let bestId = null;
    let bestScore = null;
    for(const [nid, n] of Object.entries(DATA.nodes_by_id || {})){
      if(n && n.is_buggy === true) continue;
      const sc = scoreForBest(n || {});
      if(sc === null) continue;
      if(bestScore === null || sc > bestScore){
        bestScore = sc;
        bestId = nid;
      }
    }
    return bestId;
  }

  function nodeColor(node, bestId){
    if(node.is_buggy === true) return '#ff4d6d';
    if(node.id === bestId) return '#facc15';

    const ev = (node.snapshot_event || '').toLowerCase();
    if(ev === 'completed') return '#22c55e';
    if(ev === 'created' || ev === 'updated' || ev === '' || ev === 'running') return '#4dabf7';

    return '#8893bd';
  }

  function renderDetail(id){
    const node = (DATA.nodes_by_id || {})[id] || {};
    panelNodes.innerHTML = '';

    const fields = [
      'id','stage','parent','snapshot_event','snapshot_ts',
      'metric','uct_value','visits','reward','total_reward',
      'is_buggy','has_submission','submission_file'
    ];

    fields.forEach(k=>{
      const card = document.createElement('div');
      card.className = 'card';
      const kk = document.createElement('div'); kk.className = 'k'; kk.textContent = k;
      const vv = document.createElement('div'); vv.className = 'v'; vv.textContent = fmt(node[k]);
      card.appendChild(kk);
      card.appendChild(vv);
      panelNodes.appendChild(card);
    });

    if(node.code !== undefined){
      const c = document.createElement('div'); c.className = 'card';
      c.innerHTML = '<div class="k">code</div>';
      const pre = document.createElement('pre'); pre.className = 'code'; pre.textContent = String(node.code || '');
      c.appendChild(pre);
      panelNodes.appendChild(c);
    }
  }

  function drawTree(){
    const bestId = findBestNodeId();
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
      .attr('class', d => d.data.id === selectedNodeId ? 'node sel' : 'node')
      .attr('transform', d=>`translate(${d.x},${d.y})`)
      .on('click', function(_, d){
        selectedNodeId = d.data.id;
        g.selectAll('.node').classed('sel', x => x.data.id === selectedNodeId);
        renderDetail(selectedNodeId);
      });

    nodes.append('circle')
      .attr('r', d=>{
        const v = Number(d.data.visits || 0);
        return Math.max(8, Math.min(22, 9 + Math.log10(v + 1) * 6));
      })
      .attr('fill', d=>nodeColor(d.data, bestId));

    nodes.append('text')
      .attr('text-anchor','middle')
      .attr('dominant-baseline','middle')
      .text(d=>{
        const m = Number(d.data.metric);
        return Number.isFinite(m) ? fmt(m) : 'N/A';
      });

    if(!selectedNodeId){
      selectedNodeId = DATA.tree && DATA.tree.id ? DATA.tree.id : null;
    }
    if(selectedNodeId){
      renderDetail(selectedNodeId);
      g.selectAll('.node').classed('sel', x => x.data.id === selectedNodeId);
    }
  }

  function renderTrajectoryList() {
    panelTraj.innerHTML = '';
    const trajData = DATA.trajectories || {};

    if (!trajData.by_exp || Object.keys(trajData.by_exp).length === 0) {
      panelTraj.innerHTML = '<div style="padding:20px;text-align:center;color:var(--muted);">暂无轨迹数据</div>';
      return;
    }

    const header = document.createElement('div');
    header.className = 'traj-header';
    header.innerHTML = `<span class="traj-title">轨迹概览</span><span class="traj-stats">总步数: ${trajData.total_steps || 0}</span>`;
    panelTraj.appendChild(header);

    for (const [expName, expData] of Object.entries(trajData.by_exp)) {
      const expGroup = document.createElement('div');
      expGroup.className = 'exp-group';

      const expHeader = document.createElement('div');
      expHeader.className = 'exp-header';
      const indexKeys = Object.keys(expData).map(Number).sort((a,b)=>a-b);
      expHeader.innerHTML = `<span>${expName}</span><span style="font-size:11px;color:var(--muted);">${indexKeys.length} 轮</span>`;

      const expContent = document.createElement('div');
      expContent.className = 'exp-content';

      indexKeys.forEach(idx => {
        const entries = expData[idx];
        if (!entries || entries.length === 0) return;

        const roundDiv = document.createElement('div');
        roundDiv.style.marginBottom = '8px';
        roundDiv.innerHTML = `<div style="font-size:11px;color:var(--muted);margin-bottom:4px;">第 ${idx} 轮 (${entries.length} 步)</div>`;

        entries.forEach(entry => {
          const entryDiv = document.createElement('div');
          entryDiv.className = 'traj-entry';
          entryDiv.innerHTML = `<span class="traj-step">Step ${entry.steps || '?'}</span><div class="traj-meta">${entry.exp_name || 'unknown'} | Status: ${entry.status || 'unknown'}</div>`;
          entryDiv.onclick = () => renderTrajectoryDetail(entry);
          roundDiv.appendChild(entryDiv);
        });

        expContent.appendChild(roundDiv);
      });

      expGroup.appendChild(expHeader);
      expGroup.appendChild(expContent);
      panelTraj.appendChild(expGroup);
    }
  }

  function renderTrajectoryDetail(entry) {
    panelTraj.innerHTML = '';

    const backBtn = document.createElement('button');
    backBtn.style.cssText = 'padding:8px 16px;margin-bottom:12px;background:var(--accent);border:none;border-radius:6px;color:#fff;cursor:pointer;font-size:13px;';
    backBtn.textContent = '← 返回列表';
    backBtn.onclick = renderTrajectoryList;
    panelTraj.appendChild(backBtn);

    const title = document.createElement('div');
    title.className = 'traj-header';
    title.innerHTML = `<span class="traj-title">${entry.exp_name || 'Unknown'} - Step ${entry.steps || '?'}</span><span class="traj-stats">${entry.status || 'Unknown'}</span>`;
    panelTraj.appendChild(title);

    const traj = entry.trajectory || {};
    const dialogs = traj.dialogs || [];

    if (dialogs.length > 0) {
      const lastDialog = dialogs[dialogs.length - 1];
      const messages = lastDialog.messages || [];

      messages.forEach((msg, i) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message';

        const roleDiv = document.createElement('div');
        roleDiv.className = `message-role ${msg.role || 'unknown'}`;
        roleDiv.textContent = `${msg.role || 'Unknown'} ${i+1}/${messages.length}`;
        msgDiv.appendChild(roleDiv);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
          msg.tool_calls.forEach(tc => {
            const toolDiv = document.createElement('div');
            toolDiv.className = 'tool-call';
            toolDiv.innerHTML = `<div class="tool-name">🔧 ${tc.function?.name || 'unknown'}</div>`;
            if (tc.function?.arguments) {
              const argsDiv = document.createElement('div');
              argsDiv.className = 'tool-args';
              argsDiv.textContent = tc.function.arguments;
              toolDiv.appendChild(argsDiv);
            }
            contentDiv.appendChild(toolDiv);
          });
        } else if (msg.role === 'tool') {
          const resultDiv = document.createElement('div');
          resultDiv.className = 'tool-result';
          resultDiv.textContent = msg.content || '(No output)';
          contentDiv.appendChild(resultDiv);
        } else if (msg.content) {
          contentDiv.textContent = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        } else {
          contentDiv.textContent = '(Empty)';
        }

        msgDiv.appendChild(contentDiv);
        panelTraj.appendChild(msgDiv);
      });
    }

    const stepsDiv = document.createElement('div');
    stepsDiv.className = 'card';
    stepsDiv.innerHTML = '<div class="k">Steps</div>';
    const stepsContent = document.createElement('div');
    stepsContent.className = 'v';
    stepsContent.textContent = JSON.stringify(traj.steps || [], null, 2);
    stepsDiv.appendChild(stepsContent);
    panelTraj.appendChild(stepsDiv);
  }

  async function refreshData(){
    try{
      const res = await fetch('/payload.json?t=' + Date.now(), {cache: 'no-store'});
      if(!res.ok) return;
      const next = await res.json();
      if(!next || !next.tree) return;

      const currentZoom = d3.zoomTransform(svg.node());
      DATA = next;
      updateMeta();
      drawTree();
      if (currentTab === 'traj' && panelTraj.children.length > 0) {
        renderTrajectoryList();
      }
      svg.call(zoom.transform, currentZoom);
    }catch(_){
      // keep current view on refresh failure
    }
  }

  function init(){
    updateMeta();
    drawTree();
    svg.call(zoom.transform, d3.zoomIdentity.translate(70,40).scale(1));
    setInterval(refreshData, REFRESH_MS);
  }

  init();
  </script>
</body>
</html>
"""


def write_payload(payload: Dict[str, Any], site_dir: Path) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    payload_path = site_dir / "payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
    server_pid_file = site_dir / ".vis_node_traj_server.pid"
    server_log_file = site_dir / "vis_node_traj_server.log"

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
        lf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start: {' '.join(cmd)}\n")
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


def _start_refresh_daemon(run_dir: Path, site_dir: Path, refresh_seconds: int) -> Dict[str, Any]:
    refresh_pid_file = site_dir / ".vis_node_traj_refresh.pid"
    refresh_log_file = site_dir / "vis_node_traj_refresh.log"

    _stop_old_process(refresh_pid_file)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-dir",
        str(run_dir),
        "--site-dir",
        str(site_dir),
        "--refresh-seconds",
        str(refresh_seconds),
        "--refresh-daemon",
        "--no-deploy",
    ]

    with refresh_log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start: {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, start_new_session=True)

    time.sleep(0.4)
    if proc.poll() is not None:
        raise SystemExit(f"refresh daemon failed, see log: {refresh_log_file}")

    refresh_pid_file.write_text(str(proc.pid), encoding="utf-8")
    return {
        "pid": proc.pid,
        "log": str(refresh_log_file),
        "pid_file": str(refresh_pid_file),
        "refresh_seconds": refresh_seconds,
    }


def auto_deploy(run_dir: Path, site_dir: Path, host: str, port: int, refresh_seconds: int) -> Dict[str, Any]:
    server = _start_http_server(site_dir=site_dir, host=host, port=port)
    refresher = _start_refresh_daemon(run_dir=run_dir, site_dir=site_dir, refresh_seconds=refresh_seconds)
    return {
        "server": server,
        "refresh": refresher,
    }


def run_refresh_loop(run_dir: Path, site_dir: Path, refresh_seconds: int) -> int:
    site_dir.mkdir(parents=True, exist_ok=True)
    interval = max(5, int(refresh_seconds))

    while True:
        try:
            payload = load_payload(run_dir)
            write_payload(payload, site_dir)
            print(
                f"[vis_node_traj:refresh] updated payload.json at {time.strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[vis_node_traj:refresh] update failed: {exc}", flush=True)
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate + auto-deploy ML-Master visualization with trajectory support")
    p.add_argument("--run-dir", type=str, default=None, help="target run dir (default: latest ml_master_* with logs/uct_nodes)")
    p.add_argument("--site-dir", type=str, default=None, help="output site dir (default: <run_dir>/logs/uct_nodes/vis_node_traj_site)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="http bind host")
    p.add_argument("--port", type=int, default=8788, help="preferred http port (default: 8788 to avoid conflict with vis_node)")
    p.add_argument("--refresh-seconds", type=int, default=60, help="payload refresh interval in seconds")
    p.add_argument("--no-deploy", action="store_true", help="only generate html, do not start server")
    p.add_argument("--refresh-daemon", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = discover_run_dir(args.run_dir)
    site_dir = Path(args.site_dir).expanduser().resolve() if args.site_dir else (run_dir / "logs" / "uct_nodes" / "vis_node_traj_site")

    if args.refresh_daemon:
        return run_refresh_loop(run_dir=run_dir, site_dir=site_dir, refresh_seconds=args.refresh_seconds)

    payload = load_payload(run_dir)
    index_path = write_site(payload, site_dir)

    print(f"[vis_node_traj] run_dir: {run_dir}")
    print(f"[vis_node_traj] html: {index_path}")

    if args.no_deploy:
        return 0

    dep = auto_deploy(
        run_dir=run_dir,
        site_dir=site_dir,
        host=args.host,
        port=args.port,
        refresh_seconds=args.refresh_seconds,
    )
    print(f"[vis_node_traj] deployed: {dep['server']['url']}")
    print(f"[vis_node_traj] server_pid: {dep['server']['pid']}")
    print(f"[vis_node_traj] server_log: {dep['server']['log']}")
    print(f"[vis_node_traj] refresh_pid: {dep['refresh']['pid']}")
    print(f"[vis_node_traj] refresh_log: {dep['refresh']['log']}")
    print(f"[vis_node_traj] refresh_interval: {dep['refresh']['refresh_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
