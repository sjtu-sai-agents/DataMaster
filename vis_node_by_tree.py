#!/usr/bin/env python3
"""按 UCT 树节点可视化 trajectory

python vis_node_by_tree.py --run-dir /data/yaxindu/datascientist/DataScientistEvomaster2/runs/ml_master_datatree_20260316_114412 --port 8788
从 node.json 读取节点结构，匹配 trajectory.json 中的对应数据。
每个节点只显示 steps 最大的 trajectory。
"""
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
        if d.is_dir() and (d / "logs" / "uct_nodes" / "node.json").exists():
            candidates.append(d)

    if not candidates:
        raise SystemExit("no run with node.json found under runs/")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def collect_node_ids(node: Dict[str, Any], ids: Optional[List[str]] = None) -> List[str]:
    """递归收集树中所有节点的 ID"""
    if ids is None:
        ids = []

    # 跳过虚拟根节点
    node_id = node.get("id", "")
    if node_id and node_id != "__virtual_root__":
        ids.append(node_id)

    for child in node.get("children", []):
        collect_node_ids(child, ids)
    return ids


def load_node_tree(run_dir: Path) -> Dict[str, Any]:
    """加载 UCT 节点树"""
    nodes_dir = run_dir / "logs" / "uct_nodes"

    # 尝试从 node.json 读取
    node_file = nodes_dir / "node.json"
    if node_file.exists():
        try:
            with open(node_file, 'r', encoding='utf-8') as f:
                tree = json.load(f)
            if tree:
                # 如果有 root 字段，使用它
                if isinstance(tree, dict) and "root" in tree:
                    return tree["root"]
                return tree
        except Exception as e:
            print(f"[WARNING] Failed to load node.json: {e}, building from individual files...")

    # 从单独的节点文件构建树
    print(f"[INFO] Building tree from individual node files...")

    # 读取所有节点文件
    nodes_by_id = {}
    for node_file in sorted(nodes_dir.glob("*.json")):
        if node_file.name == "node.json":
            continue
        try:
            with open(node_file, 'r', encoding='utf-8') as f:
                node = json.load(f)
                if node and node.get("id"):
                    nodes_by_id[node["id"]] = node
        except Exception as e:
            continue

    print(f"[INFO] Loaded {len(nodes_by_id)} node files")

    if not nodes_by_id:
        raise SystemExit(f"No valid node files found in {nodes_dir}")

    # 构建树结构
    roots = []
    for node_id, node in nodes_by_id.items():
        parent_id = node.get("parent")
        if not parent_id or parent_id not in nodes_by_id:
            roots.append(node)

    # 递归构建 children
    def build_children(node):
        node["children"] = [
            build_children(nodes_by_id[cid])
            for cid, child in nodes_by_id.items()
            if child.get("parent") == node["id"]
        ]
        return node

    tree_roots = [build_children(root) for root in roots]

    # 如果只有一个根节点，直接返回；否则创建虚拟根
    if len(tree_roots) == 1:
        return tree_roots[0]

    return {
        "id": "__virtual_root__",
        "stage": "root",
        "children": tree_roots,
    }


def match_trajectories_to_nodes(run_dir: Path, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """匹配 trajectory entries 到节点

    对每个节点：
    1. 找到所有匹配的 trajectory entries（agent_id的第一部分匹配node_id）
    2. 只保留 steps 最大的那个
    3. 提取 messages 信息
    """
    traj_dir = run_dir / "trajectories"
    if not traj_dir.exists():
        return {}

    # 按 node_id 分组 trajectory entries
    trajectories_by_node: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in node_ids}

    for task_dir in sorted(traj_dir.glob("task_*")):
        if not task_dir.is_dir():
            continue
        traj_file = task_dir / "trajectory.json"
        if not traj_file.exists():
            continue

        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for entry in data:
                if not isinstance(entry, dict):
                    continue

                agent_id = entry.get("agent_id", "")
                # agent_id 格式: {task_id}_{agent_name}
                # 提取 task_id（第一部分）
                parts = agent_id.split('_')
                if not parts:
                    continue

                potential_node_id = parts[0]

                # 匹配到已知节点
                for node_id in node_ids:
                    if node_id.startswith(potential_node_id) or potential_node_id.startswith(node_id[:8]):
                        trajectories_by_node[node_id].append(entry)
                        break

        except Exception as e:
            print(f"[WARNING] Failed to load trajectory from {traj_file}: {e}")
            continue

    # 对每个节点，只保留 steps 最大的 entry
    result: Dict[str, Dict[str, Any]] = {}

    for node_id, entries in trajectories_by_node.items():
        if not entries:
            continue

        # 按 steps 排序，取最大的
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
            }
        }

    return result


def load_payload(run_dir: Path) -> Dict[str, Any]:
    """加载可视化数据"""
    # 加载节点树
    tree = load_node_tree(run_dir)

    # 收集所有节点ID
    node_ids = collect_node_ids(tree)

    # 匹配 trajectory
    node_trajectories = match_trajectories_to_nodes(run_dir, node_ids)

    return {
        "generated_at": _now(),
        "run_dir": str(run_dir),
        "node_count": len(node_ids),
        "tree": tree,
        "node_trajectories": node_trajectories,
        "matched_nodes": len(node_trajectories),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>UCT Node + Trajectory Viewer (By Tree)</title>
  <script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>
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
  <div class=\"top\">
    <div class=\"title\">UCT Node + Trajectory (By Tree)</div>
    <div>
      <div class=\"meta\" id=\"meta\"></div>
    </div>
  </div>
  <div class=\"layout\">
    <div class=\"left\">
      <svg id=\"svg\"></svg>
      <div class=\"hint\">Wheel: zoom · Drag: pan · Click node: trajectory | Solid: has data · Dashed: no data</div>
    </div>
    <div class=\"right\"><div class=\"panel\" id=\"panel\"></div></div>
  </div>
  <script>
  const DATA = __DATA__;

  const panel = document.getElementById('panel');
  const metaEl = document.getElementById('meta');
  const svg = d3.select('#svg');
  const g = svg.append('g');
  const zoom = d3.zoom().scaleExtent([0.08, 4]).on('zoom', (e)=>g.attr('transform', e.transform));
  svg.call(zoom);

  let selectedNodeId = null;

  function updateMeta(){
    metaEl.textContent = `run=${DATA.run_dir} | nodes=${DATA.node_count} | with_traj=${DATA.matched_nodes}`;
  }

  function nodeColor(node, trajData) {
    if (!trajData[node.id]) return '#8893bd';  // 无 trajectory
    if (node.is_buggy) return '#ff4d6d';  // buggy
    if (node.stage === 'root') return '#ffd166';  // root
    const stage = node.stage || '';
    if (stage === 'initial') return '#4dabf7';  // initial
    if (stage === 'black') return '#51cf66';  // black
    if (stage === 'red') return '#cc5de8';  // red
    if (stage === 'terminal') return '#868e96';  // terminal
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

    // 节点信息
    const nodeCard = document.createElement('div');
    nodeCard.className = 'card';
    nodeCard.innerHTML = '<div class=\"k\">Node Info</div>';

    // 找到节点
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
      nodeCard.innerHTML += `<div class="v">Metric: ${node.metric !== null ? node.metric : 'N/A'}</div>`;
      nodeCard.innerHTML += `<div class="v">Submission Score: ${node.submission_score !== null ? node.submission_score : 'N/A'}</div>`;
      nodeCard.innerHTML += `<div class="v">Has Submission: ${node.has_submission ? 'Yes' : 'No'}</div>`;

      nodeCard.innerHTML += `<div class="v">Buggy: ${node.is_buggy ? 'Yes' : 'No'}</div>`;
      nodeCard.innerHTML += `<div class="v">Valid: ${node.is_valid !== null ? (node.is_valid ? 'Yes' : 'No') : 'N/A'}</div>`;
    }
    panel.appendChild(nodeCard);

    // Trajectory 信息
    if (traj) {
      const trajCard = document.createElement('div');
      trajCard.className = 'card';
      trajCard.innerHTML = '<div class=\"k\">Trajectory</div>';
      trajCard.innerHTML += `<div class="v">Task: ${traj.task_id || 'N/A'}</div>`;
      trajCard.innerHTML += `<div class="v">Exp: ${traj.exp_name || 'N/A'}</div>`;
      trajCard.innerHTML += `<div class="v">Steps: ${traj.steps || 0}</div>`;
      panel.appendChild(trajCard);

      // 显示对话历史
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
        // root 节点显示阶段标识，其它节点显示 metric
        if (d.data.stage === 'root') {
          return 'root';
        }
        const m = Number(d.data.metric);
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

    // 默认选择第一个有 trajectory 的节点
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

    // Structured format (OpenAI / internal object)
    if (typeof tc === 'object') {
      const fn = tc.function || {};
      const name = fn.name || tc.name || 'unknown';
      const args = fn.arguments !== undefined ? fn.arguments : (tc.arguments !== undefined ? tc.arguments : '');
      return { name, args };
    }

    // String format, e.g.
    // "id='...' type='function' function=FunctionCall(name='execute_bash', arguments='{\"command\":\"pwd\"}')"
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

    // 处理 tool_calls
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

    // 处理普通 content
    if (msg.content) {
      if (typeof msg.content === 'string') {
        const textDiv = document.createElement('div');
        textDiv.textContent = msg.content || '(empty)';
        textDiv.style.marginTop = toolCalls.length > 0 ? '8px' : '0';
        contentDiv.appendChild(textDiv);
      } else if (Array.isArray(msg.content)) {
        // 多模态内容
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

    // 对于 tool 角色，显示工具名称
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
    server_pid_file = site_dir / ".vis_node_by_tree_server.pid"
    server_log_file = site_dir / "vis_node_by_tree_server.log"

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize UCT nodes with trajectory data matched by node ID")
    p.add_argument("--run-dir", type=str, default=None, help="target run dir")
    p.add_argument("--site-dir", type=str, default=None, help="output site dir")
    p.add_argument("--host", type=str, default="127.0.0.1", help="http bind host")
    p.add_argument("--port", type=int, default=8789, help="http port")
    p.add_argument("--no-deploy", action="store_true", help="only generate html")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = discover_run_dir(args.run_dir)
    site_dir = Path(args.site_dir).expanduser().resolve() if args.site_dir else (run_dir / "logs" / "uct_nodes" / "vis_node_by_tree_site")

    payload = load_payload(run_dir)
    index_path = write_site(payload, site_dir)

    print(f"[vis_node_by_tree] run_dir: {run_dir}")
    print(f"[vis_node_by_tree] nodes: {payload['node_count']}")
    print(f"[vis_node_by_tree] matched: {payload['matched_nodes']}")
    print(f"[vis_node_by_tree] html: {index_path}")

    if args.no_deploy:
        return 0

    server = _start_http_server(site_dir=site_dir, host=args.host, port=args.port)
    print(f"[vis_node_by_tree] deployed: {server['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())