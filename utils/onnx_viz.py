#!/usr/bin/env python3

# Visualize an ONNX model graph and save it as an image.

# Usage:
#     python onnx_viz.py path/to/model.onnx --out model.png --format png --infer-shapes

# Requires:
#     - onnx (pip install onnx)
#     - graphviz Python package (pip install graphviz)
#     - Graphviz system binaries installed and on PATH (https://graphviz.org/download/)

import argparse
import os
from typing import Dict, Tuple, Optional, Set

import onnx
from onnx import numpy_helper, shape_inference  # type: ignore

try:
    from graphviz import Digraph  # type: ignore
except Exception as e:
    raise SystemExit("not installed graphviz")


def validate_onnx_file(file_path: str) -> bool:
    """Validate if the file is a valid ONNX file."""
    try:
        # Check file extension
        if not file_path.lower().endswith('.onnx'):
            print(f"[WARN] File does not have .onnx extension: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"[ERROR] File is empty: {file_path}")
            return False
        
        print(f"[INFO] File size: {file_size} bytes")
        
        # Try to read first few bytes to check if it looks like protobuf
        with open(file_path, 'rb') as f:
            first_bytes = f.read(16)
            print(f"[INFO] First 16 bytes (hex): {first_bytes.hex()}")
            
        return True
    except Exception as e:
        print(f"[ERROR] File validation failed: {e}")
        return False


def tensor_shape_str(t_type: Optional[onnx.TypeProto]) -> str:
    if t_type is None or not t_type.HasField("tensor_type"):
        return "?"
    tt = t_type.tensor_type
    if not tt.HasField("shape"):
        return "?"
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return "x".join(dims) if dims else "?"


def collect_value_info_types(model: onnx.ModelProto) -> Dict[str, onnx.TypeProto]:
    """Build a map from value name -> TypeProto (to get shapes/dtypes)."""
    vi_map: Dict[str, onnx.TypeProto] = {}
    g = model.graph

    def add_all(vinfos):
        for v in vinfos:
            if hasattr(v, "type") and v.name:
                vi_map[v.name] = v.type

    add_all(g.input)
    add_all(g.output)
    add_all(g.value_info)

    # Also try to add initializer info if present
    for init in g.initializer:
        if init.name and init.dims:
            # Create a synthetic TypeProto
            t = onnx.helper.make_tensor_type_proto(init.data_type, list(init.dims))
            vi_map.setdefault(init.name, t)

    return vi_map


def make_graph(model: onnx.ModelProto, format: str = "png", rankdir: str = "LR") -> Digraph:
    g = model.graph
    vi_map = collect_value_info_types(model)

    dot = Digraph(comment=f"ONNX Graph: {g.name}")
    dot.attr(rankdir=rankdir, fontsize="12", labelloc="t", label=f"{g.name or 'ONNX Model'}\\nIR v{model.ir_version}, opset {','.join(str(op.version) for op in model.opset_import)}")

    # Styles
    dot.attr("node", shape="record", fontsize="10", style="rounded,filled", fillcolor="white" )

    # Track producers to connect edges
    produces: Dict[str, str] = {}  # tensor name -> node_id
    seen_nodes: Set[str] = set()

    # Add graph inputs as separate nodes
    for inp in g.input:
        node_id = f"input:{inp.name}"
        if node_id in seen_nodes: 
            continue
        seen_nodes.add(node_id)
        shape = tensor_shape_str(vi_map.get(inp.name))
        dot.node(node_id, label=f"{{<p0> 🡲 INPUT|{inp.name}|shape: {shape}}}", shape="record", fillcolor="#e3f2fd")
        produces[inp.name] = node_id

    # Add initializers as CONST nodes (weights)
    init_names = {init.name for init in g.initializer}
    for init in g.initializer:
        node_id = f"const:{init.name}"
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        try:
            dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type].__name__  # type: ignore
        except Exception:
            dtype = str(init.data_type)
        shape = "x".join(map(str, init.dims)) if init.dims else "?"
        dot.node(node_id, label=f"{{<p0> ⚙ CONST|{init.name}|{dtype}|{shape}}}", shape="record", fillcolor="#fff3e0")
        produces[init.name] = node_id

    # Add nodes
    for idx, node in enumerate(g.node):
        nid = node.name if node.name else f"{node.op_type}_{idx}"
        # Node label with op type and (optional) name
        title = node.op_type if not node.name else f"{node.op_type}\\n{name_short(node.name)}"
        # Output shapes (first output shown)
        out_shapes = []
        for outp in node.output:
            out_shapes.append(tensor_shape_str(vi_map.get(outp)))
        shape_str = ", ".join(out_shapes) if out_shapes else "?"

        label = f"{{<p0> {title}|shape: {shape_str}}}"
        dot.node(nid, label=label, shape="record", fillcolor="#f5f5f5")

        # Connect inputs
        for i, inp in enumerate(node.input):
            if not inp:
                continue
            src = produces.get(inp)
            if src is None:
                # This input may come from an external graph value (missing value_info). Create a placeholder.
                ph_id = f"ph:{inp}"
                if ph_id not in seen_nodes:
                    seen_nodes.add(ph_id)
                    dot.node(ph_id, label=f"{{<p0> ?|{inp}}}", shape="oval", fillcolor="#ede7f6")
                    produces[inp] = ph_id
                src = ph_id
            dot.edge(src, nid, label="")  # no label by default

        # Register producers for outputs
        for outp in node.output:
            if outp:
                produces[outp] = nid

        # Flag control-flow subgraphs (not rendered inline for simplicity)
        for attr in node.attribute:
            if hasattr(attr, 'g') and attr.HasField("g"):
                dot.node(f"{nid}_subg", label="{control-flow subgraph}", shape="note", fillcolor="#e8f5e9")
                dot.edge(nid, f"{nid}_subg", style="dashed")

    # Add graph outputs
    for out in g.output:
        node_id = f"output:{out.name}"
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        shape = tensor_shape_str(vi_map.get(out.name))
        dot.node(node_id, label=f"{{<p0> 🡲 OUTPUT|{out.name}|shape: {shape}}}", shape="record", fillcolor="#e1f5fe")
        src = produces.get(out.name)
        if src:
            dot.edge(src, node_id)

    return dot


def name_short(name: str, maxlen: int = 32) -> str:
    if len(name) <= maxlen:
        return name
    return name[:maxlen-3] + "..."


def main():
    p = argparse.ArgumentParser(description="Visualize ONNX model graph as an image (uses Graphviz)." )
    p.add_argument("model", help="Path to .onnx file")
    p.add_argument("--out", default=None, help="Output image file path. If omitted, uses <model>.<format> in the same folder.")
    p.add_argument("--format", default="png", choices=["png", "svg", "pdf"], help="Output format (default: png)")
    p.add_argument("--rankdir", default="LR", choices=["LR", "TB", "BT", "RL"], help="Graph layout direction (default: LR)" )
    p.add_argument("--infer-shapes", action="store_true", help="Run ONNX shape inference to enrich tensor shapes in the diagram.")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"[ERROR] File not found: {args.model}")

    # Validate ONNX file
    if not validate_onnx_file(args.model):
        raise SystemExit(f"[ERROR] Invalid ONNX file: {args.model}")

    try:
        print(f"[INFO] Loading ONNX model: {args.model}")
        model = onnx.load(args.model)
        print(f"[INFO] Successfully loaded ONNX model")
        
        # Check model
        try:
            onnx.checker.check_model(model)
            print(f"[INFO] Model validation passed")
        except Exception as e:
            print(f"[WARN] Model validation failed: {e}")
            print(f"[INFO] Continuing anyway...")
            
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        print(f"[INFO] This might be due to:")
        print(f"  - Corrupted ONNX file")
        print(f"  - Unsupported ONNX version")
        print(f"  - File is not in ONNX format")
        raise SystemExit(f"[ERROR] Cannot load model: {args.model}")

    if args.infer_shapes:
        try:
            print(f"[INFO] Running shape inference...")
            model = shape_inference.infer_shapes(model)
            print(f"[INFO] Shape inference completed")
        except Exception as e:
            print(f"[WARN] Shape inference failed: {e}")

    try:
        dot = make_graph(model, format=args.format, rankdir=args.rankdir)

        out_path = args.out
        if out_path is None:
            base, _ = os.path.splitext(args.model)
            out_path = f"{base}.{args.format}"

        # Render
        # graphviz.Digraph.render wants a filename without extension and a format.
        fname, ext = os.path.splitext(out_path)
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        dot.render(filename=fname, directory=out_dir, format=args.format, cleanup=True)
        print(f"[INFO] Saved: {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate visualization: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()