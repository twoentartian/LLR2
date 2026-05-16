from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import json
import pprint
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch.fx import GraphModule, Node, symbolic_trace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.adapters import DiffusionAdapter, StandardAdapter
from py_src.ml_setup import ApplicationType, MLSetup, get_ml_setup_from_config
from py_src.ml_setup.dataloader_util import DataloaderConfig


@dataclass
class ModuleRecord:
    name: str
    type_name: str
    input_desc: Optional[str] = None
    output_desc: Optional[str] = None


MERMAID_INIT = {
    "flowchart": {
        "htmlLabels": True,
        "curve": "linear",
        "nodeSpacing": 18,
        "rankSpacing": 28,
    },
    "themeVariables": {
        "fontSize": "10px",
    },
}


def _flatten_nodes(value: Any) -> list[Node]:
    nodes: list[Node] = []
    if isinstance(value, Node):
        nodes.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            nodes.extend(_flatten_nodes(item))
    elif isinstance(value, dict):
        for item in value.values():
            nodes.extend(_flatten_nodes(item))
    return nodes


def _module_type_name(graph_module: GraphModule, target: str) -> str:
    return type(graph_module.get_submodule(target)).__name__


def _wrap_dotted_name(name: str, parts_per_line: int = 2) -> str:
    parts = name.split(".")
    if len(parts) <= parts_per_line:
        return name

    chunks = []
    for index in range(0, len(parts), parts_per_line):
        chunks.append(".".join(parts[index:index + parts_per_line]))
    return "<br/>".join(chunks)


def _strip_diagram_wrapper_prefix(name: str) -> str:
    prefixes = (
        "train_diffusion.model.",
        "train_diffusion.",
        "model.",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _diagram_group_name(target: str) -> str:
    name = _strip_diagram_wrapper_prefix(target)
    parts = name.split(".")
    if len(parts) <= 1:
        return name

    if parts[0] in {"downs", "ups"}:
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            return ".".join(parts[:3])
        if len(parts) >= 2 and parts[1].isdigit():
            return ".".join(parts[:2])
        return parts[0]

    if parts[0].startswith("mid_") or parts[0].startswith("final_") or parts[0] in {"init_conv", "time_mlp"}:
        return parts[0]

    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        return ".".join(parts[:3])
    if len(parts) >= 2 and parts[1].isdigit():
        return ".".join(parts[:2])
    if len(parts) >= 2 and parts[1] in {"tree1", "tree2", "root", "project"}:
        return ".".join(parts[:2])
    return parts[0]


def _call_module_order_from_fx(graph_module: GraphModule) -> list[ModuleRecord]:
    records: list[ModuleRecord] = []
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        records.append(ModuleRecord(name=str(node.target), type_name=_module_type_name(graph_module, str(node.target))))
    return records


def _tensor_meta_desc(meta: Any) -> Optional[str]:
    shape = getattr(meta, "shape", None)
    if shape is None:
        return None
    return str(tuple(shape))


def _node_shape_desc(node: Node) -> Optional[str]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is not None:
        return _tensor_meta_desc(tensor_meta)

    val = node.meta.get("val")
    if torch.is_tensor(val):
        return str(tuple(val.shape))
    if isinstance(val, (list, tuple)):
        parts = []
        for item in val:
            if torch.is_tensor(item):
                parts.append(str(tuple(item.shape)))
        if parts:
            return "[" + ", ".join(parts) + "]"
    return None


def _escape_mermaid(text: str) -> str:
    return text.replace('"', "'")

def _compact_group_for_fx_node(graph_module: GraphModule, node: Node, memo: dict[Node, str]) -> str:
    cached = memo.get(node)
    if cached is not None:
        return cached

    if node.op == "placeholder":
        group = "input"
    elif node.op == "output":
        group = "output"
    elif node.op == "call_module":
        group = f"module:{_diagram_group_name(str(node.target))}"
    else:
        user_groups = sorted(
            {
                _compact_group_for_fx_node(graph_module, user, memo)
                for user in node.users
            }
        )
        if len(user_groups) == 1:
            group = user_groups[0]
        elif node.op == "call_function":
            target = getattr(node.target, "__name__", repr(node.target))
            group = f"op:{target}:{node.name}"
        elif node.op == "call_method":
            group = f"op:{node.target}:{node.name}"
        elif node.op == "get_attr":
            group = f"attr:{node.target}"
        else:
            group = f"{node.op}:{node.name}"

    memo[node] = group
    return group


def _compact_group_label(graph_module: GraphModule, group_name: str, group_nodes: list[Node]) -> str:
    if group_name == "input":
        return "input"
    if group_name == "output":
        return "output"
    if group_name.startswith("module:"):
        module_name = group_name.removeprefix("module:")
        label = _wrap_dotted_name(module_name)
        shape_desc = _node_shape_desc(group_nodes[-1])
        if shape_desc is not None:
            label += f"<br/>{shape_desc}"
        return label
    if group_name.startswith("op:"):
        _, op_name, _ = group_name.split(":", 2)
        label = f"{op_name}<br/>function"
        shape_desc = _node_shape_desc(group_nodes[-1])
        if shape_desc is not None:
            label += f"<br/>{shape_desc}"
        return label
    return group_name


def _build_compact_fx_mermaid(graph_module: GraphModule) -> str:
    lines = [f"%%{{init: {json.dumps(MERMAID_INIT)} }}%%", "flowchart TB"]
    group_memo: dict[Node, str] = {}
    groups_in_order: list[str] = []
    group_nodes: dict[str, list[Node]] = {}

    for node in graph_module.graph.nodes:
        group_name = _compact_group_for_fx_node(graph_module, node, group_memo)
        if group_name not in group_nodes:
            groups_in_order.append(group_name)
            group_nodes[group_name] = []
        group_nodes[group_name].append(node)

    group_ids = {group_name: f"g{index}" for index, group_name in enumerate(groups_in_order)}
    for group_name in groups_in_order:
        label = _escape_mermaid(_compact_group_label(graph_module, group_name, group_nodes[group_name]))
        lines.append(f'    {group_ids[group_name]}["{label}"]')

    seen_edges: set[tuple[str, str]] = set()
    for node in graph_module.graph.nodes:
        dst_group = group_memo[node]
        for src in _flatten_nodes(node.args) + _flatten_nodes(node.kwargs):
            src_group = group_memo[src]
            if src_group == dst_group:
                continue
            edge = (src_group, dst_group)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            lines.append(f"    {group_ids[src_group]} --> {group_ids[dst_group]}")

    return "\n".join(lines)


def _is_leaf_module(module: torch.nn.Module) -> bool:
    return len(list(module.children())) == 0


def _shape_desc_from_value(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, dict):
        items = [f"{key}:{_shape_desc_from_value(item)}" for key, item in value.items()]
        return "{" + ", ".join(items) + "}"
    if isinstance(value, (list, tuple)):
        items = [_shape_desc_from_value(item) for item in value]
        left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
        return left + ", ".join(items) + right
    return type(value).__name__


def _record_runtime_module_order(ml_setup: MLSetup, batch: Any) -> list[ModuleRecord]:
    model = ml_setup.model
    adapter = ml_setup.adapter
    model.eval()
    model.to("cpu")

    records: list[ModuleRecord] = []
    handles = []

    for name, module in model.named_modules():
        if not name or not _is_leaf_module(module):
            continue

        def hook(mod, inputs, output, *, module_name=name):
            records.append(
                ModuleRecord(
                    name=module_name,
                    type_name=type(mod).__name__,
                    input_desc=_shape_desc_from_value(inputs),
                    output_desc=_shape_desc_from_value(output),
                )
            )

        handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            adapter.val_step(batch, 0, torch.device("cpu"))
    finally:
        for handle in handles:
            handle.remove()

    return records


def _module_state_names(
    model: torch.nn.Module,
    module_name: str,
    include_buffers: bool,
) -> list[str]:
    module = model.get_submodule(module_name)
    names = [f"{module_name}.{name}" for name, _ in module.named_parameters(recurse=False)]
    if include_buffers:
        names.extend(f"{module_name}.{name}" for name, _ in module.named_buffers(recurse=False))
    return names


def _state_order_from_module_order(
    model: torch.nn.Module,
    module_records: Iterable[ModuleRecord],
    include_buffers: bool,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in module_records:
        for state_name in _module_state_names(model, record.name, include_buffers):
            if state_name in seen:
                continue
            seen.add(state_name)
            ordered.append(state_name)
    return ordered

def _build_compact_linear_mermaid(module_records: list[ModuleRecord]) -> str:
    lines = [f"%%{{init: {json.dumps(MERMAID_INIT)} }}%%", "flowchart TB", '    input["input"]']
    grouped_records: list[ModuleRecord] = []
    for record in module_records:
        group_name = _diagram_group_name(record.name)
        if grouped_records and grouped_records[-1].name == group_name:
            grouped_records[-1].output_desc = record.output_desc
            continue
        grouped_records.append(
            ModuleRecord(
                name=group_name,
                type_name=record.type_name,
                input_desc=record.input_desc,
                output_desc=record.output_desc,
            )
        )

    previous = "input"
    for index, record in enumerate(grouped_records):
        node_id = f"m{index}"
        label = _wrap_dotted_name(record.name)
        if record.output_desc is not None:
            label += f"<br/>{record.output_desc}"
        lines.append(f'    {node_id}["{_escape_mermaid(label)}"]')
        lines.append(f"    {previous} --> {node_id}")
        previous = node_id
    lines.append('    output["output"]')
    lines.append(f"    {previous} --> output")
    return "\n".join(lines)


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return None


def _dataset_image_shape(ml_setup: MLSetup, split: str) -> Optional[tuple[int, ...]]:
    batch = _batch_from_dataloader(ml_setup, split)
    sample = batch[0] if isinstance(batch, (tuple, list)) and len(batch) > 0 else batch
    tensor = _first_tensor(sample)
    if tensor is None:
        return None
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        return tuple(tensor.shape[1:])
    return tuple(tensor.shape)


def _synthetic_batch(ml_setup: MLSetup, split: str) -> Any:
    image_shape = _dataset_image_shape(ml_setup, split)

    if ml_setup.application_type in (ApplicationType.classifier, ApplicationType.diffusion) and image_shape is not None:
        data = torch.randn(1, *image_shape)
        label = torch.zeros(1, dtype=torch.long)
        return (data, label)

    raise RuntimeError(
        "Synthetic batch generation is not implemented for "
        f"application_type={ml_setup.application_type.name} dataset_type={ml_setup.dataset_type.name}."
    )


def _batch_from_dataloader(ml_setup: MLSetup, split: str) -> Any:
    config = DataloaderConfig(
        batch_size=1,
        num_workers=0,
        num_samples=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=None,
        persistent_workers=False,
    )
    dataloader = ml_setup.train_dataloader(config) if split == "train" else ml_setup.val_dataloader(config)
    return next(iter(dataloader))


def _get_runtime_batch(ml_setup: MLSetup, input_source: str, split: str) -> Any:
    if input_source == "data":
        return _batch_from_dataloader(ml_setup, split)
    if input_source == "synthetic":
        return _synthetic_batch(ml_setup, split)
    if input_source != "auto":
        raise ValueError(f"Unsupported input source: {input_source}")

    try:
        return _batch_from_dataloader(ml_setup, split)
    except Exception:
        return _synthetic_batch(ml_setup, split)


def _model_args_from_batch(ml_setup: MLSetup, batch: Any) -> Optional[tuple[Any, ...]]:
    adapter = ml_setup.adapter
    if isinstance(adapter, (StandardAdapter, DiffusionAdapter)):
        if not isinstance(batch, (tuple, list)) or len(batch) == 0:
            return None
        return (batch[0],)
    if torch.is_tensor(batch):
        return (batch,)
    return None


def _maybe_shape_propagate(graph_module: GraphModule, model_args: Optional[tuple[Any, ...]]) -> None:
    if model_args is None:
        return

    try:
        from torch.fx.passes.shape_prop import ShapeProp

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ShapeProp(graph_module).propagate(*model_args)
    except Exception:
        return


def _format_module_order(module_records: list[ModuleRecord]) -> str:
    lines = ["Module execution order:"]
    for index, record in enumerate(module_records, start=1):
        suffix = ""
        if record.input_desc is not None or record.output_desc is not None:
            suffix = f"  [{record.input_desc} -> {record.output_desc}]"
        lines.append(f"{index}. {record.name} ({record.type_name}){suffix}")
    return "\n".join(lines)


def _format_state_order(state_names: list[str], include_buffers: bool) -> str:
    title = "State dict key order (parameters + buffers):" if include_buffers else "State dict key order (parameters only):"
    lines = [title]
    for index, name in enumerate(state_names, start=1):
        lines.append(f"{index}. {name}")
    return "\n".join(lines)


_INTERNAL_PHASE_COMPONENT_PATTERNS = (
    re.compile(r"^(conv|bn|norm|relu|act|dropout|linear|fc)\d*$"),
    re.compile(r"^(block|mlp|to_qkv|to_kv|to_q|to_k|to_v|to_out|attend|res_conv|proj)(\d+)?$"),
)


def _is_internal_phase_component(part: str) -> bool:
    return any(pattern.match(part) for pattern in _INTERNAL_PHASE_COMPONENT_PATTERNS)


def _should_keep_terminal_numeric(parts: list[str]) -> bool:
    return (
        len(parts) >= 3
        and parts[-1].isdigit()
        and parts[-2].isdigit()
        and parts[-3] in {"downs", "ups"}
    )


def _phase_prefix_from_state_name(state_name: str) -> str:
    parts = state_name.split(".")
    if len(parts) <= 1:
        return state_name

    parts = parts[:-1]
    if len(parts) > 1 and parts[-1].isdigit() and not _should_keep_terminal_numeric(parts):
        parts.pop()
    while len(parts) > 1 and _is_internal_phase_component(parts[-1]):
        parts.pop()
    return ".".join(parts)


def _recommended_phase_prefixes_from_state_order(state_names: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for state_name in state_names:
        prefix = _phase_prefix_from_state_name(state_name)
        if prefix in seen:
            continue
        seen.add(prefix)
        ordered.append(prefix)
    return ordered


def _write_text_if_requested(path: Optional[str], content: str) -> None:
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
        if not content.endswith("\n"):
            file.write("\n")


def _default_output_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "print_layer_order_output"))


def _default_render_output_path(ml_setup: MLSetup, suffix: str) -> str:
    file_name = f"{ml_setup.model_type.name}_{ml_setup.dataset_type.name}_flow.{suffix}"
    return os.path.join(_default_output_dir(), file_name)


def _txt_output_path_from_pdf_path(pdf_path: str) -> str:
    base, _ = os.path.splitext(pdf_path)
    return f"{base}.txt"


def _py_output_path_from_pdf_path(pdf_path: str) -> str:
    base, _ = os.path.splitext(pdf_path)
    return f"{base}.py"


def _format_phase_prefixes_python(phase_prefixes: list[str]) -> str:
    return "FDF_PHASE_PREFIXES = " + pprint.pformat(phase_prefixes, width=100) + "\n"


def _resolve_optional_output_path(raw_value: Optional[str], default_path: str) -> Optional[str]:
    if raw_value is None:
        return None
    if raw_value == "":
        return default_path
    return os.path.abspath(raw_value)


def _render_mermaid_file(input_path: str, output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    command = [
        "npx",
        "-y",
        "@mermaid-js/mermaid-cli",
        "-i",
        input_path,
        "-o",
        output_path,
    ]
    if output_path.lower().endswith(".pdf"):
        command.append("-f")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Failed to render Mermaid diagram to {output_path}: {details}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print module order and state-dict key order for a configured model, and optionally emit a Mermaid dataflow diagram."
    )
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("-d", "--dataset_type", type=str, default="default")
    parser.add_argument("--include_buffers", action="store_true", help="include buffers such as running_mean/running_var/num_batches_tracked")
    parser.add_argument(
        "--input_source",
        choices=["auto", "data", "synthetic"],
        default="auto",
        help="batch source for runtime tracing and shape annotation",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--force_runtime_trace", action="store_true", help="skip torch.fx and use a real runtime trace")
    parser.add_argument("--diagram_output", type=str, default=None, help="optionally write the Mermaid flowchart text to this path")
    parser.add_argument("--png_output", nargs="?", const="", default=None, help="render the Mermaid diagram to PNG; optionally provide an output path")
    parser.add_argument("--pdf_output", nargs="?", const="", default="", help="render the Mermaid diagram to PDF; optionally provide an output path")
    parser.add_argument("--print_diagram", action="store_true", help="also print the Mermaid diagram to stdout")

    args = parser.parse_args()

    ml_setup = get_ml_setup_from_config(
        args.model_type,
        args.dataset_type,
    )
    model = ml_setup.model.eval()

    module_records: list[ModuleRecord]
    diagram_text: str
    trace_source: str

    runtime_batch = None
    model_args = None

    if not args.force_runtime_trace:
        try:
            graph_module = symbolic_trace(model)
            try:
                runtime_batch = _get_runtime_batch(ml_setup, args.input_source, args.split)
                model_args = _model_args_from_batch(ml_setup, runtime_batch)
                _maybe_shape_propagate(graph_module, model_args)
            except Exception:
                runtime_batch = None
                model_args = None
            module_records = _call_module_order_from_fx(graph_module)
            diagram_text = _build_compact_fx_mermaid(graph_module)
            trace_source = "torch.fx"
        except Exception:
            graph_module = None  # type: ignore[assignment]
            module_records = []
            diagram_text = ""
            trace_source = ""
    else:
        graph_module = None  # type: ignore[assignment]
        module_records = []
        diagram_text = ""
        trace_source = ""

    if not module_records:
        runtime_batch = _get_runtime_batch(ml_setup, args.input_source, args.split)
        module_records = _record_runtime_module_order(ml_setup, runtime_batch)
        diagram_text = _build_compact_linear_mermaid(module_records)
        trace_source = "runtime hooks"

    state_order = _state_order_from_module_order(model, module_records, args.include_buffers)
    phase_prefixes = _recommended_phase_prefixes_from_state_order(state_order)

    report_lines = [
        f"model_type={ml_setup.model_type.name}",
        f"dataset_type={ml_setup.dataset_type.name}",
        f"application_type={ml_setup.application_type.name}",
        f"trace_source={trace_source}",
        "",
        _format_module_order(module_records),
        "",
        _format_state_order(state_order, args.include_buffers),
    ]
    report_text = "\n".join(report_lines)

    print(report_text)
    if args.print_diagram:
        print("")
        print("Mermaid diagram:")
        print(diagram_text)

    diagram_output_path = os.path.abspath(args.diagram_output) if args.diagram_output is not None else None
    png_output_path = _resolve_optional_output_path(
        args.png_output,
        _default_render_output_path(ml_setup, "png"),
    )
    pdf_output_path = _resolve_optional_output_path(
        args.pdf_output,
        _default_render_output_path(ml_setup, "pdf"),
    )
    assert pdf_output_path is not None
    txt_output_path = _txt_output_path_from_pdf_path(pdf_output_path)
    py_output_path = _py_output_path_from_pdf_path(pdf_output_path)

    _write_text_if_requested(txt_output_path, report_text)
    _write_text_if_requested(py_output_path, _format_phase_prefixes_python(phase_prefixes))

    render_input_path: Optional[str] = None
    temporary_diagram_path: Optional[str] = None
    try:
        if diagram_output_path is not None:
            _write_text_if_requested(diagram_output_path, diagram_text)
            render_input_path = diagram_output_path
        elif png_output_path is not None or pdf_output_path is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False, encoding="utf-8") as temp_file:
                temp_file.write(diagram_text)
                if not diagram_text.endswith("\n"):
                    temp_file.write("\n")
                temporary_diagram_path = temp_file.name
            render_input_path = temporary_diagram_path

        if png_output_path is not None:
            assert render_input_path is not None
            _render_mermaid_file(render_input_path, png_output_path)
        if pdf_output_path is not None:
            assert render_input_path is not None
            _render_mermaid_file(render_input_path, pdf_output_path)
    finally:
        if temporary_diagram_path is not None and os.path.exists(temporary_diagram_path):
            os.remove(temporary_diagram_path)

    print("")
    if diagram_output_path is not None:
        print(f"diagram_saved_to={diagram_output_path}")
    print(f"txt_saved_to={txt_output_path}")
    print(f"py_saved_to={py_output_path}")
    if png_output_path is not None:
        print(f"png_saved_to={png_output_path}")
    if pdf_output_path is not None:
        print(f"pdf_saved_to={pdf_output_path}")


if __name__ == "__main__":
    main()
