# Model Weight Viewer

This local web tool opens LLR2 `.model.pt` checkpoints and displays every
stored tensor without constructing the original model. It therefore works with
any model and dataset metadata present in a checkpoint. Plain PyTorch state
dictionaries and checkpoints using `model_state_dict` are also accepted.

Start the server from the repository root:

```bash
./result_processing_tool/model_weight_viewer/run_model_weight_viewer.sh
```

The launcher prints the URL and uses port `43817` by default. Open
<http://127.0.0.1:43817> and drop a checkpoint onto the page. Pass a custom
port to the launcher, or use `--host` and `--port` with the Python file to
change the listen address or port.

The viewer shows tensor names, dimensions, dtype, element count, storage size,
optional statistics, and paginated raw values with both flat indexes and
multidimensional coordinates. Checkpoints are loaded on CPU with
`torch.load(..., weights_only=True)`. Uploaded files are deleted immediately
after loading, and loaded tensors remain only in the local server process.
