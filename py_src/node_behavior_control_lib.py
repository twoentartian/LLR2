"""Reusable node behavior helpers for simulator configuration files."""

from __future__ import annotations

from py_src.simulation_runtime_parameters import RuntimeParameters


def global_broadcast(runtime_parameters: RuntimeParameters, src_node_name: int, mpi_world=None) -> None:
    """Copy one node's model state to every other node.

    LLR2's simulator is local, but the MPI branch is retained for compatibility
    with DFL_torch configuration files used by an MPI runner.
    """
    if runtime_parameters.mpi_enabled:
        from mpi4py import MPI
        from mpi4py.util import pkl5

        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        assert mpi_world is not None, "mpi_world is None in an MPI environment"

        mpi_comm.barrier()
        has_src_node = src_node_name in runtime_parameters.node_container
        all_have_src_node = mpi_comm.gather(has_src_node, root=0)

        rank_with_src_node = None
        if mpi_rank == 0:
            if not any(all_have_src_node):
                raise AssertionError(f"node {src_node_name} does not exist in the MPI network")
            rank_with_src_node = all_have_src_node.index(True)
        rank_with_src_node = mpi_comm.bcast(rank_with_src_node, root=0)

        src_model_stat = None
        if mpi_rank == rank_with_src_node:
            src_model_stat = runtime_parameters.node_container[src_node_name].get_model_stat()
        src_model_stat = pkl5.Intracomm(mpi_comm).bcast(src_model_stat, root=rank_with_src_node)
        mpi_comm.barrier()
    else:
        assert src_node_name in runtime_parameters.node_container, (
            f"node {src_node_name} does not exist in the network: "
            f"{runtime_parameters.node_container.keys()}"
        )
        src_model_stat = runtime_parameters.node_container[src_node_name].get_model_stat()

    for node_name, node_target in runtime_parameters.node_container.items():
        if src_node_name != node_name:
            node_target.set_model_stat(src_model_stat)
