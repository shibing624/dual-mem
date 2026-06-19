from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import KuzuGraphStore


def test_shared_resource_reuse(tmp_storage):
    f = ComponentFactory(settings=Settings(mode="system1", storage_dir=tmp_storage))
    assert f.embed is f.embed
    assert f.vector is f.vector
    assert f.cache is f.cache
    assert f.history is f.history


def test_graph_none_when_disabled(tmp_storage):
    f = ComponentFactory(settings=Settings(mode="system1", storage_dir=tmp_storage))
    assert f.graph is None


def test_graph_present_in_dual(tmp_storage):
    f = ComponentFactory(settings=Settings(mode="dual", storage_dir=tmp_storage))
    assert isinstance(f.graph, KuzuGraphStore)
    assert f.graph is f.graph


def test_llm_present_in_system1(tmp_storage):
    f = ComponentFactory(settings=Settings(mode="system1", storage_dir=tmp_storage))
    assert f.llm is not None
    assert f.llm is f.llm


def test_llm_present_in_dual(tmp_storage):
    f = ComponentFactory(settings=Settings(mode="dual", storage_dir=tmp_storage))
    assert f.llm is not None
    assert f.llm is f.llm
