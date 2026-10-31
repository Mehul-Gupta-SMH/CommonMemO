"""The LangGraph adapter (#111), against the installed BaseStore and LangGraph's own InMemoryStore."""

import asyncio

import pytest

pytest.importorskip("langgraph.store.base")

from langgraph.store.base import BaseStore  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from commonmemo import Memory  # noqa: E402
from commonmemo.adapters.langgraph import CommonMemoStore  # noqa: E402

DOCS = [
    (("users", "ada", "prefs"), "theme", {"theme": "dark", "size": 12, "tags": ["a", "b"], "nested": {"x": 1}}),
    (("users", "ada", "prefs"), "lang", {"lang": "pt", "size": 20}),
    (("users", "ada", "notes"), "n1", {"text": "Maya is allergic to peanuts", "size": 3}),
    (("users", "grace", "prefs"), "theme", {"theme": "light", "size": 14}),
    (("docs",), "d1", {"text": "oat bars are peanut free"}),
]


@pytest.fixture
def pair(tmp_path):
    return CommonMemoStore(Memory(f"sqlite:///{tmp_path}/m.db", project="shop")), InMemoryStore()


def fill(*stores):
    for store in stores:
        for namespace, key, value in DOCS:
            store.put(namespace, key, value)


def view(items):
    return [(i.namespace, i.key, i.value) for i in items]


def test_is_a_base_store(pair):
    assert isinstance(pair[0], BaseStore)


def test_get_and_put_behave_like_langgraphs_own_store(pair):
    ours, theirs = pair
    fill(ours, theirs)
    for namespace, key, value in DOCS:
        assert ours.get(namespace, key).value == theirs.get(namespace, key).value == value
    assert ours.get(("users", "ada", "prefs"), "missing") is None
    assert ours.get(("nobody",), "theme") is None


def test_an_overwrite_keeps_created_at_and_moves_updated_at(pair):
    ours, _ = pair
    ours.put(("a",), "k", {"v": 1})
    first = ours.get(("a",), "k")
    ours.put(("a",), "k", {"v": 2})
    second = ours.get(("a",), "k")
    assert second.value == {"v": 2} and second.created_at == first.created_at and second.updated_at >= first.updated_at


def test_delete_and_put_none(pair):
    ours, theirs = pair
    fill(ours, theirs)
    for store in pair:
        store.delete(("users", "ada", "prefs"), "lang")
    assert ours.get(("users", "ada", "prefs"), "lang") is None
    assert sorted(view(ours.search(("users",), limit=50))) == sorted(view(theirs.search(("users",), limit=50)))
    for store in pair:
        store.delete(("docs",), "d1")
    # An emptied namespace is gone (as in LangGraph's Postgres store; InMemoryStore keeps listing it).
    assert ours.list_namespaces() == [n for n in theirs.list_namespaces() if n != ("docs",)]


@pytest.mark.parametrize("kwargs", [
    {"namespace_prefix": ("users",)},
    {"namespace_prefix": ("users", "ada")},
    {"namespace_prefix": ("users",), "filter": {"theme": "dark"}},
    {"namespace_prefix": ("users",), "filter": {"size": {"$gt": 12}}},
    {"namespace_prefix": ("users",), "filter": {"size": {"$gte": 12, "$lt": 20}}},
    {"namespace_prefix": ("users",), "filter": {"theme": {"$ne": "dark"}}},
    {"namespace_prefix": ("users",), "filter": {"nested": {"x": 1}}},
    {"namespace_prefix": ("users",), "filter": {"tags": ["a", "b"]}},
    {"namespace_prefix": ("users",), "limit": 2},
    {"namespace_prefix": ("users",), "limit": 2, "offset": 2},
    {"namespace_prefix": ()},
])
def test_search_filters_like_langgraphs_own_store(pair, kwargs):
    ours, theirs = pair
    fill(ours, theirs)
    prefix = kwargs.pop("namespace_prefix")
    assert view(ours.search(prefix, **kwargs)) == view(theirs.search(prefix, **kwargs))


@pytest.mark.parametrize("kwargs", [
    {},
    {"prefix": ("users",)},
    {"prefix": ("users", "ada")},
    {"suffix": ("prefs",)},
    {"prefix": ("users",), "suffix": ("prefs",)},
    {"prefix": ("users", "*"), "max_depth": 2},
    {"max_depth": 1},
    {"limit": 2, "offset": 1},
])
def test_list_namespaces_like_langgraphs_own_store(pair, kwargs):
    ours, theirs = pair
    fill(ours, theirs)
    assert ours.list_namespaces(**kwargs) == theirs.list_namespaces(**kwargs)


def test_search_by_query_ranks_by_the_text(pair):
    ours, _ = pair
    fill(ours)
    found = ours.search(("users", "ada", "notes"), query="peanut allergy Maya")
    assert found[0].key == "n1" and found[0].score is not None
    everywhere = ours.search((), query="peanut free oat bars", limit=1)
    assert everywhere[0].key == "d1"


def test_async_calls(pair):
    ours, theirs = pair

    async def go():
        for namespace, key, value in DOCS:
            await ours.aput(namespace, key, value)
        assert (await ours.aget(("docs",), "d1")).value == DOCS[-1][2]
        assert len(await ours.asearch(("users",), limit=50)) == 4
        assert await ours.alist_namespaces(prefix=("docs",)) == [("docs",)]
        await ours.adelete(("docs",), "d1")
        assert await ours.aget(("docs",), "d1") is None
    asyncio.run(go())


def test_a_batch_sees_its_own_earlier_puts(pair):
    from langgraph.store.base import GetOp, PutOp
    ours, _ = pair
    got = ours.batch([PutOp(("a",), "k", {"v": 1}, None), GetOp(("a",), "k"), PutOp(("a",), "k", None, None),
                      GetOp(("a",), "k")])
    assert got[0] is None and got[1].value == {"v": 1} and got[2] is None and got[3] is None


def test_it_survives_a_restart_and_two_stores_share_a_log(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    one = CommonMemoStore(Memory(url, project="shop"))
    one.put(("a",), "k", {"v": 1})
    two = CommonMemoStore(Memory(url, project="shop"))
    assert two.get(("a",), "k").value == {"v": 1}
    two.put(("a",), "k2", {"v": 2})
    assert one.get(("a",), "k2").value == {"v": 2}  # written by another process, read on the next call


def test_stores_with_different_ids_are_separate(tmp_path):
    memory = Memory(f"sqlite:///{tmp_path}/m.db", project="shop")
    one, two = CommonMemoStore(memory, session_id="one"), CommonMemoStore(memory, session_id="two")
    one.put(("a",), "k", {"v": 1})
    assert two.get(("a",), "k") is None and two.list_namespaces() == []


def test_writes_append_and_delete_nothing(pair):
    ours, _ = pair
    ours.put(("a",), "k", {"v": 1})
    before = len(ours.memory.store.events())
    ours.put(("a",), "k", {"v": 2})
    ours.delete(("a",), "k")
    assert len(ours.memory.store.events()) == before + 2


def test_a_compiled_graph_uses_it_across_runs(tmp_path):
    graph_mod = pytest.importorskip("langgraph.graph")
    from typing import TypedDict

    class State(TypedDict):
        user: str
        reply: str

    def remember(state: State, *, store: BaseStore):
        found = store.get(("memories", state["user"]), "city")
        if found is None:
            store.put(("memories", state["user"]), "city", {"text": "lives in Lisbon"})
        return {"reply": found.value["text"] if found else "nothing yet"}

    def build(url):
        builder = graph_mod.StateGraph(State)
        builder.add_node("remember", remember)
        builder.add_edge(graph_mod.START, "remember")
        builder.add_edge("remember", graph_mod.END)
        return builder.compile(store=CommonMemoStore(Memory(url, project="shop")))

    url = f"sqlite:///{tmp_path}/m.db"
    assert build(url).invoke({"user": "ada"})["reply"] == "nothing yet"
    assert build(url).invoke({"user": "ada"})["reply"] == "lives in Lisbon"  # a new graph and memory, same Log
    assert build(url).invoke({"user": "grace"})["reply"] == "nothing yet"
