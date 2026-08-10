"""The per-build schema memo: what it costs to build a pipeline, and the pairing it
relies on.

Every verb resolves its input's schema, and a plan is built as nested resolvers, so
without memoization an N-verb chain re-walks the whole subtree beneath it at every
level -- O(N^2) inference (#207). ``builders.plan._remember_input_schemas`` records
each embedded input relation's schema as the plan is assembled, and
``infer_plan_schema`` builds its ``rel_anchor`` index only when an id-based outer
reference asks for one. Both are invisible in the emitted plan, so they need tests
that observe cost, plus tests that the recorded schemas land on the right relations.
"""

import collections

import pytest
import substrait.algebra_pb2 as stalg
import substrait.plan_pb2 as stp
import substrait.type_pb2 as stt

import substrait.type_inference as type_inference
from substrait.builders.extended_expression import column, literal
from substrait.builders.plan import (
    cross,
    hash_join,
    join,
    lateral_join,
    merge_join,
    nested_loop_join,
    project,
    read_named_table,
    reference,
    with_execution_behavior,
    write_named_table,
)
from substrait.builders.type import boolean, i64, string
from substrait.extension_registry import ExtensionRegistry
from substrait.type_inference import infer_plan_schema, schema_memo

registry = ExtensionRegistry(load_default_extensions=False)

named_struct = stt.NamedStruct(
    names=["k", "v"],
    struct=stt.Type.Struct(
        types=[i64(nullable=False), i64(nullable=False)],
        nullability=stt.Type.NULLABILITY_REQUIRED,
    ),
)

# Same arity as `named_struct` but a different second column type, so pairing the
# two sides of a join the wrong way round shows up in the types alone.
right_named_struct = stt.NamedStruct(
    names=["rk", "rv"],
    struct=stt.Type.Struct(
        types=[i64(nullable=False), string()],
        nullability=stt.Type.NULLABILITY_REQUIRED,
    ),
)


@pytest.fixture
def counts(monkeypatch):
    """Counts of the two whole-subtree walks a build must not repeat per level.

    Both are patched on ``substrait.type_inference`` because that is where the
    recursion and the anchor index resolve them from.
    """
    counted: collections.Counter = collections.Counter()

    infer_rel_schema = type_inference.infer_rel_schema
    iter_plan_rels = type_inference.iter_plan_rels

    def counting_infer_rel_schema(rel, **kwargs):
        counted["infer_rel_schema"] += 1
        return infer_rel_schema(rel, **kwargs)

    def counting_iter_plan_rels(plan):
        counted["iter_plan_rels"] += 1
        return iter_plan_rels(plan)

    monkeypatch.setattr(type_inference, "infer_rel_schema", counting_infer_rel_schema)
    monkeypatch.setattr(type_inference, "iter_plan_rels", counting_iter_plan_rels)
    return counted


def _project_chain(length: int):
    plan = read_named_table("t", named_struct)
    for _ in range(length):
        plan = project(plan, expressions=[column("v")])
    return plan


# Deliberately well above the 4N-4 this currently does and well below the ~N^2/2 it
# did before, so the test tracks the complexity class rather than the exact count.
_CALLS_PER_VERB = 6


@pytest.mark.parametrize("length", [4, 8, 16, 32])
def test_building_a_chain_infers_each_level_a_bounded_number_of_times(counts, length):
    built = _project_chain(length)(registry)

    assert len(built.relations[-1].root.names) == 2 + length
    assert counts["infer_rel_schema"] <= _CALLS_PER_VERB * length


def test_chain_inference_grows_linearly_not_quadratically(counts):
    _project_chain(8)(registry)
    short = counts["infer_rel_schema"]
    counts.clear()
    _project_chain(32)(registry)
    long = counts["infer_rel_schema"]

    # Four times the verbs, so linear allows roughly four times the inferences (with
    # headroom); quadratic would be sixteen.
    assert long <= 6 * short


def test_building_a_chain_never_indexes_rel_anchors(counts):
    # Indexing walks every relation and expression in the plan. Nothing here carries
    # an id-based OuterReference, so nothing should ask for the index.
    _project_chain(8)(registry)

    assert counts["iter_plan_rels"] == 0


def test_memo_does_not_outlive_the_build():
    # The memo keys on object identity and holds its keys alive, so leaking it past
    # the build would both pin memory and answer for relations of a later one.
    assert schema_memo.get() is None
    built = _project_chain(2)(registry)
    assert schema_memo.get() is None

    # Inference of the finished plan is unmemoized and still correct.
    assert list(infer_plan_schema(built, registry=registry).names) == [
        "k",
        "v",
        "v",
        "v",
    ]


def _left():
    return read_named_table("left", named_struct)


def _right():
    return read_named_table("right", right_named_struct)


def _true():
    return literal(True, boolean())


# Every builder that embeds more than one input relation, since those are the ones
# whose recorded schemas could be paired with the wrong side.
TWO_INPUT_BUILDERS = {
    "join": lambda: join(_left(), _right(), _true(), stalg.JoinRel.JOIN_TYPE_INNER),
    "cross": lambda: cross(_left(), _right()),
    "nested_loop_join": lambda: nested_loop_join(
        _left(), _right(), _true(), stalg.NestedLoopJoinRel.JOIN_TYPE_INNER
    ),
    "hash_join": lambda: hash_join(
        _left(), _right(), ["k"], ["rk"], stalg.HashJoinRel.JOIN_TYPE_INNER
    ),
    "merge_join": lambda: merge_join(
        _left(), _right(), ["k"], ["rk"], stalg.MergeJoinRel.JOIN_TYPE_INNER
    ),
    "lateral_join": lambda: lateral_join(
        _left(), lambda handle: _right(), stalg.JoinRel.JOIN_TYPE_INNER
    ),
}


@pytest.mark.parametrize("builder", TWO_INPUT_BUILDERS.values(), ids=TWO_INPUT_BUILDERS)
def test_two_input_builders_record_each_side_against_its_own_relation(builder):
    # write_named_table emits the schema it inferred for its input, so it reports what
    # the level above the join sees: left columns then right columns. Swapping the
    # recorded schemas keeps the arity and the names but reorders the types.
    written = write_named_table("out", builder())(registry)

    table_schema = written.relations[-1].root.input.write.table_schema
    assert list(table_schema.names) == ["k", "v", "rk", "rv"]
    assert list(table_schema.struct.types) == [
        i64(nullable=False),
        i64(nullable=False),
        i64(nullable=False),
        string(),
    ]


def test_reference_records_the_promoted_subtree_and_the_reference():
    # A ReferenceRel's schema is its subtree's, and `reference` records both so a
    # downstream verb resolves it by lookup rather than by walking the subtree.
    written = write_named_table("out", reference(_left()))(registry)

    table_schema = written.relations[-1].root.input.write.table_schema
    assert table_schema == named_struct


def test_with_execution_behavior_records_the_copied_root():
    # This builder copies its input plan wholesale rather than assembling a fresh
    # relation, so the copy needs its own record.
    written = write_named_table(
        "out",
        with_execution_behavior(
            _left(), stp.ExecutionBehavior.VARIABLE_EVALUATION_MODE_PER_RECORD
        ),
    )(registry)

    table_schema = written.relations[-1].root.input.write.table_schema
    assert table_schema == named_struct
