from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.bench.livecodebench_prepare import RELEASE_FILES
from app.codegen import catalog
from app.codegen.catalog import list_codegen_task_summaries, load_codegen_tasks
from tests.helpers import disabled_registry_task_ids, enabled_registry_entries, enabled_registry_task_ids


class CodegenCatalogTest(unittest.TestCase):
    def test_active_registry_uses_one_main_benchmark_lane(self) -> None:
        tasks = load_codegen_tasks()
        comparable_tasks = [task for task in tasks if task["benchmark_tier"] == "comparable"]
        main_tasks = [task for task in tasks if task["included_in_main_comparison"]]
        off_main_tasks = [task for task in tasks if not task["included_in_main_comparison"]]
        expected_task_ids = enabled_registry_task_ids()
        expected_tracks = {str(entry["path"]).split("/", 1)[0] for entry in enabled_registry_entries()}

        self.assertTrue(comparable_tasks)
        self.assertEqual({task["id"] for task in tasks}, set(expected_task_ids))
        self.assertEqual(
            {task["id"] for task in off_main_tasks},
            {"alfworld"},
        )
        self.assertEqual(
            {task["track"] for task in comparable_tasks},
            expected_tracks,
        )
        self.assertTrue(all("runtime_backend" not in task for task in tasks))
        self.assertTrue(all("optimization_scope" not in task for task in tasks))
        self.assertTrue(all(task["included_in_main_comparison"] for task in main_tasks))
        self.assertEqual([task["id"] for task in main_tasks[:5]], ["olymmath", "math-500", "aime", "planbench-t1", "planbench-t2"])

    def test_main_comparison_filter_returns_all_active_benchmark_tasks(self) -> None:
        all_tasks = load_codegen_tasks()
        main_tasks = load_codegen_tasks(included_in_main_comparison=True)
        expected_task_ids = enabled_registry_task_ids()
        self.assertTrue(main_tasks)
        self.assertEqual(len(main_tasks), len(all_tasks) - 1)
        self.assertEqual({task["id"] for task in main_tasks}, set(expected_task_ids) - {"alfworld"})
        self.assertTrue(all(task["included_in_main_comparison"] for task in main_tasks))
        self.assertTrue(all(task["benchmark_tier"] == "comparable" for task in main_tasks))
        self.assertEqual(
            {task["id"] for task in all_tasks if not task["included_in_main_comparison"]},
            {"alfworld"},
        )

    def test_task_summaries_include_benchmark_metadata(self) -> None:
        summaries = list_codegen_task_summaries()
        olymmath = next(task for task in summaries if task["id"] == "olymmath")
        math_500 = next(task for task in summaries if task["id"] == "math-500")
        aime = next(task for task in summaries if task["id"] == "aime")
        summaries_by_id = {task["id"]: task for task in summaries}
        planbench_t1 = summaries_by_id.get("planbench-t1")
        planbench_t2 = summaries_by_id.get("planbench-t2")
        planbench_t3 = summaries_by_id.get("planbench-t3")
        arc_challenge = summaries_by_id.get("arc-challenge")
        bbh = summaries_by_id.get("bbh")
        mmlu_pro = summaries_by_id.get("mmlu-pro")
        longbench = summaries_by_id.get("longbench-v2")
        incharacter = summaries_by_id.get("incharacter")
        characterbench = summaries_by_id.get("characterbench")
        timechara = summaries_by_id.get("timechara")
        personamem = summaries_by_id.get("personamem-32k")
        socialbench = summaries_by_id.get("socialbench")
        xstest = summaries_by_id.get("xstest-refusal-calibration")
        harmbench = summaries_by_id.get("harmbench-text-harmful")
        jailbreakbench = summaries_by_id.get("jailbreakbench-harmful")
        or_bench_hard = summaries_by_id.get("or-bench-hard-1k")
        or_bench_toxic = summaries_by_id.get("or-bench-toxic")
        hallulens_precise = summaries_by_id.get("hallulens-precisewikiqa")
        hallulens_mixed = summaries_by_id.get("hallulens-mixedentities")
        hallulens_longwiki = summaries_by_id.get("hallulens-longwiki")
        longsafety = summaries_by_id.get("longsafety")
        sciq = summaries_by_id.get("sciq")
        qasc = summaries_by_id.get("qasc")
        scienceqa = summaries_by_id.get("scienceqa")
        openbookqa = summaries_by_id.get("openbookqa")
        gpqa_diamond = summaries_by_id.get("gpqa-diamond")
        alfworld = summaries_by_id.get("alfworld")
        acpbench = summaries_by_id.get("acpbench")
        livecodebench = summaries_by_id.get("livecodebench")
        alpsbench_retrieval = summaries_by_id.get("alpsbench-retrieval")
        alpsbench_utilization = summaries_by_id.get("alpsbench-utilization")
        co_bench = summaries_by_id.get("co-bench")
        summary_ids = {task["id"] for task in summaries}
        self.assertEqual({task["id"] for task in summaries}, set(enabled_registry_task_ids()))
        self.assertTrue(disabled_registry_task_ids().isdisjoint(summary_ids))

        self.assertTrue(olymmath["local_dataset_only"])
        self.assertEqual(olymmath["dataset_size"], 100)
        self.assertEqual(olymmath["split"], "en-hard:test")
        self.assertEqual(math_500["track"], "math_verified")
        self.assertEqual(math_500["split"], "test")
        self.assertEqual(aime["dataset_size"], 90)
        self.assertEqual(aime["split"], "2024:train + 2025:(I+II):test + 2026:test")
        self.assertEqual(
            [option["value"] for option in aime["runtime_split_selector"]["options"]],
            ["all", "2024", "2025", "2026"],
        )
        self.assertEqual(
            [option["item_count"] for option in aime["runtime_split_selector"]["options"]],
            [90, 30, 30, 30],
        )
        self.assertNotIn("aime-2024", summaries_by_id)
        self.assertNotIn("aime-2025", summaries_by_id)
        self.assertNotIn("aime-2026", summaries_by_id)
        self.assertIsNotNone(planbench_t1)
        self.assertIsNotNone(planbench_t2)
        self.assertIsNotNone(planbench_t3)
        self.assertNotIn("planbench", summaries_by_id)
        self.assertTrue(planbench_t1["local_dataset_only"])
        self.assertEqual(planbench_t1["dataset_size"], 2270)
        self.assertEqual(planbench_t1["track"], "reasoning_verified")
        self.assertTrue(planbench_t1["included_in_main_comparison"])
        self.assertEqual(planbench_t1["split"], "task_1_plan_generation:train")
        self.assertEqual(planbench_t1["task_mode"], "answer")
        self.assertEqual(planbench_t1["interaction_mode"], "single_turn")
        self.assertEqual(planbench_t1["selection_spec"]["profile"], "objective_only")
        self.assertEqual(
            [option["value"] for option in planbench_t1["runtime_split_selector"]["options"]],
            [
                "all",
                "blocksworld",
                "blocksworld_3",
                "depots",
                "logistics",
                "mystery_blocksworld",
                "mystery_blocksworld_3",
                "obfuscated_deceptive_logistics",
            ],
        )
        self.assertEqual(planbench_t2["dataset_size"], 1692)
        self.assertEqual(planbench_t2["split"], "task_2_plan_optimality:train")
        self.assertEqual(planbench_t2["answer_metric"], "optimal_plan_rate")
        self.assertEqual(
            [option["value"] for option in planbench_t2["runtime_split_selector"]["options"]],
            [
                "all",
                "blocksworld",
                "blocksworld_3",
                "logistics",
                "mystery_blocksworld",
                "mystery_blocksworld_3",
                "obfuscated_deceptive_logistics",
            ],
        )
        self.assertEqual(planbench_t3["dataset_size"], 1584)
        self.assertEqual(planbench_t3["split"], "task_3_plan_verification:train")
        self.assertEqual(planbench_t3["answer_metric"], "verification_accuracy")
        self.assertEqual(
            [option["value"] for option in planbench_t3["runtime_split_selector"]["options"]],
            [
                "all",
                "blocksworld",
                "blocksworld_3",
                "logistics",
                "mystery_blocksworld",
                "mystery_blocksworld_3",
            ],
        )
        self.assertEqual(arc_challenge["dataset_size"], 299)
        self.assertEqual(arc_challenge["track"], "reasoning_verified")
        self.assertEqual(arc_challenge["split"], "validation:ARC-Challenge")
        self.assertEqual(bbh["dataset_size"], 6511)
        self.assertEqual(bbh["track"], "reasoning_verified")
        self.assertEqual(bbh["split"], "train:all_configs")
        self.assertTrue(bbh["included_in_main_comparison"])
        self.assertEqual(bbh["runtime_split_selector"]["default_value"], "all")
        self.assertEqual(bbh["runtime_split_selector"]["options"][0]["item_count"], 6511)
        self.assertIn("boolean_expressions", {option["value"] for option in bbh["runtime_split_selector"]["options"]})
        self.assertIn("tracking_shuffled_objects_seven_objects", {option["value"] for option in bbh["runtime_split_selector"]["options"]})
        self.assertEqual(mmlu_pro["dataset_size"], 12032)
        self.assertEqual(mmlu_pro["track"], "reasoning_verified")
        self.assertEqual(mmlu_pro["split"], "default:test")
        self.assertEqual(mmlu_pro["interaction_mode"], "single_turn")
        self.assertTrue(mmlu_pro["included_in_main_comparison"])
        self.assertEqual(mmlu_pro["runtime_split_selector"]["default_value"], "all")
        self.assertEqual(mmlu_pro["runtime_split_selector"]["options"][0]["item_count"], 12032)
        self.assertEqual(
            [option["value"] for option in mmlu_pro["runtime_split_selector"]["options"][:5]],
            ["all", "biology", "business", "chemistry", "computer-science"],
        )
        self.assertEqual(longbench["dataset_size"], 503)
        self.assertEqual(longbench["track"], "longcontext_verified")
        self.assertEqual(longbench["split"], "train")
        self.assertTrue(longbench["included_in_main_comparison"])
        self.assertEqual(
            [option["value"] for option in longbench["runtime_split_selector"]["options"]],
            [
                "all",
                "single-document-qa",
                "multi-document-qa",
                "long-in-context-learning",
                "code-repository-understanding",
                "long-dialogue-history-understanding",
                "long-structured-data-understanding",
            ],
        )
        self.assertEqual(
            [option["item_count"] for option in longbench["runtime_split_selector"]["options"]],
            [503, 175, 125, 81, 50, 39, 33],
        )
        self.assertEqual(incharacter["track"], "personalization_verified")
        self.assertEqual(incharacter["dataset_size"], 448)
        self.assertEqual(incharacter["interaction_mode"], "single_turn")
        self.assertEqual(incharacter["task_shape"], "dialogue_judgement")
        self.assertTrue(incharacter["supports_eval_model"])
        self.assertTrue(incharacter["requires_eval_model"])
        self.assertTrue(incharacter["included_in_main_comparison"])
        self.assertEqual(characterbench["track"], "personalization_verified")
        self.assertEqual(characterbench["dataset_size"], 3250)
        self.assertEqual(characterbench["interaction_mode"], "single_turn")
        self.assertEqual(characterbench["task_shape"], "dialogue_judgement")
        self.assertTrue(characterbench["supports_eval_model"])
        self.assertTrue(characterbench["requires_eval_model"])
        self.assertTrue(characterbench["included_in_main_comparison"])
        if timechara is not None:
            self.assertEqual(timechara["dataset_size"], 10895)
            self.assertEqual(timechara["interaction_mode"], "single_turn")
            self.assertEqual(timechara["task_shape"], "dialogue_judgement")
            self.assertTrue(timechara["supports_eval_model"])
            self.assertTrue(timechara["requires_eval_model"])
            self.assertTrue(timechara["included_in_main_comparison"])
        self.assertEqual(personamem["track"], "personalization_verified")
        self.assertEqual(personamem["dataset_size"], 589)
        self.assertEqual(personamem["split"], "benchmark:32k")
        self.assertEqual(personamem["research_line"], "personalization")
        self.assertEqual(personamem["personalization_category"], "user_persona")
        self.assertEqual(personamem["personalization_focus"], "preference_following")
        self.assertFalse(personamem["supports_eval_model"])
        self.assertTrue(personamem["included_in_main_comparison"])
        self.assertEqual(socialbench["track"], "personalization_verified")
        self.assertEqual(socialbench["dataset_size"], 7702)
        self.assertEqual(socialbench["split"], "official:all")
        self.assertEqual(socialbench["research_line"], "personalization")
        self.assertEqual(socialbench["personalization_category"], "role_play")
        self.assertEqual(socialbench["personalization_focus"], "sociality")
        self.assertFalse(socialbench["supports_eval_model"])
        self.assertTrue(socialbench["included_in_main_comparison"])
        self.assertEqual(xstest["track"], "safety_verified")
        self.assertEqual(xstest["title"], "XSTest Refusal Calibration")
        self.assertEqual(xstest["dataset_size"], 450)
        self.assertEqual(xstest["split"], "hf:train")
        self.assertEqual(xstest["research_line"], "safety")
        self.assertEqual(xstest["safety_category"], "over_refusal")
        self.assertEqual(xstest["safety_focus"], "over_refusal")
        self.assertTrue(xstest["included_in_main_comparison"])
        if harmbench is not None:
            self.assertEqual(harmbench["track"], "safety_verified")
            self.assertEqual(harmbench["dataset_size"], 240)
            self.assertEqual(harmbench["interaction_mode"], "single_turn")
            self.assertEqual(harmbench["safety_category"], "jailbreak_attack")
            self.assertEqual(harmbench["safety_focus"], "jailbreak_attack")
            self.assertTrue(harmbench["included_in_main_comparison"])
            self.assertTrue(harmbench["supports_max_items"])
            self.assertEqual(harmbench["default_max_items"], 240)
            self.assertFalse(harmbench["supports_max_episodes"])
        if jailbreakbench is not None:
            self.assertEqual(jailbreakbench["track"], "safety_verified")
            self.assertEqual(jailbreakbench["dataset_size"], 100)
            self.assertEqual(jailbreakbench["interaction_mode"], "single_turn")
            self.assertEqual(jailbreakbench["safety_category"], "jailbreak_attack")
            self.assertEqual(jailbreakbench["safety_focus"], "jailbreak_attack")
            self.assertTrue(jailbreakbench["included_in_main_comparison"])
            self.assertTrue(jailbreakbench["supports_max_items"])
            self.assertEqual(jailbreakbench["default_max_items"], 100)
            self.assertFalse(jailbreakbench["supports_max_episodes"])
        if or_bench_hard is not None:
            self.assertEqual(or_bench_hard["track"], "safety_verified")
            self.assertEqual(or_bench_hard["dataset_size"], 1319)
            self.assertEqual(or_bench_hard["interaction_mode"], "single_turn")
            self.assertEqual(or_bench_hard["safety_category"], "over_refusal")
            self.assertEqual(or_bench_hard["safety_focus"], "over_refusal")
            self.assertTrue(or_bench_hard["included_in_main_comparison"])
            self.assertTrue(or_bench_hard["supports_max_items"])
            self.assertEqual(or_bench_hard["default_max_items"], 1319)
        if or_bench_toxic is not None:
            self.assertEqual(or_bench_toxic["track"], "safety_verified")
            self.assertEqual(or_bench_toxic["dataset_size"], 655)
            self.assertEqual(or_bench_toxic["interaction_mode"], "single_turn")
            self.assertEqual(or_bench_toxic["safety_category"], "jailbreak_attack")
            self.assertEqual(or_bench_toxic["safety_focus"], "should_refuse")
            self.assertTrue(or_bench_toxic["included_in_main_comparison"])
            self.assertTrue(or_bench_toxic["supports_max_items"])
            self.assertEqual(or_bench_toxic["default_max_items"], 655)
        if hallulens_precise is not None:
            self.assertEqual(hallulens_precise["track"], "safety_verified")
            self.assertEqual(hallulens_precise["dataset_size"], 250)
            self.assertEqual(hallulens_precise["interaction_mode"], "single_turn")
            self.assertEqual(hallulens_precise["safety_category"], "factuality_hallucination")
            self.assertEqual(hallulens_precise["safety_focus"], "factuality_hallucination")
            self.assertTrue(hallulens_precise["supports_max_items"])
            self.assertEqual(hallulens_precise["default_max_items"], 250)
        if hallulens_mixed is not None:
            self.assertEqual(hallulens_mixed["track"], "safety_verified")
            self.assertEqual(hallulens_mixed["dataset_size"], 400)
            self.assertEqual(hallulens_mixed["interaction_mode"], "single_turn")
            self.assertEqual(hallulens_mixed["safety_category"], "factuality_hallucination")
            self.assertEqual(hallulens_mixed["safety_focus"], "factuality_hallucination")
            self.assertTrue(hallulens_mixed["supports_max_items"])
            self.assertEqual(hallulens_mixed["default_max_items"], 400)
        if hallulens_longwiki is not None:
            self.assertEqual(hallulens_longwiki["track"], "safety_verified")
            self.assertEqual(hallulens_longwiki["dataset_size"], 250)
            self.assertEqual(hallulens_longwiki["interaction_mode"], "single_turn")
            self.assertEqual(hallulens_longwiki["safety_category"], "factuality_hallucination")
            self.assertEqual(hallulens_longwiki["safety_focus"], "factuality_hallucination")
            self.assertTrue(hallulens_longwiki["supports_max_items"])
            self.assertEqual(hallulens_longwiki["default_max_items"], 250)
        if longsafety is not None:
            self.assertEqual(longsafety["track"], "safety_verified")
            self.assertEqual(longsafety["dataset_size"], 1543)
            self.assertEqual(longsafety["interaction_mode"], "single_turn")
            self.assertEqual(longsafety["safety_category"], "jailbreak_attack")
            self.assertEqual(longsafety["safety_focus"], "safety_degradation")
            self.assertTrue(longsafety["supports_max_items"])
            self.assertEqual(longsafety["default_max_items"], 1543)
        self.assertEqual(sciq["dataset_size"], 1000)
        self.assertEqual(sciq["track"], "science_verified")
        self.assertEqual(sciq["split"], "validation")
        self.assertEqual(qasc["dataset_size"], 926)
        self.assertEqual(qasc["split"], "validation")
        self.assertEqual(scienceqa["dataset_size"], 768)
        self.assertEqual(scienceqa["title"], "ScienceQA Text Bio/Chem/Phys")
        self.assertEqual(scienceqa["split"], "validation:natural-science:text-only:biology-chemistry-physics")
        self.assertEqual(
            [option["value"] for option in scienceqa["runtime_split_selector"]["options"]],
            ["all", "biology", "chemistry", "physics"],
        )
        self.assertEqual(
            [option["item_count"] for option in scienceqa["runtime_split_selector"]["options"]],
            [768, 406, 136, 226],
        )
        self.assertEqual(openbookqa["dataset_size"], 500)
        self.assertEqual(openbookqa["track"], "science_verified")
        self.assertEqual(openbookqa["split"], "validation:additional")
        self.assertEqual(gpqa_diamond["dataset_size"], 198)
        self.assertEqual(gpqa_diamond["track"], "science_verified")
        self.assertEqual(gpqa_diamond["split"], "official:diamond")
        self.assertEqual(gpqa_diamond["interaction_mode"], "single_turn")
        self.assertTrue(gpqa_diamond["included_in_main_comparison"])
        self.assertIsNotNone(alfworld)
        self.assertEqual(alfworld["track"], "agent_verified")
        self.assertEqual(alfworld["family"], "agent-benchmark")
        self.assertEqual(alfworld["dataset_size"], 140)
        self.assertEqual(alfworld["split"], "valid_seen")
        self.assertEqual(alfworld["task_mode"], "artifact")
        self.assertEqual(alfworld["interaction_mode"], "multi_turn")
        self.assertFalse(alfworld["included_in_main_comparison"])
        self.assertTrue(alfworld["supports_runtime_config"])
        self.assertTrue(alfworld["supports_max_episodes"])
        self.assertEqual(alfworld["default_max_episodes"], 140)
        self.assertFalse(alfworld["supports_max_items"])
        self.assertIsNone(alfworld["default_max_items"])
        self.assertIsNotNone(acpbench)
        self.assertEqual(acpbench["dataset_size"], 1800)
        self.assertEqual(acpbench["track"], "reasoning_verified")
        self.assertEqual(acpbench["split"], "test:bool+mcq")
        self.assertTrue(acpbench["included_in_main_comparison"])
        self.assertEqual(acpbench["task_mode"], "answer")
        self.assertEqual(acpbench["interaction_mode"], "single_turn")
        self.assertFalse(acpbench["supports_runtime_config"])
        self.assertTrue(acpbench["supports_max_items"])
        self.assertEqual(acpbench["default_max_items"], 1800)
        self.assertEqual(acpbench["runtime_split_selector"]["default_value"], "all")
        self.assertEqual(acpbench["runtime_split_selector"]["options"][0]["item_count"], 1800)
        self.assertEqual(
            {option["value"] for option in acpbench["runtime_split_selector"]["options"]},
            {
                "all",
                "bool",
                "mcq",
                "acp_app_bool",
                "acp_app_mcq",
                "acp_areach_bool",
                "acp_areach_mcq",
                "acp_just_bool",
                "acp_just_mcq",
                "acp_land_bool",
                "acp_land_mcq",
                "acp_prog_bool",
                "acp_prog_mcq",
                "acp_reach_bool",
                "acp_reach_mcq",
                "acp_val_bool",
                "acp_val_mcq",
            },
        )
        self.assertIsNotNone(livecodebench)
        self.assertEqual(livecodebench["dataset_size"], 1055)
        self.assertEqual(livecodebench["track"], "coding_verified")
        self.assertEqual(livecodebench["split"], "v1+v2+v3+v4+v5+v6:test")
        self.assertTrue(livecodebench["included_in_main_comparison"])
        self.assertEqual(livecodebench["task_mode"], "artifact")
        self.assertEqual(livecodebench["interaction_mode"], "single_turn")
        self.assertFalse(livecodebench["supports_runtime_config"])
        self.assertTrue(livecodebench["supports_max_items"])
        self.assertEqual(livecodebench["default_max_items"], 1055)
        self.assertEqual(livecodebench["runtime_split_selector"]["default_value"], "all")
        self.assertEqual(
            [option["value"] for option in livecodebench["runtime_split_selector"]["options"]],
            ["all", "v1", "v2", "v3", "v4", "v5", "v6"],
        )
        self.assertEqual(
            [option["item_count"] for option in livecodebench["runtime_split_selector"]["options"]],
            [1055, 400, 111, 101, 101, 167, 175],
        )
        self.assertIsNotNone(alpsbench_retrieval)
        self.assertEqual(
            [option["value"] for option in alpsbench_retrieval["runtime_split_selector"]["options"]],
            ["all", "d100", "d300", "d500", "d700", "d1000"],
        )
        self.assertEqual(
            [option["item_count"] for option in alpsbench_retrieval["runtime_split_selector"]["options"]],
            [2380, 476, 476, 476, 476, 476],
        )
        self.assertIsNotNone(alpsbench_utilization)
        self.assertEqual(
            [option["value"] for option in alpsbench_utilization["runtime_split_selector"]["options"]],
            ["all", "ability1", "ability2", "ability3", "ability4", "ability5"],
        )
        self.assertEqual(
            [option["item_count"] for option in alpsbench_utilization["runtime_split_selector"]["options"]],
            [577, 115, 116, 115, 115, 116],
        )
        self.assertIsNotNone(co_bench)
        self.assertEqual(co_bench["track"], "or_verified")
        self.assertTrue(co_bench["included_in_main_comparison"])
        self.assertTrue(co_bench["local_dataset_only"])
        self.assertEqual(co_bench["dataset_size"], 36)
        self.assertEqual(co_bench["split"], "official:test")
        self.assertEqual(co_bench["task_mode"], "artifact")
        self.assertFalse(co_bench["supports_runtime_config"])
        self.assertTrue(co_bench["supports_max_items"])
        self.assertEqual(co_bench["default_max_items"], 36)
        self.assertFalse(co_bench["run_baseline_verifier"])
        self.assertEqual(co_bench["runtime_split_selector"]["default_value"], "all")
        self.assertEqual(co_bench["runtime_split_selector"]["options"][0]["item_count"], 36)
        self.assertIn("travelling-salesman-problem", {option["value"] for option in co_bench["runtime_split_selector"]["options"]})
        aircraft_landing = next(
            option for option in co_bench["runtime_split_selector"]["options"] if option["value"] == "aircraft-landing"
        )
        self.assertIn("benchmark instances", aircraft_landing["description"])
        if hallulens_precise is not None:
            self.assertFalse(hallulens_precise["run_baseline_verifier"])

    def test_livecodebench_aggregate_uses_release_split_selector(self) -> None:
        tasks = {task["id"]: task for task in load_codegen_tasks()}
        task = tasks["livecodebench"]
        self.assertEqual(task["dataset_id"], "livecodebench_all")
        self.assertEqual(task["dataset_size"], 1055)
        self.assertEqual(task["split"], "v1+v2+v3+v4+v5+v6:test")
        self.assertTrue(task["lazy_item_manifest"])
        selector = task["runtime_split_selector"]
        self.assertEqual(selector["label"], "Split")
        self.assertEqual(selector["default_value"], "all")
        self.assertEqual(
            [option["value"] for option in selector["options"]],
            ["all", "v1", "v2", "v3", "v4", "v5", "v6"],
        )
        self.assertEqual(
            [option["item_count"] for option in selector["options"]],
            [1055, 400, 111, 101, 101, 167, 175],
        )
        self.assertEqual(sum(option["item_count"] for option in selector["options"][1:]), 1055)
        self.assertEqual(RELEASE_FILES["v1"], ["test.jsonl"])
        self.assertEqual(RELEASE_FILES["v2"], ["test2.jsonl"])
        self.assertEqual(RELEASE_FILES["v3"], ["test3.jsonl"])
        self.assertEqual(RELEASE_FILES["v4"], ["test4.jsonl"])
        self.assertEqual(RELEASE_FILES["v5"], ["test5.jsonl"])
        self.assertEqual(RELEASE_FILES["v6"], ["test6.jsonl"])

    def test_missing_local_benchmark_assets_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "missing-task", "path": "missing-task", "enabled": True},
                        ]
                    }
                )
            )
            with (
                patch.object(catalog, "BENCHMARK_ROOT", root),
                patch.object(catalog, "REGISTRY_PATH", registry_path),
            ):
                self.assertEqual(load_codegen_tasks(), [])
                self.assertEqual(list_codegen_task_summaries(), [])

    def test_new_personalization_tasks_are_registered_with_expected_metadata(self) -> None:
        summaries = {task["id"]: task for task in list_codegen_task_summaries()}
        expected = {
            "incharacter": ("role_play", "single_turn", "dialogue_judgement", "hybrid", 448, True),
            "characterbench": ("role_play", "single_turn", "dialogue_judgement", "judge_model", 3250, True),
            "timechara": ("role_play", "single_turn", "dialogue_judgement", "judge_model", 10895, True),
            "socialbench": ("role_play", "single_turn", "dialogue_judgement", "hybrid", 7702, True),
            "personafeedback": ("user_persona", "single_turn", "mcq", "exact_match", 8298),
            "alpsbench-extraction": ("user_persona", "single_turn", "agentic_open_ended", "rubric_score", 466),
            "alpsbench-update": ("user_persona", "single_turn", "agentic_open_ended", "rubric_score", 469),
            "alpsbench-retrieval": ("user_persona", "single_turn", "agentic_open_ended", "rubric_score", 2380),
            "alpsbench-utilization": ("user_persona", "single_turn", "dialogue_judgement", "rubric_score", 577),
            "alpbench": ("user_persona", "single_turn", "classification", "label_match", 800),
        }

        for task_id, spec in expected.items():
            if task_id not in summaries:
                continue
            if len(spec) == 6:
                category, interaction_mode, task_shape, scoring_mode, dataset_size, included = spec
            else:
                category, interaction_mode, task_shape, scoring_mode, dataset_size = spec
                included = True
            task = summaries[task_id]
            self.assertEqual(task["track"], "personalization_verified")
            self.assertEqual(task["research_line"], "personalization")
            self.assertEqual(task["personalization_category"], category)
            self.assertEqual(task["interaction_mode"], interaction_mode)
            self.assertEqual(task["task_shape"], task_shape)
            self.assertEqual(task["scoring_mode"], scoring_mode)
            self.assertEqual(task["dataset_size"], dataset_size)
            self.assertEqual(task["included_in_main_comparison"], included)

    def test_lazy_manifest_keeps_declared_dataset_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "coding_verified" / "lazy-livecodebench"
            data_dir = task_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"id": "livecodebench", "path": "coding_verified/lazy-livecodebench", "enabled": True},
                        ]
                    }
                )
            )
            (task_dir / "editable.py").write_text("def solve():\n    return None\n")
            (task_dir / "verifier.py").write_text("def evaluate_candidate(**_kwargs):\n    return {}\n")
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "id": "livecodebench",
                        "title": "Lazy LiveCodeBench",
                        "description": "Synthetic lazy dataset task.",
                        "benchmark_tier": "comparable",
                        "track": "coding_verified",
                        "dataset_id": "livecodebench_all",
                        "dataset_size": 1055,
                        "local_dataset_only": True,
                        "lazy_item_manifest": True,
                        "item_manifest": "data/questions.json",
                        "split": "v1+v2+v3+v4+v5+v6:test",
                        "runtime_split_selector": {
                            "label": "Split",
                            "default_value": "all",
                            "options": [
                                {"value": "all", "title": "All Releases", "item_count": 1055},
                                {"value": "v6", "title": "v6", "item_count": 175, "match_tags_any": ["release:v6"]},
                            ],
                        },
                        "allow_browsing": False,
                        "answer_metric": "test_pass_rate",
                        "family": "coding",
                        "task_signature": ["dataset-task"],
                        "task_mode": "artifact",
                        "interaction_mode": "single_turn",
                        "editable_file": "editable.py",
                        "verifier": "verifier.py",
                        "entry_symbol": "solve",
                        "generation_budget": 3,
                        "candidate_budget": 2,
                        "branching_factor": 3,
                        "item_workers": 6,
                        "epsilon": 0.01,
                        "objective_spec": {
                            "display_name": "Test pass rate",
                            "direction": "max",
                            "unit": "ratio",
                            "summary_template": "Higher is better.",
                            "formula": "test_pass_rate = passed_cases / total_cases"
                        }
                    }
                )
            )
            (data_dir / "questions.json").write_text(
                json.dumps(
                    {
                        "dataset_id": "livecodebench_all",
                        "dataset_size": 1055,
                        "prepared_count": 2,
                        "items": [
                            {"item_id": "item-1", "prompt": "a", "expected_answer": "ok", "metadata": {"runtime_split_tags": ["release:v6"]}},
                            {"item_id": "item-2", "prompt": "b", "expected_answer": "ok", "metadata": {"runtime_split_tags": ["release:v6"]}},
                        ],
                    }
                )
            )

            with (
                patch.object(catalog, "BENCHMARK_ROOT", root),
                patch.object(catalog, "REGISTRY_PATH", registry_path),
            ):
                task = next(item for item in load_codegen_tasks() if item["id"] == "livecodebench")
                self.assertEqual(task["dataset_size"], 1055)
                self.assertEqual(task["prepared_item_count"], 2)


if __name__ == "__main__":
    unittest.main()
