import os, sys, pathlib

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)


from dataset.mss_bench import load_mss_bench_annotated
from dataset.scienceqa import load_science_qa
from dataset.siuo import load_siuo_dataset
from dataset.mm_safety import load_mm_safetybench_annotated
from dataset.spa_vl_harm import load_spa_vl_harm_annotated
from dataset.mme import load_mme
from dataset.mmbench import load_mmbench
from dataset.textvqa import load_textvqa


from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import as_completed
import json, logging, argparse, traceback

from tqdm.auto import tqdm

from models import ModelInterface, Models
from rag import MinimalRAG


from enum import Enum

this_folder = str(os.path.dirname(__file__))


class Dataset(Enum):
    MM_SAFETY = (
        "MM-Safety",
        load_mm_safetybench_annotated,
        "MMSB",
        "kbs/kb_MMSB.md",
        {"sample_size": 200, "seed": 0},
    )
    SIUO = ("SIUO", load_siuo_dataset, "SIUO", "kbs/kb_SIUO.md", {})
    SPA_VL_HARM = ("SPA-VL_Harm", load_spa_vl_harm_annotated, "SPAVLH", "kbs/kb_SPA_VL_HARM.md", {})
    MSS_BENCH = ("MSS-Bench", load_mss_bench_annotated, "MSSB", "kbs/kb_MSSB.md", {})
    SCIENCE_QA = ("ScienceQA", load_science_qa, "SQA", "kbs/kb_SIUO.md", {})
    MME = ("MME-Benchmark", load_mme, "MME", "kbs/kb_SIUO.md", {})
    MMBENCH = ("MMBench", load_mmbench, "MMBENCH", "kbs/kb_SIUO.md", {})
    TEXTVQA = ("TextVQA", load_textvqa, "TVQA", "kbs/kb_SIUO.md", {})

    def __init__(self, display_name: str, load_ds_func, prefix: str, kb: str, ds_kwargs: dict):
        self.display_name = display_name
        self.load_ds_func = load_ds_func
        self.prefix = prefix
        self.kb = os.path.normpath(os.path.join(this_folder, kb))
        self.ds_kwargs = ds_kwargs


def process_datapoint(
    rag_system: MinimalRAG, datapoint: dict[str, str | int | dict], k: int, min_sim: float, include_empty: bool = False
):

    question = str(datapoint["question"])
    image_path = str(datapoint["image"])
    try:

        # try:
        #     orig_resp = datapoint["original_response"]["response"]
        #     ror = datapoint["rag_on_response"]
        #     ror_safe = ror["resp_safe"]
        #     ror_reason = ror["resp_reason"]
        #     ref_des = ror["rephrased"]
        #     ror_response = ror["response"]
        #     ror_retrieved = [r["retrieved"] for r in ror["retrieved"]]
        #     ror_probs = [r["similarity"] for r in ror["retrieved"]]
        #
        # except KeyError:

        orig_resp = rag_system.generate_vanilla(question=question, image_path=image_path)

        ror_safe, ror_reason, ror_retrieved, ror_probs, ref_des = rag_system.judge_ror(
            question=question, image_path=image_path, min_sim=min_sim, k=k
        )
        ror_response = (
            rag_system.safe_response(question=question, retrieved=ror_retrieved, image_path=image_path)
            if not ror_safe
            else orig_resp
        )

        result = {
            **datapoint,
            "original_response": {"response": orig_resp},
            "rag_on_response": {
                "response": ror_response,
                "retrieved": [{"similarity": p, "retrieved": r} for r, p in zip(ror_retrieved, ror_probs)],
                "rephrased": ref_des,
                "resp_safe": ror_safe,
                "resp_reason": ror_reason,
            },
        }

        if include_empty:
            try:
                empty = datapoint["empty_context"]
                empty_safe = empty["resp_safe"]
                empty_reason = empty["resp_reason"]
                empty_response = empty["response"]
            except KeyError:
                empty_safe, empty_reason = rag_system.decide_safe(question=question, image_path=image_path, contexts=[])
                empty_response = (
                    rag_system.safe_response(question=question, retrieved=[], image_path=image_path)
                    if not empty_safe
                    else orig_resp
                )

            result = result | {
                "empty_context": {"response": empty_response, "resp_safe": empty_safe, "resp_reason": empty_reason},
            }

        return result

    except Exception as e:
        logging.error(f"Error processing datapoint: {e}\n{traceback.format_exc()}")
        return datapoint


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset",
        type=lambda s: next((d for d in Dataset if d.display_name == s), None),
        choices=[d for d in Dataset],
        default=Dataset.SIUO.display_name,
        help="(default: %(default)s) dataset to use.",
    )
    parser.add_argument(
        "--vlm_type",
        type=lambda s: next((m for m in Models if m.type_name == s), None),
        choices=[m for m in Models],
        default=Models.QWEN3VL.type_name,
        help="(default: %(default)s) model type of the VLM.",
    )
    parser.add_argument(
        "--vlm_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="(default: %(default)s) VLM model name."
    )
    parser.add_argument(
        "--sen_emb", type=str, default="all-MiniLM-L6-v2", help="(default: %(default)s) sentence-transformers model."
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        default=False,
        help="(default: %(default)s) ablation without knowledge base.",
    )
    parser.add_argument("--min_sim", type=float, default=0.25, help="(default: %(default)s) minimum similarity.")
    parser.add_argument(
        "--max_pick", type=int, default=2, help="(default: %(default)s) maximum number of candidates to accept."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="(default: %(default)s) maximum number of concurrent datapoints to process.",
    )

    args = parser.parse_args()

    if args.dataset is None:
        valid_names = [d.display_name for d in Dataset]
        raise ValueError(f"Invalid dataset. Valid options: {valid_names}")

    filepath = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)),
                "result",
                f"{args.dataset.prefix}_{args.vlm_id.replace('/', '__')}.json",
            )
        )
    )

    dataset = args.dataset.load_ds_func(**args.dataset.ds_kwargs)

    results = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as rag_file:
            results = json.load(rag_file)

    logging.info(f"length of results: {len(results)}")

    def merge_dataset(old_results, new_results):
        merged_dict = {result["image"]: result for result in old_results}

        for new_result in new_results:
            img_key = new_result["image"]
            merged_dict[img_key] = new_result  # Updates existing, or inserts new

        return list(merged_dict.values())

    dataset = merge_dataset(dataset, results)
    rag_system = None
    try:
        if dataset:
            rag_system = MinimalRAG(
                kb_path=args.dataset.kb,
                embedder_name=args.sen_emb,
                model_interface=ModelInterface(model_id=args.vlm_id, max_new_tokens=args.vlm_type.max_new_tokens),
            )

            with ThreadPoolExecutor(max_workers=args.batch_size) as tpe:
                futures = []
                for datapoint in tqdm(dataset, desc="Dispatching"):
                    try:
                        orig_resp = datapoint["original_response"]["response"]
                        ror = datapoint["rag_on_response"]
                        ror_safe = ror["resp_safe"]
                        ror_reason = ror["resp_reason"]
                        ref_des = ror["rephrased"]
                        ror_response = ror["response"]
                        ror_retrieved = [r["retrieved"] for r in ror["retrieved"]]
                        ror_probs = [r["similarity"] for r in ror["retrieved"]]
                    except (KeyError, ValueError):
                        future = tpe.submit(
                            process_datapoint,
                            rag_system,
                            datapoint,
                            args.max_pick,
                            args.min_sim,
                            args.include_empty,
                        )
                        futures.append(future)

                p_bar = tqdm(as_completed(futures), total=len(futures), desc="Collecting")

                for future in p_bar:
                    results = merge_dataset(results, [future.result()])
                    with open(filepath, "w", encoding="utf-8") as rag_file:
                        json.dump(results, rag_file, ensure_ascii=False, indent="\t")

    except KeyboardInterrupt:
        pass

    finally:

        with open(filepath, "w", encoding="utf-8") as rag_file:
            json.dump(results, rag_file, ensure_ascii=False, indent="\t")
