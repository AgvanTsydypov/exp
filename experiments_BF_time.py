import random
import pandas as pd
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from datetime import datetime
from IPython.display import display, update_display, clear_output, HTML
from pathlib import Path
import sys
def exploop(count):
    # =====================
    # 0. Config & output folder
    # =====================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(f"results_{timestamp}")
    folder.mkdir(exist_ok=True)

    # s1-style Budget Forcing switches/params
    USE_BUDGET_FORCING = True          # <--- теперь BF применяется ТОЛЬКО для fix_code
    BF_MAX_THINK_TOKENS = count          # бюджет "мыслей" для фикса
    BF_NUM_IGNORE = 0                  # сколько раз игнорировать закрытие мыслей
    BF_WAIT_TOKEN = " Wait"            # токен-продление
    BF_THINK_OPEN = "<think>"
    BF_THINK_CLOSE = "</think>"

    # generation params
    GEN_TEMPERATURE = 0.2
    GEN_TOP_P = 0.9
    GEN_DO_SAMPLE = False
    LABEL_MAX_NEW_TOKENS = 10         # лимит токенов для вывода метки
    FIX_MAX_NEW_TOKENS = 4096          # лимит токенов для итогового кода-фикса

    # =====================
    # 1. Transformers logging & model load (FAST)
    # =====================
    logging.set_verbosity_error()

    # Путь до локальной модели
    base_name = "/home/agvanu/models/Qwen2.5 - 0.5B-cotData"
    # base_name = "/home/agvanu/models/r1-1.5B-CoTData"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    tokenizer.pad_token = tokenizer.eos_token

    # dtype: bf16 (Ampere+), иначе fp16 на GPU, иначе fp32 на CPU
    if torch.cuda.is_available():
        major_cc, _ = torch.cuda.get_device_capability(0)
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        pad_token_id=tokenizer.eos_token_id,
    ).to(device).eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    # =====================
    # 2. Low-level helpers (FAST)
    # =====================
    def _generate_raw(prompt: str, max_new_tokens: int):
        """
        Быстрый генератор: возвращает (new_text, new_tokens_count) без повторной токенизации.
        """
        with torch.inference_mode():
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # GEN_DO_SAMPLE=False
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )
        seq = out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = int(seq.shape[0] - prompt_len)
        # декодируем только добавленные токены
        new_text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        return new_text.strip(), new_tokens

    def generate_with_local_model(prompt: str, max_new_tokens: int = 1024):
        """
        Обёртка: возвращает (text, new_tokens).
        """
        return _generate_raw(prompt, max_new_tokens=max_new_tokens)

    # =====================
    # 3. Budget Forcing "thinking" phase (FAST) — ИСПОЛЬЗУЕМ ТОЛЬКО ДЛЯ FIX
    # =====================
    def budget_forcing_think(prompt_prefix: str,
                            max_thinking_tokens: int = BF_MAX_THINK_TOKENS,
                            num_ignore: int = BF_NUM_IGNORE,
                            wait_token: str = BF_WAIT_TOKEN):
        """
        Генерация рассуждений внутри <think>...</think> с лимитом по токенам и подавлением закрытия.
        Возвращает: (reasoning_text, elapsed_seconds, total_generated_tokens)
        """
        start = time.perf_counter()
        remaining = max_thinking_tokens
        total_tokens = 0
        reasoning_accum = ""

        # стартуем внутри "мыслей"
        prompt = prompt_prefix + BF_THINK_OPEN

        while remaining > 0:
            chunk, new_tok = _generate_raw(prompt, max_new_tokens=remaining)
            # защита от редкого случая нулевого прироста (ранний EOS и т.п.)
            if new_tok <= 0 and chunk == "":
                break

            total_tokens += max(new_tok, 0)
            remaining -= max(new_tok, 0)

            reasoning_accum += chunk
            close_idx = reasoning_accum.find(BF_THINK_CLOSE)

            if close_idx != -1:
                # модель попыталась закрыть мысли
                if num_ignore > 0:
                    kept = reasoning_accum[:close_idx]
                    prompt = prompt_prefix + BF_THINK_OPEN + kept + wait_token
                    reasoning_accum = kept
                    num_ignore -= 1
                    continue
                else:
                    reasoning_accum = reasoning_accum[:close_idx]
                    break

            if remaining <= 0:
                break

            # продолжаем размышление
            prompt = prompt_prefix + BF_THINK_OPEN + reasoning_accum

        elapsed = time.perf_counter() - start
        return reasoning_accum.strip(), elapsed, total_tokens

    # =====================
    # 4. Metrics helpers (unchanged)
    # =====================
    def compute_derived_metrics(TP, FP, FN, TN):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "specificity": specificity,
            "npv": npv,
            "tpr": recall,
            "fpr": fpr,
        }

    from sklearn.metrics import confusion_matrix

    def compute_per_class_counts(y_true, y_pred):
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        total = cm.sum()
        counts = {}
        for i, cls in enumerate(labels):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = total - (TP + FP + FN)
            counts[cls] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
        return labels, cm, counts

    def result_report(results):
        lines = []
        for n_shot, data in results.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            lines.append(f"--- {n_shot}-shot results ---")
            if not y_true:
                lines.append("Total samples: 0")
                lines.append("")
                continue
            total = len(y_true)
            lines.append(f"Total samples: {total}")
            lines.append("")
            labels, _, per_class = compute_per_class_counts(y_true, y_pred)
            for cls in labels:
                c = per_class[cls]
                dm = compute_derived_metrics(c['TP'], c['FP'], c['FN'], c['TN'])
                lines.append(f"Class {cls}:")
                lines.append(f"  TP={c['TP']}   FP={c['FP']}   FN={c['FN']}   TN={c['TN']}")
                lines.append(f"  TPR={dm['tpr']:.3f}   FPR={dm['fpr']:.3f}")
                lines.append("")
        return "\n".join(lines)

    # =====================
    # 5. CWE helpers (unchanged)
    # =====================
    def normalize_label(label: str) -> str:
        lab = label.strip().upper()
        if lab == 'SAFE':
            return 'SAFE'
        m = re.search(r'CWE-?(\d+)', lab)
        return f"CWE{m.group(1)}" if m else lab

    def parse_prediction_label(text: str) -> str:
        if re.search(r"\bSAFE\b", text, flags=re.IGNORECASE):
            return 'SAFE'
        m = re.search(r'CWE-?(\d+)', text, flags=re.IGNORECASE)
        return normalize_label(f"CWE{m.group(1)}") if m else 'Unknown'

    # =====================
    # 6. Prompts
    # =====================
    def generate_prompt_short(n_shot: int, test_text: str, examples: list) -> str:
        header = "Determine CWE of the code."
        if n_shot > 0:
            header += " Examples:\n"
            for code, label in examples[:n_shot]:
                header += f"Code: \"{code}\" Type: {label}\n"
            header += "\n"
        tail = f"Now: Code: \"{test_text}\" Type:"
        return header + tail

    def generate_fix_think_prefix(cwe: str, code: str) -> str:
        return (
            "You are a security expert.\n"
            f"Goal: fix the vulnerability {cwe} in the following code.\n"
            f"First, think step by step between <think> and </think>. "
            f"Do NOT output code yet.\n"
            f"Vulnerable code:\n{code}\n"
        )

    def generate_prompt_fix_code(cwe: str, code: str, reasoning: str | None = None) -> str:
        extra = ""
        if reasoning:
            extra = (
                f"Use this reasoning to guide the patch:\n{BF_THINK_OPEN}{reasoning}{BF_THINK_CLOSE}\n"
            )
        return (
            f"{extra}"
            f"The code has vulnerability {cwe}. Rewrite to fix it.\n"
            f"Return ONLY the fixed code (no comments, no explanations).\n"
            f"Vulnerable:\n{code}\nFixed:\n"
        )

    # =====================
    # 7. Sliding windows for examples (unchanged)
    # =====================
    def sliding_windows_df(df, count_of_exp, i):
        N, L = len(df), 6
        if count_of_exp <= 0:
            return df.iloc[0:0]
        if count_of_exp == 1:
            return df.iloc[0:L]
        step = (N - L) / (count_of_exp - 1)
        start = N - L if i == count_of_exp - 1 else int(i * step)
        return df.iloc[start:start+L]

    # =====================
    # 8. Prediction helpers
    # =====================
    def predict_label(n_shot, test_text, examples):
        """Без BF: метка уязвимости."""
        prompt_label = generate_prompt_short(n_shot, test_text.replace('"','\"'), examples)
        t0 = time.perf_counter()
        output_label, ans_tokens = generate_with_local_model(prompt_label, max_new_tokens=LABEL_MAX_NEW_TOKENS)
        t_ans = time.perf_counter() - t0
        return output_label, t_ans, ans_tokens

    def fix_code_with_optional_bf(cwe_label: str, code: str):
        """
        Генерация фикса:
        - если USE_BUDGET_FORCING=True: BF-thinking -> финальный код (учёт времени/токенов)
        - иначе: сразу финальный код
        Возврат: fixed_code, t_fix_think, fix_think_tokens, t_fix_ans, fix_ans_tokens, reasoning_text
        """
        if not USE_BUDGET_FORCING:
            # без мыслей
            fix_prompt = generate_prompt_fix_code(cwe_label, code)
            t1 = time.perf_counter()
            fixed_code, fix_ans_tokens = generate_with_local_model(fix_prompt, max_new_tokens=FIX_MAX_NEW_TOKENS)
            t_fix_ans = time.perf_counter() - t1
            return fixed_code, 0.0, 0, t_fix_ans, fix_ans_tokens, ""
        # с мыслями (BF)
        think_prefix = generate_fix_think_prefix(cwe_label, code)
        reasoning_text, t_fix_think, fix_think_tokens = budget_forcing_think(
            prompt_prefix=think_prefix,
            max_thinking_tokens=BF_MAX_THINK_TOKENS,
            num_ignore=BF_NUM_IGNORE,
            wait_token=BF_WAIT_TOKEN
        )
        fix_prompt = generate_prompt_fix_code(cwe_label, code, reasoning_text)
        t2 = time.perf_counter()
        fixed_code, fix_ans_tokens = generate_with_local_model(fix_prompt, max_new_tokens=FIX_MAX_NEW_TOKENS)
        t_fix_ans = time.perf_counter() - t2
        return fixed_code, t_fix_think, fix_think_tokens, t_fix_ans, fix_ans_tokens, reasoning_text

    # =====================
    # 9. Main experiment (single thread) + timing + token counts
    # =====================
    def run_experiments_single_thread(SAFE_df, train_df, true_label, count_of_exp):
        # 0- и 1-shot в текущей конфигурации
        results = {n: {'y_true': [], 'y_pred': []} for n in [1,3]}
        acc = {n: 0 for n in [1,3]}
        data = {k: [] for k in [
            'EXP-Number','CWE-type','N-shot',
            # inputs/outputs
            'Input_Label','Output_Label','Predicted_CWE',
            'Code_before_GT','Code_after_GT','Diff',
            'Code_after_predicted',
            # Label stage metrics
            'LabelAnswerTimeSec','LabelAnswerTokens',
            # Fix stage metrics (BF applies here)
            'FixThinkTimeSec','FixAnswerTimeSec','FixTotalGenTimeSec',
            'FixThinkTokens','FixAnswerTokens','FixTotalNewTokens',
            'FixGenerated_Reasoning',
            # BF config snapshot
            'BF_Used','BF_TokensBudget','BF_NumIgnore','BF_WaitToken',
        ]}

        # агрегаты по блоку CWE
        total_label_answer_tokens_block = 0
        total_fix_think_tokens_block = 0
        total_fix_answer_tokens_block = 0

        run_start = time.perf_counter()

        for i in range(count_of_exp):
            sys.stdout.write(f"\rExperiment {i+1}, {true_label}")
            sys.stdout.flush()

            rows6 = sliding_windows_df(train_df, count_of_exp, i)
            if len(rows6) < 6:
                continue
            examples = list(rows6.sample(n=5).apply(lambda r: (r['method_before'], r['cwe_id']), axis=1))
            GT = rows6.iloc[5]
            test_text, fixed_after, diff = GT['method_before'], GT['method_after'], GT['diff']

            for n_shot in [1,3]:
                # 1) LABEL (без BF)
                out_label, t_label_ans, label_ans_tokens = predict_label(n_shot, test_text, examples)
                pred = parse_prediction_label(out_label)

                # 2) FIX (BF только здесь, если есть что фиксить)
                if pred not in ['Unknown', 'SAFE']:
                    fixed_code, t_fix_think, fix_think_tokens, t_fix_ans, fix_ans_tokens, fix_reasoning = \
                        fix_code_with_optional_bf(pred, test_text)
                else:
                    fixed_code = 'SAFE' if pred == 'SAFE' else 'N/A'
                    t_fix_think = 0.0
                    fix_think_tokens = 0
                    t_fix_ans = 0.0
                    fix_ans_tokens = 0
                    fix_reasoning = ""

                # агрегаты
                total_label_answer_tokens_block += label_ans_tokens
                total_fix_think_tokens_block += fix_think_tokens
                total_fix_answer_tokens_block += fix_ans_tokens

                # лог-строка
                data['EXP-Number'].append(i)
                data['CWE-type'].append(true_label)
                data['N-shot'].append(f"{n_shot}-shot")
                data['Input_Label'].append(generate_prompt_short(n_shot, test_text.replace('"','\"'), examples))
                data['Output_Label'].append(out_label)
                data['Predicted_CWE'].append(pred)
                data['Code_before_GT'].append(test_text)
                data['Code_after_GT'].append(fixed_after)
                data['Diff'].append(diff)
                data['Code_after_predicted'].append(fixed_code)

                # метрики: Label
                data['LabelAnswerTimeSec'].append(round(t_label_ans, 6))
                data['LabelAnswerTokens'].append(label_ans_tokens)

                # метрики: Fix
                data['FixThinkTimeSec'].append(round(t_fix_think, 6))
                data['FixAnswerTimeSec'].append(round(t_fix_ans, 6))
                data['FixTotalGenTimeSec'].append(round(t_fix_think + t_fix_ans, 6))
                data['FixThinkTokens'].append(fix_think_tokens)
                data['FixAnswerTokens'].append(fix_ans_tokens)
                data['FixTotalNewTokens'].append(fix_think_tokens + fix_ans_tokens)
                data['FixGenerated_Reasoning'].append(fix_reasoning)

                # BF конфиг (на момент запуска)
                data['BF_Used'].append(int(USE_BUDGET_FORCING))
                data['BF_TokensBudget'].append(BF_MAX_THINK_TOKENS if USE_BUDGET_FORCING else 0)
                data['BF_NumIgnore'].append(BF_NUM_IGNORE if USE_BUDGET_FORCING else 0)
                data['BF_WaitToken'].append(BF_WAIT_TOKEN if USE_BUDGET_FORCING else "")

                # метрики классификации
                if pred == normalize_label(true_label):
                    acc[n_shot] += 1
                results[n_shot]['y_true'].append(normalize_label(true_label))
                results[n_shot]['y_pred'].append(pred)

        run_elapsed = time.perf_counter() - run_start

        # report
        report = result_report(results)
        print("\n" + report)
        print(f"[{true_label}] block time: {run_elapsed:.3f} sec")
        print(f"[{true_label}] LABEL tokens — answer: {total_label_answer_tokens_block}")
        print(f"[{true_label}] FIX tokens   — think: {total_fix_think_tokens_block}, answer: {total_fix_answer_tokens_block}, "
            f"total: {total_fix_think_tokens_block + total_fix_answer_tokens_block}")

        # save files
        txt_path = f"results_{timestamp}/results_{true_label}.txt"
        with open(txt_path, 'w', encoding="utf-8") as f:
            f.write(
                report
                + f"\n\n[{true_label}] block time: {run_elapsed:.6f} sec\n"
                "Stages:\n"
                f"  LABEL: AnswerTokens={total_label_answer_tokens_block}\n"
                f"  FIX:   ThinkTokens={total_fix_think_tokens_block}, AnswerTokens={total_fix_answer_tokens_block}, "
                f"TotalNewTokens={total_fix_think_tokens_block + total_fix_answer_tokens_block}\n"
                + f"BF_Used={USE_BUDGET_FORCING}, BF_MAX_THINK_TOKENS={BF_MAX_THINK_TOKENS}, "
                f"BF_NUM_IGNORE={BF_NUM_IGNORE}, BF_WAIT_TOKEN='{BF_WAIT_TOKEN}'\n"
            )
        print(f"Saved detailed report to {txt_path}")

        pd.DataFrame(data).to_csv(f"results_{timestamp}/results_{true_label}.csv", index=False)
        return (results, acc, data, run_elapsed,
                total_label_answer_tokens_block,
                total_fix_think_tokens_block, total_fix_answer_tokens_block)

    # =====================
    # 10. Main
    # =====================
    if __name__ == '__main__':
        total_start = time.perf_counter()

        # Load data
        data_path = "/home/agvanu/Desktop/QM/T/MComparsion/test_raw.csv"
        if data_path.endswith('.jsonl'):
            full_df = pd.read_json(data_path, lines=True)
        elif data_path.endswith('.csv'):
            full_df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # accumulators for _all_ CWEs
        all_results = {n: {'y_true': [], 'y_pred': []} for n in [1,3]}
        all_acc = {n: 0 for n in [1,3]}
        all_data = {key: [] for key in [
            'EXP-Number','CWE-type','N-shot',
            'Input_Label','Output_Label','Predicted_CWE',
            'Code_before_GT','Code_after_GT','Diff','Code_after_predicted',
            'LabelAnswerTimeSec','LabelAnswerTokens',
            'FixThinkTimeSec','FixAnswerTimeSec','FixTotalGenTimeSec',
            'FixThinkTokens','FixAnswerTokens','FixTotalNewTokens',
            'FixGenerated_Reasoning',
            'BF_Used','BF_TokensBudget','BF_NumIgnore','BF_WaitToken',
        ]}

        # Show counts
        cwe_counts = full_df['cwe_id'].value_counts()
        print("Количество примеров по каждому CWE ID:")
        print(cwe_counts)

        if 'cwe_id' not in full_df.columns:
            print("Ошибка: в данных нет столбца 'cwe_id'. Доступные столбцы:", full_df.columns.tolist())
            raise KeyError("'cwe_id' column is required for experiment")

        SAFE_df = full_df[full_df['cwe_id'] == 'SAFE']
        cwe_list = ['CWE-190','CWE-416','CWE-835']
        count_of_exp = 20

        per_cwe_times = []
        total_label_answer_tokens_all = 0
        total_fix_think_tokens_all = 0
        total_fix_answer_tokens_all = 0

        for cwe in cwe_list:
            train_df = full_df[full_df['cwe_id'] == cwe].reset_index(drop=True)
            print(f"Запуск для {cwe}: примеров {len(train_df)}")
            (local_results, local_acc, local_data, block_time,
            block_label_answer_tokens,
            block_fix_think_tokens, block_fix_answer_tokens) = \
                run_experiments_single_thread(SAFE_df, train_df, cwe, count_of_exp)

            per_cwe_times.append((cwe, block_time))
            total_label_answer_tokens_all += block_label_answer_tokens
            total_fix_think_tokens_all += block_fix_think_tokens
            total_fix_answer_tokens_all += block_fix_answer_tokens

            for shot in all_results:
                all_results[shot]['y_true'].extend(local_results[shot]['y_true'])
                all_results[shot]['y_pred'].extend(local_results[shot]['y_pred'])
                all_acc[shot] += local_acc[shot]
            for k, v in local_data.items():
                all_data[k].extend(v)

        final_report = result_report(all_results)
        print(final_report)

        df_out = pd.DataFrame(all_data)
        out_csv = f"results_{timestamp}/ALL_CWEs_outputs.csv"
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        with open(f"results_{timestamp}/ALL_CWEs_outputs.txt", "w", encoding="utf-8") as file:
            file.write(final_report)

        # Save timing summary
        total_elapsed = time.perf_counter() - total_start
        timing_txt = folder / "TIMING_SUMMARY.txt"
        with open(timing_txt, "w", encoding="utf-8") as f:
            f.write(f"Total elapsed: {total_elapsed:.6f} sec\n")
            f.write(
                "Totals across all CWEs:\n"
                f"  LABEL AnswerTokens(all): {total_label_answer_tokens_all}\n"
                f"  FIX   ThinkTokens(all):  {total_fix_think_tokens_all}\n"
                f"  FIX   AnswerTokens(all): {total_fix_answer_tokens_all}\n"
                f"  FIX   NewTokens(all):    {total_fix_think_tokens_all + total_fix_answer_tokens_all}\n\n"
            )
            f.write(f"BudgetForcing(on FIX): {USE_BUDGET_FORCING}, "
                    f"MAX_THINK_TOKENS={BF_MAX_THINK_TOKENS}, NUM_IGNORE={BF_NUM_IGNORE}, WAIT_TOKEN='{BF_WAIT_TOKEN}'\n")
            for cwe, secs in per_cwe_times:
                f.write(f"{cwe}: {secs:.6f} sec\n")
        print(f"Saved timing summary to {timing_txt}")

loopSet = [4096, 2048, 1024, 512]
for i in loopSet:
    exploop(i)