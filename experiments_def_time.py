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


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = Path(f"results_{timestamp}")
folder.mkdir(exist_ok=True)

# === NEW: тайминг всего запуска и файл с логом времени ===
start_total = time.perf_counter()
timings_path = folder / "timings.txt"
with open(timings_path, "w", encoding="utf-8") as f:
    f.write(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Отключаем лишние предупреждения от transformers
logging.set_verbosity_error()

# Путь до локальной модели
# base_name = "/home/agvanu/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B/snapshots/8f445e3628f3500ee69f24e1303c9f10f5342a39"
# base_name = "/home/agvanu/Desktop/QM/T/s1-qwen-0.5B/ckpts/s1-20250728_212621"
base_name = "/home/agvanu/models/r1-1.5B-CoTData"
# =====================
# 1. Загрузка локальной модели
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(base_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_name,
    pad_token_id=tokenizer.eos_token_id
).to(device)
# Устанавливаем pad_token_id в конфигурации, чтобы подавить повторные предупреждения
model.config.pad_token_id = tokenizer.eos_token_id

# =====================
# 2. Утилиты для генерации локально
# =====================
def generate_with_local_model(prompt: str, max_new_tokens: int = 1024) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        temperature=0.2,
        top_p=0.9
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Возвращаем только сгенерированную часть после prompt
    return text[len(prompt):].strip()

# =====================
# 3. Функции для подсчёта метрикa
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
        # если нет примеров, просто отмечаем и идём дальше
        if not y_true:
            lines.append("Total samples: 0")
            lines.append("")  # пустая строка
            continue

        # общее число образцов
        total = len(y_true)
        lines.append(f"Total samples: {total}")
        lines.append("")  # пустая строка перед первым классом

        # собираем по-классовые метрики
        labels, _, per_class = compute_per_class_counts(y_true, y_pred)
        for cls in labels:
            c = per_class[cls]
            dm = compute_derived_metrics(c['TP'], c['FP'], c['FN'], c['TN'])
            lines.append(f"Class {cls}:")
            lines.append(f"  TP={c['TP']}   FP={c['FP']}   FN={c['FN']}   TN={c['TN']}")
            lines.append(f"  TPR={dm['tpr']:.3f}   FPR={dm['fpr']:.3f}")
            lines.append("")  # пустая строка между классами

    return "\n".join(lines)
# =====================
# 4. Функции для работы с CWE-промптами
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


def generate_prompt_short(n_shot: int, test_text: str, examples: list) -> str:
    header = "Determine CWE of the code."
    if n_shot > 0:
        header += " Examples:\n"
        for code, label in examples[:n_shot]:
            header += f"Code: \"{code}\" Type: {label}\n"
        header += "\n"
    tail = f"Now: Code: \"{test_text}\" Type:"
    return header + tail


def generate_prompt_fix_code(cwe: str, code: str) -> str:
    return (
        f"The code has vulnerability {cwe}. Rewrite to fix it, return only code.\n"
        f"Vulnerable:\n{code}\nFixed:"
    )

# =====================
# 5. Sliding windows для примеров
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
# 6. Основной эксперимент без потоков
# =====================

import sys

def run_experiments_single_thread(SAFE_df, train_df, true_label, count_of_exp):
    # === NEW: старт таймера для конкретного CWE ===
    t0 = time.perf_counter()

    # results = {n: {'y_true': [], 'y_pred': []} for n in [0,1,3,5]}
    # acc = {n: 0 for n in [0,1,3,5]}
    results = {n: {'y_true': [], 'y_pred': []} for n in [0,1]}
    acc = {n: 0 for n in [0,1]}
    data = {k: [] for k in [
        'EXP-Number','CWE-type','N-shot','Input1','Output1',
        'Predicted_CWE','Code_before_GT','Code_after_GT','Code_after_predicted','Diff'
    ]}

    for i in range(count_of_exp):
        sys.stdout.write(f"\rExperiment {i+1}, {true_label}")
        sys.stdout.flush()
        rows6 = sliding_windows_df(train_df, count_of_exp, i)
        if len(rows6) < 6:
            continue
        examples = list(rows6.sample(n=5).apply(lambda r: (r['method_before'], r['cwe_id']), axis=1))
        GT = rows6.iloc[5]
        test_text, fixed_after, diff = GT['method_before'], GT['method_after'], GT['diff']

        # for n_shot in [0,1,3,5]:
        for n_shot in [0,1]:
            prompt_label = generate_prompt_short(n_shot, test_text.replace('"','\"'), examples)
            output_label = generate_with_local_model(prompt_label, max_new_tokens=10)
            pred = parse_prediction_label(output_label)

            data['EXP-Number'].append(i)
            data['CWE-type'].append(true_label)
            data['N-shot'].append(f"{n_shot}-shot")
            data['Input1'].append(prompt_label)
            data['Output1'].append(output_label)
            data['Predicted_CWE'].append(pred)
            data['Code_before_GT'].append(test_text)
            data['Code_after_GT'].append(fixed_after)
            data['Diff'].append(diff)

            if pred not in ['Unknown', 'SAFE']:
                fix_prompt = generate_prompt_fix_code(pred, test_text)
                fixed_code = generate_with_local_model(fix_prompt, max_new_tokens=2048)
            else:
                fixed_code = 'SAFE' if pred == 'SAFE' else 'N/A'
            data['Code_after_predicted'].append(fixed_code)

            if pred == normalize_label(true_label):
                acc[n_shot] += 1
            results[n_shot]['y_true'].append(normalize_label(true_label))
            results[n_shot]['y_pred'].append(pred)

    report = result_report(results)
    print(report)

    # === NEW: write the report out as a .txt ===
    txt_path = f"results_{timestamp}/results_{true_label}.txt"
    with open(txt_path, 'w', encoding="utf-8") as f:
        f.write(report)
    print(f"Saved detailed report to {txt_path}")

    pd.DataFrame(data).to_csv(f"results_{timestamp}/results_{true_label}.csv", index=False)

    # === NEW: фиксируем время для текущего CWE ===
    t1 = time.perf_counter()
    dur = t1 - t0
    with open(timings_path, "a", encoding="utf-8") as f:
        f.write(f"{true_label}: {dur:.2f} s for {count_of_exp} experiments\n")

    return results, acc, data

# =====================
# 7. Запуск по списку CWE
# =====================
if __name__ == '__main__':
    # Загрузка данных: CSV или JSONL
    data_path = "/home/agvanu/Desktop/QM/T/MComparsion/test_raw.csv"  # измените на имя файла или .csv
    # Чтение в зависимости от расширения
    if data_path.endswith('.jsonl'):
        full_df = pd.read_json(data_path, lines=True)
    elif data_path.endswith('.csv'):
        full_df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # accumulators for _all_ CWEs
    # all_results = {n: {'y_true': [], 'y_pred': []} for n in [0,1,3,5]}
    # all_acc = {n: 0 for n in [0,1,3,5]}
    all_results = {n: {'y_true': [], 'y_pred': []} for n in [0,1]}
    all_acc = {n: 0 for n in [0,1]}
    all_data = {key: [] for key in [
    'EXP-Number','CWE-type','N-shot','Input1','Output1',
    'Predicted_CWE','Code_before_GT','Code_after_GT','Code_after_predicted', 'Diff',
]}
    
    # Показать количество примеров для каждого cwe_id
    cwe_counts = full_df['cwe_id'].value_counts()
    print("Количество примеров по каждому CWE ID:")
    print(cwe_counts)

    # Проверка наличия столбца cwe_id
    if 'cwe_id' not in full_df.columns:
        print("Ошибка: в данных нет столбца 'cwe_id'. Доступные столбцы:", full_df.columns.tolist())
        raise KeyError("'cwe_id' column is required for experiment")

    # Выбор безопасных и целевых примеров
    SAFE_df = full_df[full_df['cwe_id'] == 'SAFE']
    cwe_list = ['CWE-835']

    count_of_exp = 20

    # Запуск экспериментов
    for cwe in cwe_list:
        train_df = full_df[full_df['cwe_id'] == cwe].reset_index(drop=True)
        print(f"Запуск для {cwe}: примеров {len(train_df)}")
        local_results, local_acc, local_data = run_experiments_single_thread(SAFE_df, train_df, cwe, count_of_exp)
        for shot in all_results:
            all_results[shot]['y_true'].extend(local_results[shot]['y_true'])
            all_results[shot]['y_pred'].extend(local_results[shot]['y_pred'])
            all_acc[shot] += local_acc[shot]
        for k, v in local_data.items():
            all_data[k].extend(v)

    final_report = result_report(all_results)
    print(final_report)

    # and save out a single CSV if you like:
    df_out = pd.DataFrame(all_data)

    df_out.to_csv(
        f"results_{timestamp}/ALL_CWEs_outputs.csv",
        index=False, encoding="utf-8-sig"
    )

    with open(f"results_{timestamp}/ALL_CWEs_outputs.txt", "w", encoding="utf-8") as file:
        file.write(final_report)

    # === NEW: общий тайминг всего скрипта ===
    total_elapsed = time.perf_counter() - start_total
    with open(timings_path, "a", encoding="utf-8") as f:
        f.write(f"TOTAL: {total_elapsed:.2f} s\n")
    print(f"Saved timings to {timings_path}")