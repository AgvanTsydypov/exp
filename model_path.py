from transformers import AutoModelForCausalLM

# Загружаем модель (и кэшируем её при первом запуске)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")

# Путь до локального репозитория модели
print(model.base_model_prefix)           # префикс
print(model.config._name_or_path)       # обычно именно этот идентификатор
# но можно посмотреть каталог, где он реально лежит:
from huggingface_hub import snapshot_download
local_dir = snapshot_download("Qwen/Qwen1.5-0.5B")
print(local_dir)