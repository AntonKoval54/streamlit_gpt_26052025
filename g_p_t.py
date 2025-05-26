import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Простой Чат-Бот", layout="centered")

# --- Настройки модели и кэширование ---
@st.cache_resource
def load_model():
    st.write("Загрузка модели DialoGPT-medium и токенизатора... Это может занять некоторое время (около 1.5 ГБ).")
    # 'microsoft/DialoGPT-medium' -  основана на GPT-2, адаптирована для чата.
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    model.eval()
    model.to("cpu")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    st.write("Модель DialoGPT-medium загружена и готова к работе.")
    return tokenizer, model

# Загружаем модель и токенизатор один раз при старте приложения
tokenizer, model = load_model()

st.title("Простой Чат-Бот (DialoGPT-medium, Локальный CPU)")
st.caption("Это демонстрация DialoGPT-medium, запущенной на вашем процессоре. Она обучена для диалогов и лучше справляется с поддержанием контекста.")

# Инициализация истории чата для отображения в Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Добавляем первое сообщение от ассистента
    st.session_state.messages.append({"role": "assistant", "content": "Привет! Я DialoGPT-medium. Спроси меня что-нибудь."})

# Инициализация истории токенов для DialoGPT (для поддержания контекста)
if "conversation_history_ids" not in st.session_state:
    st.session_state.conversation_history_ids = None

# Отображение предыдущих сообщений чата в интерфейсе
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Поле для ввода нового сообщения пользователя
if prompt := st.chat_input("Напишите свое сообщение здесь..."):
    # Добавляем сообщение пользователя в историю для отображения
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Генерируем ответ модели
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            # Токенизируем новый ввод пользователя и добавляем EOS токен
            new_user_input_ids = tokenizer.encode(
                prompt + tokenizer.eos_token, return_tensors='pt'
            ).to("cpu")

            # Объединяем новую историю пользователя с предыдущей историей чата
            if st.session_state.conversation_history_ids is None:
                bot_input_ids = new_user_input_ids
            else:
                bot_input_ids = torch.cat(
                    [st.session_state.conversation_history_ids, new_user_input_ids], dim=-1
                )

            # Генерируем ответ
            max_response_length = 200 # Максимальное количество новых токенов для ответа бота
            current_input_length = bot_input_ids.shape[-1] # Длина текущего контекста

            generated_ids = model.generate(
                bot_input_ids,
                max_length=current_input_length + max_response_length, # Общая максимальная длина
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, # Включаем семплирование для более разнообразных ответов
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
            )

            st.session_state.conversation_history_ids = generated_ids

            # Декодируем только сгенерированную часть (ответ бота)
            response_text = tokenizer.decode(
                generated_ids[0, current_input_length:], skip_special_tokens=True
            )

            # Дополнительная очистка ответа (на случай, если модель сгенерирует что-то лишнее)
            if tokenizer.eos_token in response_text:
                response_text = response_text.split(tokenizer.eos_token)[0].strip()
            # Иногда модель может начать свой ответ с "bot: " или повторить промт, хотя DialoGPT
            # обычно хорошо справляется с этим.
            if response_text.lower().startswith("bot:"):
                response_text = response_text[4:].strip()

            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
