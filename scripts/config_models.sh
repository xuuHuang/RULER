# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    131072
    # 65536
    # 32768
    # 16384
    # 8192
    # 4096
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    
    case $MODEL_NAME in
        llama2-7b-chat)
            MODEL_PATH="${MODEL_DIR}/llama2-7b-chat-hf"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="vllm"
            ;;
        llama3.1-8b-chat)
            MODEL_PATH="${MODEL_DIR}/llama3.1-8b-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="vllm"
            ;;
        llama3.1-70b-chat)
            MODEL_PATH="${MODEL_DIR}/Llama3.1-70B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            MODEL_FRAMEWORK="vllm"
            ;;
        qwen2.5-7b-chat)
            MODEL_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct"
            MODEL_TEMPLATE_TYPE="qwen2.5"
            MODEL_FRAMEWORK="vllm"
            ;;
        qwen2.5-72b-chat)
            MODEL_PATH="${MODEL_DIR}/Qwen2.5-72B-Instruct"
            MODEL_TEMPLATE_TYPE="qwen2.5"
            MODEL_FRAMEWORK="vllm"
            ;;
        internlm2.5-7b-chat-1m)
            MODEL_PATH="${MODEL_DIR}/InternLM2_5-7B-chat-1m"
            MODEL_TEMPLATE_TYPE="internlm2.5"
            MODEL_FRAMEWORK="vllm"
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
            ;;
        aya-expanse-32b)
            MODEL_PATH="${MODEL_DIR}/Aya-Expanse-32B"
            MODEL_TEMPLATE_TYPE="aya-expanse"
            MODEL_FRAMEWORK="vllm"
            ;;
        deepseek-v3)
            MODEL_PATH="${MODEL_DIR}/DeepSeek-V3"
            MODEL_TEMPLATE_TYPE="deepseek-v3"
            MODEL_FRAMEWORK="vllm"
            ;;
        jamba1.5-mini)
            MODEL_PATH="${MODEL_DIR}/Jamba-1.5-Mini"
            MODEL_TEMPLATE_TYPE="jamba"
            MODEL_FRAMEWORK="vllm"
            ;;
        gpt-3.5-turbo)
            MODEL_PATH="gpt-3.5-turbo-0125"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="cl100k_base"
            TOKENIZER_TYPE="openai"
            OPENAI_API_KEY=""
            AZURE_ID=""
            AZURE_SECRET=""
            AZURE_ENDPOINT=""
            ;;
        gpt-4-turbo)
            MODEL_PATH="gpt-4"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="cl100k_base"
            TOKENIZER_TYPE="openai"
            OPENAI_API_KEY=""
            AZURE_ID=""
            AZURE_SECRET=""
            AZURE_ENDPOINT=""
            ;;
        gpt-4o-mini)
            MODEL_PATH="gpt-4o-mini"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="openai"
            TOKENIZER_PATH="o200k_base"
            TOKENIZER_TYPE="openai"
            OPENAI_API_KEY=""
            ;;
        gemini_1.0_pro)
            MODEL_PATH="gemini-1.0-pro-latest"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="gemini"
            TOKENIZER_PATH=$MODEL_PATH
            TOKENIZER_TYPE="gemini"
            GEMINI_API_KEY=""
            ;;
        gemini_1.5_pro)
            MODEL_PATH="gemini-1.5-pro-latest"
            MODEL_TEMPLATE_TYPE="base"
            MODEL_FRAMEWORK="gemini"
            TOKENIZER_PATH=$MODEL_PATH
            TOKENIZER_TYPE="gemini"
            GEMINI_API_KEY=""
            ;;
    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
