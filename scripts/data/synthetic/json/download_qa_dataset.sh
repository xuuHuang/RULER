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

# wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json
# wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O hotpotqa.json

mkdir -p qas
curl -o qas/xquad_en.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_en.json
curl -o qas/xquad_zh.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_zh.json
curl -o qas/xquad_ar.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_ar.json
curl -o qas/xquad_bn.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_bn.json
curl -o qas/xquad_cs.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_cs.json
curl -o qas/xquad_de.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_de.json
curl -o qas/xquad_es.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_es.json
curl -o qas/xquad_fr.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_fr.json
curl -o qas/xquad_hu.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_hu.json
curl -o qas/xquad_ja.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_ja.json
curl -o qas/xquad_ko.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_ko.json
curl -o qas/xquad_ru.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_ru.json
curl -o qas/xquad_sr.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_sr.json
curl -o qas/xquad_sw.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_sw.json
curl -o qas/xquad_te.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_te.json
curl -o qas/xquad_th.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_th.json
curl -o qas/xquad_vi.json https://huggingface.co/datasets/ggdcr/Question_answering/resolve/main/qas/xquad_vi.json