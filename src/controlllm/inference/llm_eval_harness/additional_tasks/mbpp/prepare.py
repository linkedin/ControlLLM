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

import json
from pathlib import Path
# pre-requisite: install evalplus from https://github.com/evalplus/evalplus
from evalplus.data.mbpp import get_mbpp

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    output_file = str(data_dir / f"test.jsonl")

    # this downloads the data of mbpp sanitzed, not mbpp full
    problems = get_mbpp()

    with open(output_file, "wt", encoding="utf-8") as fout:
        for problem in problems.values():
            # somehow models like tabs more than spaces
            problem['question'] = problem['prompt'].replace('    ', '\t')
            fout.write(json.dumps(problem) + "\n")
