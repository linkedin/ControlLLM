import datasets


def doc_to_text(doc: dict) -> str:
    return doc["input_final_prompts"][0]


def process_docs_cot_zh(dataset: datasets.Dataset, pretrain=False) -> datasets.Dataset:
    # follow example of https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__mmlu__0_shot__cot__details?row=0
    # example: [ "<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\n\nQuestion: A 26-year-old woman is brought to the emergency department because of an 8-hour history of severe back and abdominal pain and mild but persistent vaginal bleeding. Ultrasonography of the abdomen shows a 2-cm ectopic pregnancy in the ampulla. The ampulla has ruptured into the surrounding tissue. Fluid from this rupture will most likely be found in which of the following locations?\nA. Lesser peritoneal cavity\nB. Mesometrium\nC. Pouch of Douglas\nD. Uterine cavity\n\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" ]
    # input_final_prompt = "<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\n\nQuestion: {Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    input_final_prompt = (
        ("<|start_header_id|>user<|end_header_id|>\n\n" if not pretrain else "")
        + "在以下问题和四个候选答案（A、B、C 和 D）中，选择最佳答案。\n\n"
        + "问题：{Question}\n"
        + "A. {A}\n"
        + "B. {B}\n"
        + "C. {C}\n"
        + "D. {D}\n\n"
        + "- 对于简单的问题：\n"
        + "尽量简洁地解释, 并提供答案。\n\n"
        + "- 对于复杂的问题：\n"
        + "使用以下逐步格式：\n"
        + "## 步骤 1: [简明描述]\n"
        + "[简要解释]\n"
        + "## 步骤 2: [简明描述]\n"
        + "[简要解释]\n\n"
        + "无论采用哪种方法，始终以以下内容结束：\n"
        + "最佳答案是 [答案字母]。[答题结束]\n"
        + "其中 [答案字母] 是 A、B、C 或 D 中的一个。\n\n"
        + "让我们一步一步思考。"
        + ("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" if not pretrain else "")
    )

    def _process_doc(doc: dict) -> dict:
        input_final_prompts = [input_final_prompt.format(
            Question=doc["question"],
            A=doc["A"],
            B=doc["B"],
            C=doc["C"],
            D=doc["D"],
        )]
        out_doc = {
            "problem": doc["question"],
            "gold": doc["answer"],
            "input_final_prompts": input_final_prompts,
        }
        return out_doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


def process_docs_cot_en(dataset: datasets.Dataset, pretrain=False) -> datasets.Dataset:
    # follow example of https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__mmlu__0_shot__cot__details?row=0
    # example: [ "<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\n\nQuestion: A 26-year-old woman is brought to the emergency department because of an 8-hour history of severe back and abdominal pain and mild but persistent vaginal bleeding. Ultrasonography of the abdomen shows a 2-cm ectopic pregnancy in the ampulla. The ampulla has ruptured into the surrounding tissue. Fluid from this rupture will most likely be found in which of the following locations?\nA. Lesser peritoneal cavity\nB. Mesometrium\nC. Pouch of Douglas\nD. Uterine cavity\n\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" ]
    input_final_prompt = (
        ("<|start_header_id|>user<|end_header_id|>\n\n" if not pretrain else "")
        + "Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n\n"
        + "Question: {Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
        + "- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n"
        + "- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n## Step 2: [Concise description]\n[Brief explanation]\n\n"
        + "Regardless of the approach, always conclude with:\nThe best answer is [the_answer_letter] [Done].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\n"
        + "Let's think step by step."
        + ("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" if not pretrain else "")
    )

    def _process_doc(doc: dict) -> dict:
        input_final_prompts = [input_final_prompt.format(
            Question=doc["question"],
            A=doc["A"],
            B=doc["B"],
            C=doc["C"],
            D=doc["D"],
        )]
        out_doc = {
            "problem": doc["question"],
            "gold": doc["answer"],
            "input_final_prompts": input_final_prompts,
        }
        return out_doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


def process_docs_default(dataset: datasets.Dataset, pretrain=False) -> datasets.Dataset:
    # follow example of https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__multilingual_mmlu_de__details?row=0
    # example: [ "<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: Glukose wird in die Muskelzelle transportiert:\nA. über Proteintransporter namens GLUT4.\nB. nur in Anwesenheit von Insulin.\nC. über Hexokinase.\nD. über Monocarbylsäure-Transporter.\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is A.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: Welche der folgenden Aussagen ist nicht wahr?\nA. Muskelglykogen wird enzymatisch zu Glucose-1-phosphat abgebaut\nB. Leistungsstarke Ausdauerläufer haben einen hohen Anteil an Typ-I-Fasern in ihrer Beinmuskulatur\nC. Leberglykogen ist wichtig für die Aufrechterhaltung der Blutglukosekonzentration\nD. Insulin fördert die Glukoseaufnahme durch alle Gewebe im Körper\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is D.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: Bei einem Gentest eines Neugeborenen wird eine seltene genetische Störung gefunden, die X-chromosomal-rezessiv vererbt wird. Welche der folgenden Aussagen trifft wahrscheinlich auf den Stammbaum dieser Störung zu?\nA. Alle Nachkommen auf der mütterlichen Seite werden die Störung haben.\nB. Frauen sind in dieser Familie etwa doppelt so stark betroffen wie Männer.\nC. Alle Töchter eines betroffenen Mannes werden betroffen sein.\nD. Es wird eine gleichmäßige Verteilung von Männern und Frauen geben, die betroffen sind.\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is C.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: Ein Gymnasiallehrer für Naturwissenschaften füllt eine 1-Liter-Flasche mit reinem Stickstoff und versiegelt den Deckel. Der Druck beträgt 1,70 atm und die Raumtemperatur 25°C. Welche zwei Variablen werden beide den Druck des Systems erhöhen, wenn alle anderen Variablen konstant gehalten werden?\nA. Steigende Temperatur, steigende Mole an Gas\nB. Temperatur erhöhen, Volumen erhöhen\nC. Abnehmendes Volumen, abnehmende Temperatur\nD. Abnehmende Gasmole, zunehmendes Volumen\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is A.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: Eine erwartete Nebenwirkung einer Kreatinsupplementierung ist:\nA. Muskelschwäche.\nB. Zunahme an Körpermasse.\nC. Muskelkrämpfe.\nD. Verlust von Elektrolyten.\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is B.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: Fettsäuren werden in die Mitochondrien transportiert, gebunden an:\nA. Thiokinase.\nB. Coenzym A (CoA).\nC. Acetyl-CoA.\nD. Carnitin.\nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe best answer is" ]
    input_final_prompt = (
        ("<|start_header_id|>user<|end_header_id|>\n\n" if not pretrain else "")
        + "Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n\n"
        + "Question: {Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nYour response should end with \"The best answer is [the_answer_letter] [Done]\" where the [the_answer_letter] is one of A, B, C or D."
        + ("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" if not pretrain else "")
        + "The best answer is "
    )

    def _process_doc(doc: dict) -> dict:
        input_final_prompts = [input_final_prompt.format(
            Question=doc["question"],
            A=doc["A"],
            B=doc["B"],
            C=doc["C"],
            D=doc["D"],
        )]
        out_doc = {
            "problem": doc["question"],
            "gold": doc["answer"],
            "input_final_prompts": input_final_prompts,
        }
        return out_doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


def process_docs_cot_zh_pretrain(dataset: datasets.Dataset, pretrain=True) -> datasets.Dataset:
    return process_docs_cot_zh(dataset, pretrain)


def process_docs_cot_en_pretrain(dataset: datasets.Dataset, pretrain=True) -> datasets.Dataset:
    return process_docs_cot_en(dataset, pretrain)


def process_docs_default_pretrain(dataset: datasets.Dataset, pretrain=True) -> datasets.Dataset:
    return process_docs_default(dataset, pretrain)
