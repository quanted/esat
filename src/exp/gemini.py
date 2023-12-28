import os
import sys
import pathlib
import textwrap
import inspect
import time
import google.api_core.exceptions
import numpy as np
import json
import logging

import google.generativeai as genai

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data.datahandler import DataHandler
from src.model.nmf import NMF
from src.model.batch_nmf import BatchNMF
from src.model.ls_nmf import LSNMF
from src.model.ws_nmf import WSNMF

logger = logging.getLogger("NMF")
logger.setLevel(logging.DEBUG)


class GeminiModels:

    def __init__(self, output_dir: str = None, algorithm: str = "ls-nmf"):
        self.base_code = None
        self.base_models = None
        self.base_Qrobust = None
        self.algorithm = algorithm if algorithm in ("ls-nmf", "ws-nmf") else "ls-nmf"
        self.output_dir = f"D:\\\\projects\\nmf_py\\funsearch\\algorithms\\{self.algorithm}" if output_dir is None else output_dir
        self.get_base()
        self.train_success = False
        self.train_message = ""

    def get_base(self):
        if self.algorithm == "ws-nmf":
            def_index = 7
            start_i = 1
            code_index = 34
            base_alg = inspect.getsourcelines(WSNMF.update)
        else:
            def_index = 7
            start_i = 0
            code_index = 30
            base_alg = inspect.getsourcelines(LSNMF.update)
        base_code = []
        for i in range(start_i, len(base_alg[0])):
             if i < def_index or i >= code_index:
                code_line = base_alg[0][i]
                if "def" in code_line or "staticmethod" in code_line:
                    code_line = textwrap.dedent(code_line)
                    if "self" not in code_line and "staticmethod" not in code_line:
                        code_line = code_line.replace("\n", "self,\n")
                base_code.append(code_line)
                if "return" in code_line:
                    break
        base_code = "".join(base_code)
        # base_code = base_code[0: -1]
        self.base_code = base_code

    def run_algorithm(self, new_algorithm):
        input_file = os.path.join("data", "Dataset-BatonRouge-con.csv")
        uncertainty_file = os.path.join("data", "Dataset-BatonRouge-unc.csv")

        data_handler = DataHandler(
            input_path=input_file,
            uncertainty_path=uncertainty_file,
            index_col='Date'
        )
        V, U = data_handler.get_data()
        nmf_models = BatchNMF(V=V, U=U, factors=6, models=10, method='ls-nmf', parallel=False, verbose=True)
        nmf_models.update_step = new_algorithm
        self.train_success, self.train_message = nmf_models.train(min_limit=4)
        return nmf_models

    def aggregate_results(self, models: BatchNMF):
        runtime = models.runtime / models.models
        qtrue = []
        qrobust = []
        for model in models.results:
            if model is None:
                continue
            qtrue.append(model.Qtrue)
            qrobust.append(model.Qrobust)
        return {"runtime": (round(runtime, 2), round(models.runtime)),
                "Q(true)": (round(np.min(qtrue), 2), round(np.mean(qtrue), 2), round(np.max(qtrue), 2)),
                "Q(robust)": (round(np.min(qrobust), 2), round(np.mean(qrobust), 2), round(np.max(qrobust), 2))}

    def update_summary(self, name, alg_summary, code_path):
        summary_file = os.path.join(self.output_dir, "summary.json")
        alg_summary["code_path"] = code_path
        if os.path.exists(summary_file):
            alg_key = name
            summary = {name: alg_summary}
            with open(summary_file, 'r') as sum_file:
                existing_summary = json.load(sum_file)
                if alg_key not in existing_summary.keys():
                    existing_summary[alg_key] = summary[alg_key]
                alg_summary = existing_summary
        else:
            summary = {name: alg_summary}
            alg_summary = summary
        with open(summary_file, "w") as sum_file:
            json.dump(alg_summary, sum_file)
        logger.info(f"Updated summary for algorithm {name}.")
        logger.info(alg_summary)

    def save_algorithm(self, name, code):
        code_file = os.path.join(self.output_dir, f"gemini-{name}.txt")
        with open(code_file, "w") as cfile:
            for cline in code:
                cfile.write(cline)
        logger.info(f"Saved algorithm {name}.")
        return code_file

    def save_history(self, name, chat_history: list):
        history_file = os.path.join(self.output_dir, f"gemini-{name}-history.txt")
        with open(history_file, 'w') as hfile:
            chat_history = str(chat_history)
            hfile.write(chat_history)
        logger.info(f"Saved history for algorithm {name}.")

    def select_algorithm(self):
        summary_file = os.path.join(self.output_dir, "summary.json")
        index = 1
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as sum_file:
                existing_models = json.load(sum_file)
                model_keys = list(existing_models.keys())
                random_key = np.random.choice(model_keys, 1)[0]
                index = len(model_keys)
                while index in model_keys:
                    index += 1
                code_path = existing_models[random_key]["code_path"]
            with open(code_path, 'r') as code_file:
                alg_code = code_file.read()
            return index, alg_code
        return index, None

    def start_connection(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_GEMINI_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat(history=[])
        return chat

    def generate_prompt(self, code, status: int, message: str = None) -> list:
        # The initial prompt for a new chat session and new algorithm. Specify algorithm requirements and the code.
        prompt = ""
        semi_nmf = "" if self.algorithm == "ls-nmf" else "Semi-"
        if status == 0:
            prompt_list = [
                f"Can you create a better version of the {semi_nmf}NMF algorithm than this code? ",
                f"Can you come up with a more creative {semi_nmf}NMF algorithm than this? ",
                f"Make the most ingenious version of the {semi_nmf}NMF algorithm, even better than this! ",
                f"Can you make this {semi_nmf}NMF algorithm more optimized? ",
                f"Write me code for {semi_nmf}the best possible NMF algorithm. "
            ]
            code_requirements = "Function must be called update and take self as the first argument. The input " \
                                "parameters are data V: (NxM), data weights We: (NxM), factor contribution W: (Nxk) " \
                                "and factor profile H: (kxM) and must remain the same. Code must be in Python and " \
                                "only use numpy. If new parameters are added, make sure to set default values. Only " \
                                "respond with the new code for all future prompts in this session."
            if self.algorithm == "ws-nmf":
                code_requirements += " The Semi-NMF algorithm allows for negative values in the input data matrix V " \
                                     "and the factor contribution matrix W. The weights and factor profile are all " \
                                     "positive matrices."
            random_prompt_i = np.random.choice(range(len(prompt_list)), 1)[0]
            prompt = prompt_list[random_prompt_i] + code_requirements + f"Code: {code}"
        elif status == 1:
            # code produces an error, found in message. Ask for a corrected algorithm.
            if "not defined" in message:
                prompt = "A parameter doesn't have a default value, and if can't have a default value it needs to be " \
                         "removed so the code has the same signature as the original or any new function parameters have" \
                         f"default values. {message}"
            else:
                prompt_list = [
                    f"The code you gave me generated an error: {message}. Can you give me code that fixes this error?",
                    f"An following error occurred with the code you gave me: {message}. Can you update the algorithm to "
                    f"correct this issue?",
                    f"A problem occurred while running the algorithm you gave me. Error: {message}. Can you fix the code?",
                    f"Ah oh! I got this error when running the code {message}. Can you fix the code to correct the mistake?",
                    f"That code is broken. Error: {message}. Can you fix the issue in the code?"
                ]
                random_prompt_i = np.random.choice(range(len(prompt_list)), 1)[0]
                prompt = prompt_list[random_prompt_i]
        elif status == 2:
            # produces the same result as the initial code.
            prompt_list = [
                "That algorithm produced the same results as the original. Can you modify the code to make it even better?",
                "The new code only replicated the original, get creative and make code for a new better NMF algorithm.",
                "Can you update the code by reworking the algorithm to make it even better?",
                "Create code that is completely new that no one has made before.",
                "What about creating a new and improved version of that code?",
                "Do the impossible and create the ultimate version of that code.",
                "Tap into your creative genius and make the best version of this code possible.",
                "How would Isaac Asimov make this code better?"
            ]
            random_prompt_i = np.random.choice(range(len(prompt_list)), 1)[0]
            prompt = prompt_list[random_prompt_i]
        elif status == 3:
            # produces results that are at least 5x worse than the base model
            prompt_list = [
                f"This algorithm produces a loss value of {message} compared to {self.base_Qrobust}. Can you make the code better?",
                f"The algorithm doesn't perform very well. The loss is a lot more than the original code. Can you fix the code?",
                f"We need an improved algorithm, the loss is {round(float(message) / self.base_Qrobust, 2)} times as "
                f"must as the original. Can you give me better code?",
                f"Lets start brainstorming on how to make the algorithm more efficient. It's loss is significantly more "
                f"than the original. Can you give me your best code updates?",
                f"What if further optimize the code to reduce our loss value?",
                f"The loss value is not really good. Can I get code that does better?",
                f"There is a difference of {float(message) - self.base_Qrobust} in the loss value, can I get code that "
                f"reduces that difference?"
            ]
            random_prompt_i = np.random.choice(range(len(prompt_list)), 1)[0]
            prompt = prompt_list[random_prompt_i]
        elif status == 4:
            # The code results in loss value of NAN
            prompt_list = [
                "That code causes the loss value to be a NAN. Can you update the code to fix the problem?",
                "I'm getting NANs for the loss value calculation. We need a code update to fix the issue.",
                "Can you update the code to correct for the loss value calculating NAN instead of a float?",
                "We have a problem with that code. It's causing the loss value to be NAN, can you correct the code to "
                "fix this?",
                "When I calculate the loss value, I'm getting NANs. I need code that will correct this issue.",
                "That code you gave me is no good. I'm getting NANs for the loss calculation. Can you update the code "
                "to correct the problem?"
            ]
            random_prompt_i = np.random.choice(range(len(prompt_list)), 1)[0]
            prompt = prompt_list[random_prompt_i]
        elif status == 5:
            # The training time exceeded the training runtime limit
            prompt_list = [
                "The algorithm takes too much time to run. Can you optimize the code to decrease runtime?",
                "We need the algorithm to run more efficiently, it's currently taking too much time to train. Can you "
                "make the algorithm quicker?",
                "Can you make the algorithm run faster?",
                "The algorithm is too slow, can you make it even faster?",
            ]
            random_prompt_i = np.random.choice(range(len(prompt_list)), 1)[0]
            prompt = prompt_list[random_prompt_i]
        return prompt

    def submit(self, prompt, session) -> str:
        response = session.send_message(prompt)
        return response.text

    def parse_response(self, new_alg_str):
        alg_list = (new_alg_str).split("\n")
        new_alg = []
        copy = False
        for code_line in alg_list:
            if "def" in code_line:
                copy = True
            if copy:
                new_alg.append(code_line)
            if "return" in code_line:
                break
        return "\n".join(new_alg)

    def run_base(self):
        logger.info("Starting base model run")
        self.base_models = self.run_algorithm(new_algorithm=self.base_code)
        self.base_Qrobust = self.base_models.results[self.base_models.best_model].Qrobust

        base_results = self.aggregate_results(models=self.base_models)
        code_path = self.save_algorithm("0", self.base_code)
        self.update_summary(0, base_results, code_path)
        logger.info("Completed running base model and updating results.")

    def execute(self, max_iterations: int = 500):
        gemini_search = True
        search_i = 1
        added_algs = 0
        best_alg = 0
        best_q = self.base_Qrobust
        logger.info("Starting Gemini Algorithm Search")
        # the previous model. If an error provide the error and ask for a better model or correct model.
        # Steps:
        # 1. Select a random algorithm from the collection.
        # 2. Generate prompt that asks for a better version of the algorithm (status=0)
        # 3. Based upon the results of the new algorithm:
        #    a. An error occurred. Ask for a corrected algorithm, providing the error (status=1, message=error)
        #    b. Produces the same results as the base model. Ask for a more better or more creative algorithm. (status=2)
        #    c. Produces results that are 5 times worse than the base model, ask if it can do better. (status=3, message=Qrobust)
        #    d. Produces NAN loss values. Ask for a corrected algorithm, stating that the prior one produced NAN in the loss value. (status=4)
        #    e. The max number of attempts on this algorithm have been reached. Don't save and restart the process.
        #    f. Otherwise save the algorithm to the collection.
        # For 3a-3d, have multiple possible prompts to work with.

        while gemini_search:
            logger.info(f"Starting new algorithm selection... {search_i}/{max_iterations}")
            if search_i >= max_iterations:
                gemini_search = False
            search_i += 1
            index, alg = self.select_algorithm()                        # Step 1
            if alg is None:
                alg = self.base_code
                index = 1
            alg_session = True
            max_session = 10
            chat_session = self.start_connection()
            gemini_prompt = self.generate_prompt(code=alg, status=0)      # Step 2
            try:
                gemini_response = self.submit(prompt=gemini_prompt, session=chat_session)
            except google.api_core.exceptions.InternalServerError as e:
                logger.error(f"Google API Server Error: {e}")
                logger.info(f"Waiting 60 seconds before attempting a new connection to Google API server.")
                time.sleep(60)      # Pause for 60 seconds
                continue
            gemini_code = self.parse_response(new_alg_str=gemini_response)
            while alg_session:
                max_session -= 1
                if max_session < 0:
                    alg_session = False
                message = None
                status = 0
                new_qrobust = float("inf")
                logger.info(f"Testing algorithm. Session prompts remaining: {max_session}")
                try:
                    new_results = self.run_algorithm(gemini_code)
                    new_qrobust = new_results.results[new_results.best_model].Qrobust
                except Exception as e:
                    message = str(e)
                    status = 1                                                  # Step 3a
                    logger.info(f"Algorithm failed due to error {e}")
                if self.train_success:
                    if status == 0:
                        summary = self.aggregate_results(models=new_results)
                        save = False
                        if np.isnan(new_qrobust):                                   # Step 3d
                            status = 4
                        elif round(new_qrobust) == round(self.base_Qrobust):        # Step 3b
                            status = 2
                        elif round(new_qrobust) > 5 * round(self.base_Qrobust):     # Step 3c
                            status = 3
                            message = round(new_qrobust)
                        elif round(new_qrobust) < round(self.base_Qrobust):
                            alg_session = False
                            alg_file = self.save_algorithm(index, gemini_code)
                            self.update_summary(index, summary, alg_file)
                            self.save_history(name=index, chat_history=list(chat_session.history))
                            status = -1
                        else:
                            save = True
                            status = -1
                        if max_session < 0:                                         # Step 3e
                            alg_session = False
                            if save:
                                alg_file = self.save_algorithm(index, gemini_code)
                                self.update_summary(index, summary, alg_file)
                                self.save_history(name=index, chat_history=list(chat_session.history))
                else:
                    status = 5
                    message = self.train_message
                if alg_session:
                    logger.info(f"Prompting for code update for status: {status}, message: {message}, remaining "
                                f"session prompts: {max_session}")
                    try:
                        gemini_prompt = self.generate_prompt(code=gemini_code, status=status, message=message)
                        gemini_response = self.submit(prompt=gemini_prompt, session=chat_session)
                        gemini_code = self.parse_response(new_alg_str=gemini_response)
                    except IndexError as ex:
                        logger.error(f"Gemini chat error due to {ex}")
                        logger.info("Ending current session.")
                        alg_session = False


if __name__ == "__main__":
    algorithm = "ws-nmf"
    gemini = GeminiModels(algorithm=algorithm)
    gemini.run_base()
    gemini.execute()
