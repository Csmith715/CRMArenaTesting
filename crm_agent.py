import sqlite3
from datasets import load_dataset
import os
from litellm import completion
from crm_testing_utils import soql_to_sql
import re
import time
import pandas as pd


# os.environ["AWS_BEARER_TOKEN_BEDROCK"] = ""
open_ai_key = os.environ["OPENAI_API_KEY"]
RESPOND_RE = re.compile(r"<respond>(.*?)</respond>", re.DOTALL)

class LiteLLMClient:
    def __init__(self, model: str, provider: str):
        self.model = model
        self.provider = provider
        self.api_base = None
        self.api_key = None
        # self.temperature = 0

    def generate(self, messages: list):
        if self.provider == "ollama":
            self.api_base = "http://localhost:11434"
        elif self.provider == "bedrock":
            self.model = f"bedrock/{self.model}"
        elif self.provider == "openai":
            self.api_key = open_ai_key
        res = completion(
            messages=messages,
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            )
        return res.choices[0].message.model_dump(), res.usage

class Environment:
    def __init__(self):
        self.con = sqlite3.connect("crm_dbs/crmarenapro_b2b_data.db")
        self.cur = self.con.cursor()
        self.dataset = load_dataset("Salesforce/CRMArenaPro", "CRMArenaPro", split="b2b")
        self.schema = {}
        _schema = load_dataset("Salesforce/CRMArenaPro", "b2b_schema", split="b2b_schema")
        for _s in _schema:
            schema_key = _s.get("object")
            schema_fields = _s.get("fields")
            self.schema[schema_key] = {k: v for k, v in schema_fields.items() if v is not None}

    def reset(self):
        return {
            "response": "",
            "error": False,
            "done": False,
            "steps": 0,
            "executions": [],          # {"query": ..., "result": [...]}
            "final_answer": None,      # <respond> ... </respond>
            "question": None,          # original query
            "sample_id": None,
            "task": "",
            "answer": ""
        }

    @property
    def is_complete(self):
        return False

    def step(self, state, action):
        if state["done"]:
            return state

        state["steps"] += 1

        if action["name"] == "execute":
            try:
                sql = soql_to_sql(action["content"])
                res = self.cur.execute(sql)
                out = []
                for row in res.fetchall():
                    out.append({k[0]: v for k, v in zip(res.description, row)})
                state["response"] = f"Salesforce instance output: {out}"
                state["executions"].append({"query": action["content"], "sql": sql, "result": out})
            except sqlite3.OperationalError as ex:
                state["error"] = True
                state["response"] = f"ERROR: {ex}"

        elif action["name"] == "respond":
            state["done"] = True
            state["final_answer"] = action["content"]
            state["response"] = 1

        else:
            state["error"] = True
            state["response"] = "Invalid action"

        return state

    def db_query(self, action) -> str:
        if action["name"] == "execute":
            try:
                sql = soql_to_sql(action["content"])
                # sql = action["content"]
                res = self.cur.execute(sql)
                out = []
                for row in res.fetchall():
                    out.append({k[0]: v for k, v in zip(res.description, row)})
                query_response = f"Salesforce instance output: {out}"
            except sqlite3.Error as ex:
                print(f'Query Error: {ex}')
                query_response = f"ERROR: {ex}"
        elif action["name"] == "respond":
            query_response = "Expert action has been completed or needs more information"
        else:
            query_response = "Invalid action"
        return query_response

class Agent:
    def __init__(self, client, should_halt, action_parser, prompt_generator):
        self.client = client
        self.should_halt = should_halt
        self.action_parser = action_parser
        self.prompt_generator = prompt_generator
        self.message_states = []
        self.all_states = []
        self.task_tokens = []
        self.completion_tokens = 0
        self.total_tokens = 0

    def run(self, env, task_type: str, pause, total_tasks: int = 10):
        i = 0
        filtered_dataset = [d for d in env.dataset if d["task"] == task_type]
        for idx, sample in enumerate(filtered_dataset):
            self.total_tokens = 0
            self.completion_tokens = 0
            state = env.reset()
            state["question"] = sample["query"].strip()
            state["response"] = state["question"]  # first user turn
            state["sample_id"] = sample.get("id", idx)
            state["task"] = sample["task"]
            state["answer"] = sample["answer"]

            messages = self.prompt_generator.generate_init_prompt(sample["metadata"])
            ongoing_states = []
            while not state["done"]:
                messages, state = self.act(messages, state, env)
                ongoing_states.append(state)
            self.task_tokens.append({'total_tokens': self.total_tokens, 'completion_tokens': self.completion_tokens})
            self.all_states.append(ongoing_states)
            self.message_states.append((messages, state, sample))
            i += 1
            time.sleep(pause)
            if i % 10 == 0:
                print(f"Task {i} completed.")
            if i >= total_tasks:
                break

    def act(self, messages, state, env):
        messages.append({"role": "user", "content": state["response"]})
        state["done"] |= self.should_halt(messages, state)
        if not state["done"]:
            try:
                response, usage = self.client.generate(messages)
                self.total_tokens += usage.total_tokens
                self.completion_tokens += usage.completion_tokens
            except Exception as ex:
                print(f"Action failed: {ex}")
                response = {"content": "No response"}
            content = response["content"]
            messages.append({"role": "assistant", "content": content})
            action = self.action_parser(content)
            state = env.step(state, action)
            if state["response"] == "Invalid action":
                state["response"] = self.prompt_generator.ACT_RULE_STRING
        return messages, state

    def collect_agent_outputs(self) -> pd.DataFrame:
        """
        Builds a dataframe with one row per sample you ran:
        columns: sample_id, question, final_answer, n_executes, error, steps, executions
        """
        rows = []
        for messages, state, sample in self.message_states:
            final_answer = state.get("final_answer") or extract_final_from_messages(messages)
            answer = state.get("answer", [])
            if None not in answer:
                str_answer = "\n".join(answer)
            else:
                str_answer = "None"
            rows.append({
                "sample_id": state.get("sample_id", 99999),
                "task": state.get("task", ""),
                "answer": str_answer,
                "question": state.get("question"),
                "final_answer": final_answer,
                "n_executes": len(state.get("executions", [])),
                "error": bool(state.get("error")),
                "steps": int(state.get("steps", 0)),
            })
        return pd.DataFrame(rows)

class PromptGenerator:
    ACT_RULE_STRING = """\
Invalid output format! Use the following format: <execute> a valid SOQL/SOSL query </execute> or <respond> response to user </respond>
"""

    ACT_PROMPT = """\
You are an expert in Salesforce and you have access to a {system}. 

# Instructions
- You will be provided a question, the system description, and relevant task context.
- Interact with the {system} to build Salesforce Object Query Language (SOQL) or Salesforce Object Search Language (SOSL) queries as appropriate, to help you answer the question.
- Salesforce Object Search Language (SOSL) can be used to construct text-based search queries against the search index.
- Your generation should always be an Action command and NOTHING ELSE. Generate only one Action command.
- DO NOT generate ANY system observation, you will receive this based on your Action command.
- If no record is found matching the requirements mentioned, just return 'None'.
- If the user's request is unclear or under-specified, use the respond action to ask for clarification before proceeding with queries or submitting an answer.

Here is a description of how to use this command:
## Action
- Can be 'execute' or 'submit'.
- execute, to execute SOQL that will return the observation from running the query on the {system}.
- submit, to return the final answer of the task to the user.
- Format: <execute> a valid SOQL query </execute> or <respond> user response </respond>

# Guidelines
- Execute SOQL/SOSL queries to understand the {system} that will help you find the answer to the question.
- When you are confident about the answer, submit it.
- Always end with a submit action containing ONLY the answer, NO full sentence or any explanation.
- If no record is found matching the requirements mentioned, just return 'None'.

# Example 1
Question: What is the total number of opportunities?
Output:
<execute> SELECT COUNT() FROM Opportunity </execute>
     (If the observation from the {system} 100, your next step can be)
<respond> 100 </respond> OR <respond> The total number of opportunities is 100 </respond>

# Example 2
Question: Look for the name Joe Smith in the name field of a lead and return the name and phone number.
Output:
<execute> FIND {{Joe Smith}} IN NAME FIELDS RETURNING Lead(Name, Phone) </execute>
    (If the observation from the {system} is [{{Joe Smith, 1234567890}}], your next step can be)
<respond> Joe Smith, 1234567890 </respond> OR <respond> The name is Joe Smith and the phone number is 1234567890 </respond>

# {system} description
{system_description}
"""

    SYSTEM_METADATA = """\
# {system} Metadata
{system_metadata}
"""

    SCHEMA_STRING = """\
The objects available in the Salesforce instance are:
{object_names}

## The fields available for the objects along with their descriptions and dependencies are:
{object_fields}
"""

    def __init__(self, schema):
        self.sys_desc = self.build_schema(schema)

    def build_schema(self, schema):
        object_description = {o: "\n".join([f"  - {k}: {v}" for k, v in f.items()]) for o, f in schema.items()}

        template = self.SCHEMA_STRING.format(
            object_names=", ".join(object_description.keys()),
            object_fields="\n".join(
                [f"{obj}\n{fields}" for obj, fields in object_description.items()]
            )
        )
        return template

    def generate_init_prompt(self, metadata):
        sys_prompt = self.ACT_PROMPT.format(system_description=self.sys_desc, system="Salesforce instance")
        if metadata["required"]:
            sys_prompt += self.SYSTEM_METADATA.format(system_metadata=metadata["required"], system="Salesforce instance")
        messages = [
            {"role": "system", "content": sys_prompt.strip()},
        ]
        return messages


def extract_final_from_messages(messages: list) -> str | None:
    # Look at the last assistant messages for a <respond> ... </respond>
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            m = RESPOND_RE.search(msg.get("content", ""))
            if m:
                return m.group(1).strip()
    return None

def halt_on_step(steps: int):
    def fn(messages, state) -> bool:
        return state["steps"] >= steps

    return fn

def message_action_parser(message: str) -> dict[str, str]:
    content = message.strip()

    resp = re.search(r'<execute>(.*?)</execute>', content, re.DOTALL)
    if resp:
        action = {"name": "execute", "content": resp.group(1).strip()}
        return action

    resp = re.search(r'<respond>(.*?)</respond>', content, re.DOTALL)
    if resp:
        action = {"name": "respond", "content": resp.group(1).strip()}
        return action
    return {"name": "null", "content": ""}

def capture_results(llm_model: str, task_name: str, llm_type: str, number_of_tasks: int = 200, pause_time: int = 1):
    ts_env = Environment()
    ts_llm_client = LiteLLMClient(llm_model, llm_type)
    ts_pg = PromptGenerator(ts_env.schema)
    ts_agent = Agent(ts_llm_client, halt_on_step(2), message_action_parser, ts_pg)
    ts_agent.run(ts_env, task_name, pause_time, number_of_tasks)

    return ts_agent.message_states
