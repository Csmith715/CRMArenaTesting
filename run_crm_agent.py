import argparse
from crm_agent import Environment, PromptGenerator, LiteLLMClient, Agent
import re

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

def main():
    parser = argparse.ArgumentParser(description="CRM Arena Agent Testing")
    parser.add_argument("--llm_model", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--llm_type", required=True, type=str)
    parser.add_argument("--number_of_tasks", required=True, type=int)
    parser.add_argument("--output_save_path", required=True, type=int)
    parser.add_argument("--fine_tune_path", required=False, type=str)
    parser.add_argument("--pause_time", required=False, type=int)

    args = parser.parse_args()

    ts_env = Environment()
    ts_llm_client = LiteLLMClient(args.llm_model, args.llm_type, args.fine_tune_path)
    ts_pg = PromptGenerator(ts_env.schema)
    ts_agent = Agent(ts_llm_client, halt_on_step(6), message_action_parser, ts_pg)
    ts_agent.run(ts_env, args.task_name, args.pause_time, args.number_of_tasks)
    merged_df = ts_agent.collect_agent_outputs()
    merged_df.to_csv(args.output_save_path, index=False)


if __name__ == "__main__":
    main()
