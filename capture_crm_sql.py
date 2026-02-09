import argparse
from crm_agent import Environment, PromptGenerator, LiteLLMClient, Agent, halt_on_step, message_action_parser
import pandas as pd
from pathlib import Path

def make_dataframe(results: list):
    transformed_data = []
    for cap in results:
        messages = cap[0]
        executions = cap[1].get('executions', [])
        if executions:
            query = executions[0]
        else:
            query = {}
        transformed_pair = {
                "instruction": messages[0]['content'] if messages[0]['role'] == 'system' else "",
                "input": messages[1]['content'] if messages[1]['role'] == 'user' else "",
                "output": query.get('query', "")
            }
        transformed_data.append(transformed_pair)
    transformed_df = pd.DataFrame(transformed_data)
    return transformed_df

# ts = crm_agent.capture_results('ollama/deepseek-v3.1:671b-cloud', 'lead_qualification', 'ollama', 10)

def main():
    parser = argparse.ArgumentParser(description="CRM Arena Data Extraction")
    parser.add_argument("--llm_base_model", required=True, type=str)
    parser.add_argument("--crm_task_name",
                        required=True,
                        type=str,
                        choices=[
                            'case_routing',
                            'knowledge_qa',
                            'lead_qualification',
                            'lead_routing',
                            'named_entity',
                            'activity_priority',
                            'monthly_trend_analysis',
                            'top_issue_identification',
                            'quote_approval'
                        ])
    parser.add_argument("--llm_provider", required=True, type=str)
    parser.add_argument("--output_file_path", required=True, type=str)
    parser.add_argument("--number_of_steps", required=True, type=int, default=6)
    parser.add_argument("--number_of_tasks", required=False, type=int, default=200)
    # parser.add_argument("--pause_time", required=False, type=int)
    args = parser.parse_args()
    crm_env = Environment()
    crm_llm_client = LiteLLMClient(args.llm_base_model, args.llm_provider)
    crm_pg = PromptGenerator(crm_env.schema)
    crm_agent = Agent(crm_llm_client, halt_on_step(args.number_of_steps), message_action_parser, crm_pg)
    crm_agent.run(crm_env, args.crm_task_name, 0.5, args.number_of_tasks)

    final_df = make_dataframe(crm_agent.message_states)
    output_path = Path(args.output_file_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_path.parent}: {e}")
        return
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
