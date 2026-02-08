# CRMArenaTesting

This is an assorted collection of random tests carried out on Salesforce CRMArena

To run the CRMArena agentic workflow use the following command

```bash
python run_crm_agent.py \
  --llm_model "Qwen/Qwen3-4B-Instruct-2507" \
  --task_name "lead_qualification"
  --llm_type "fine_tune"
  --train_data_file_path "/file/path/saved_fine_tune_model" \
  --number_of_tasks 10
  --output_save_path "output/crm_agent_test_results.csv"
```
Parameters:
- llm_model: The base LLM that was fine-tuned
- task_name: One of the CRMArena tasks, which includes 
    `'case_routing', 'knowledg_qa', 'lead_qualification', 'lead_routing', 'named_entity', 'activity_priority', 'monthly_trend_analysis', 'top_issue_identification', 'quote_approval'`
- llm_type: The LLM host/provider. `'ollama', 'bedrock', 'openai', 'fine_tune'`
- train_data_file_path: The path where the fine-tuned model is saved, if `fine_tune` is selected as `llm_type`
- number_of_tasks: The specific number of agentic tasks from the task_name category selected. CRMArena's tasks vary from 100 to 200 tasks in total.
- output_save_path: The path and csv filename for the agentic pipeline output.