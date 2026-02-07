from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
import copy


######################################################
# Tree of Thought ####################################
######################################################

@dataclass
class ThoughtNode:
    # Nodes in the tree of thoughts.
    content: str
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    evaluation_score: float = 0.0
    depth: int = 0
    is_terminal: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = []

    # Adding child node
    def add_child(self, child: 'ThoughtNode') -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    # Root node path
    def get_path(self) -> List[str]:
        path = []
        current = self
        while current:
            path.insert(0, current.content)
            current = current.parent
        return path


def json_extractor(llm_response):
    # Extract JSON from response
    if "```json" in llm_response:
        json_str = llm_response.split("```json")[1].split("```")[0].strip()
    elif "```" in llm_response:
        json_str = llm_response.split("```")[1].split("```")[0].strip()
    else:
        json_str = llm_response
    return json_str


class TreeOfThoughts:
    """Tree of Thoughts implementation."""

    def __init__(self, llm, max_depth: int = 4, branching_factor: int = 3):
        self.llm = llm
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.root = None
        self.evaluation_criteria = {
            "logic": "Is the reasoning logically sound?",
            "feasibility": "Is this approach feasible?",
            "completeness": "Does this address the problem completely?",
            "creativity": "Is this a creative or novel approach?"
        }
        self.tree_system_user = {
            "role": "system",
            "content": "You are an expert at debugging irregular Salesforce Object Query Language (SOQL) or Salesforce Object Search Language (SOSL) queries."
        }

    def generate_thoughts(self, problem: str, current_path: List[str], num_thoughts: int = 3) -> List[str]:
        """Generate multiple thought branches from current state."""
        path_context = " -> ".join(current_path) if current_path else "Starting point"

        prompt = f"""
        Problem: {problem}

        Current reasoning path: {path_context}

        Generate {num_thoughts} different next steps or approaches to continue solving this problem.
        Each thought should be:
        1. A specific, actionable step
        2. Logically connected to the current path
        3. Different from the other thoughts
        4. One sentence or short phrase

        Format as a JSON array of strings:
        ["thought1", "thought2", "thought3"]
        """

        # response = self.llm.generate(prompt).content
        response = self.llm.generate([self.tree_system_user, {"role": "user", "content": prompt}])
        response = response['content']

        try:
            jstr = json_extractor(response)
            thoughts = json.loads(jstr)
            # Self imposed limit
            return thoughts[:num_thoughts]
        except json.JSONDecodeError:
            # Generate simple thoughts (as fallback if needed)
            return [
                       f"Approach 1: Continue with current path",
                       f"Alternative 2: Try different method",
                       f"Backup 3: Reconsider previous step"
                   ][:num_thoughts]

    def evaluate_thought(self, problem: str, thought_path: List[str]) -> Dict[str, Any]:
        """Evaluate a thought path using multiple criteria."""
        path_text = " -> ".join(thought_path)

        prompt = f"""
        Problem: {problem}

        Thought path: {path_text}

        Evaluate this reasoning path on the following criteria (1-10 scale):
        1. Logic: Is the reasoning logically sound?
        2. Feasibility: Is this approach feasible?
        3. Completeness: Does this address the problem completely?
        4. Creativity: Is this a creative or novel approach?

        Also provide:
        - Overall assessment (1-10)
        - Strengths of this approach
        - Weaknesses or concerns
        - Whether this path should be continued or pruned

        Respond with JSON:
        {{
            "logic": 8,
            "feasibility": 7,
            "completeness": 6,
            "creativity": 9,
            "overall": 7.5,
            "strengths": "Good logical flow, creative approach",
            "weaknesses": "May be too complex",
            "continue": true
        }}
        """

        # response = self.llm.generate(prompt).content
        response = self.llm.generate([self.tree_system_user, {"role": "user", "content": prompt}])
        response = response['content']

        try:
            jstr = json_extractor(response)
            evaluation = json.loads(jstr)
            return evaluation
        except json.JSONDecodeError:
            # Fallback evaluation
            return {
                "logic": 7,
                "feasibility": 7,
                "completeness": 7,
                "creativity": 7,
                "overall": 7.0,
                "strengths": "Reasonable approach",
                "weaknesses": "Standard approach",
                "continue": True
            }

    # Determine if thought path is a complete solution
    def is_terminal(self, problem: str, thought_path: List[str]) -> bool:
        path_text = " -> ".join(thought_path)

        prompt = f"""
        Problem: {problem}

        Thought path: {path_text}

        Does this path represent a complete solution to the problem?
        A complete solution should:
        1. Address all aspects of the problem
        2. Provide actionable steps or conclusions
        3. Be implementable or verifiable

        Respond with JSON:
        {{
            "is_complete": true/false,
            "reason": "explanation of why it is or isn't complete"
        }}
        """

        # response = self.llm.generate(prompt).content
        response = self.llm.generate([self.tree_system_user, {"role": "user", "content": prompt}])
        response = response['content']

        try:
            jstr = json_extractor(response)
            result = json.loads(jstr)
            return result.get("is_complete", False)
        except json.JSONDecodeError:
            # Fallback: consider complete if path is long enough
            return len(thought_path) >= 3

    def build_tree(self, problem: str) -> ThoughtNode:
        """Used to build a tree of thoughts"""
        # print(f"🌳 Building Tree of Thoughts for: {problem}")
        # print("=" * 50)

        # Initialize root
        self.root = ThoughtNode(content="Start solving the problem")

        # BFS to build tree
        queue = [self.root]
        level = 0

        while queue and level < self.max_depth:
            level += 1
            # print(f"\n--- Level {level} ---")

            next_queue = []

            for node in queue:
                current_path = node.get_path()

                # Generate thoughts for this node
                thoughts = self.generate_thoughts(problem, current_path, self.branching_factor)
                # print(f"Node: {node.content}")
                # print(f"Generated {len(thoughts)} thoughts")

                for thought_content in thoughts:
                    child = ThoughtNode(content=thought_content)
                    node.add_child(child)

                    # Evaluate the new path
                    new_path = child.get_path()
                    evaluation = self.evaluate_thought(problem, new_path)
                    child.evaluation_score = evaluation["overall"]

                    # Check if terminal
                    child.is_terminal = self.is_terminal(problem, new_path)

                    # print(f"  - {thought_content} (score: {child.evaluation_score:.1f})")

                    # Add to next level if not terminal and score is good
                    # Score is very arbitrary at the moment
                    if not child.is_terminal and child.evaluation_score >= 6.0:
                        next_queue.append(child)

            queue = next_queue

        return self.root

    def find_best_solution(self, problem: str) -> Dict[str, Any]:
        """
        Find the best solution by exploring the tree.
        Problems are passed in as a list of dictionaries (prompts).
        """
        root = self.build_tree(problem)

        # Find all terminal nodes
        terminal_nodes = self._find_terminal_nodes(root)

        if not terminal_nodes:
            return {
                "solution": "No complete solutions found",
                "best_path": [],
                "score": 0.0,
                "explanation": "No terminal nodes reached"
            }

        # Finding the best terminal node
        best_node = max(terminal_nodes, key=lambda n: n.evaluation_score)
        best_path = best_node.get_path()

        # Generate final solution
        solution = self._generate_final_solution(problem, best_path)

        return {
            "solution": solution,
            "best_path": best_path,
            "score": best_node.evaluation_score,
            "explanation": f"Best path found with score {best_node.evaluation_score:.1f}"
        }

    def _find_terminal_nodes(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Find all terminal nodes in the tree."""
        terminals = []

        if node.is_terminal:
            terminals.append(node)

        for child in node.children:
            terminals.extend(self._find_terminal_nodes(child))

        return terminals

    def _generate_final_solution(self, problem: str, path: List[str]) -> str:
        """Generate final solution from the best path."""
        path_text = " -> ".join(path)

        prompt = f"""
        Problem: {problem}

        Best reasoning path: {path_text}

        Provide a corrected and improved solution. If the initial solution is satisfactory, DO NOT CHANGE IT.
        """

        # return self.llm.generate(prompt).content
        # return self.llm.generate([self.tree_system_user, {"role": "user", "content": prompt}]).content
        response = self.llm.generate([self.tree_system_user, {"role": "user", "content": prompt}])
        return response['content']

    def print_tree(self, node: ThoughtNode = None, depth: int = 0) -> None:
        """Print the tree structure."""
        if node is None:
            node = self.root

        if node is None:
            return

        indent = "  " * depth
        print(f"{indent}{node.content} (score: {node.evaluation_score:.1f})")

        for child in node.children:
            self.print_tree(child, depth + 1)


######################################################
# SELF CORRECTION ####################################
######################################################

def self_correction_reasoning(problem: list, possible_solution: str, llm_client):
    correction_prompt = f""" 
    Suggested solution: {possible_solution}

    Review your solution and identify any errors or improvements:
    1. Is the solution correct?
    2. Are there any logical errors?
    3. Can the solution be improved?
    4. What would be a better approach?

    Provide a corrected and improved solution strictly following the guidelines and proper output format initially defined. If the initial solution is satisfactory, 
    DO NOT CHANGE IT."""
    problem.append({"role": "user", "content": correction_prompt})
    try:
        # corrected_solution, usage = llm_client.generate(problem)
        corrected_solution = llm_client.generate(problem)
        # _total_tokens = usage.total_tokens
        # _completion_tokens = usage.completion_tokens
    except Exception as exception:
        print(f"Self correction action failed: {exception}")
        corrected_solution = {"content": possible_solution}
        _total_tokens, _completion_tokens = 0, 0
    # return corrected_solution['content'], _total_tokens, _completion_tokens
    return corrected_solution['content']


RESPOND_RE = re.compile(r"<respond>(.*?)</respond>", re.DOTALL)
EXEC_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL)


def build_self_reflection_prompt(situation: str, expert_actions_responses: list, llm_client) -> str:
    base_sys_prompt = f'''
    Self-Reflection Prompt Template

    You will be presented with a situation where you need to choose between multiple possible actions.
    Your task is to analyze the situation and decide which expert action to take.
    Situation Description: 
    {situation}
    '''
    action_prompts = []
    for i, (expert_action, db_response) in enumerate(expert_actions_responses, start=1):
        action_prompt = f'''
        Action {i}.
        • Expert Action: {expert_action}
        • Action Outcome: {db_response}
        -----------

        '''
        action_prompts.append(action_prompt)
    prompt_suffix = """Guidelines:
    - Stay strictly within the provided information.
    - Avoid meta-commentary about being an AI.
    - Use natural, step-by-step reasoning.
    - Focus on logical decision-making.
    Output: Return the specific Expert Action that reflects that best response to the situation presented. Select ONLY one of:
        1) <execute>SOQL or SOSL</execute>
        2) <respond>final short answer or clarifying question</respond>"""

    final_prompt = 'Select from the following Action/Outcome pairs the best approach:\n' + '\n'.join(action_prompts) + prompt_suffix

    try:
        selected_action = llm_client.generate([
            {"role": "system", "content": base_sys_prompt},
            {"role": "user", "content": final_prompt}
        ])
        derived_action = selected_action['content']
    except Exception as exception:
        print(f"Self reflection prompt failed: {exception}")
        derived_action = [e[0] for e in expert_actions_responses][0]

    return derived_action


def _json_extractor(text: str) -> str:
    # safer JSON block grab
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    return m.group(1) if m else text


######################################################
# Tree of Thought (Version 2) ########################
######################################################

class ActionToT:
    """
    Proposes ACT actions (<execute>... or <respond>...) using a tree/beam and
    picks the best by combining environment feedback + optional LLM critique.
    """

    def __init__(self, llm_client, max_depth=3, branching_factor=3, use_critic=True):
        self.llm = llm_client
        self.max_depth = max_depth
        self.k = branching_factor
        self.use_critic = use_critic

    def propose_actions(self, messages: List[Dict[str, str]], k: int) -> List[str]:
        """
        Ask the LLM for k candidate ACT actions. Returns raw strings containing
        <execute>...</execute> or <respond>...</respond>.
        """
        prompt = f"""You are an expert Salesforce assistant following this strict action format:
                    
                    - Generate ONLY one of:
                      1) <execute>SOQL or SOSL</execute>
                      2) <respond>final short answer or clarifying question</respond>
                    
                    Generate exactly {k} diverse candidate actions that would be the NEXT step,
                    given the ongoing conversation.
                    
                    Return them as JSON list of strings, e.g.:
                    [
                      "<execute> SELECT COUNT() FROM Opportunity </execute>",
                      "<respond> None found. Can you clarify the date range? </respond>"
                    ]
                    """
        resp = self.llm.generate(messages + [{"role": "user", "content": prompt}])["content"]
        try:
            j = json.loads(_json_extractor(resp))
            return [s.strip() for s in j][:k]
        except Exception as e:
            print(e)
            # very conservative fallback
            return ["<respond> Can you clarify? </respond>"]

    def score_by_env(self, env, state, candidate: str) -> Dict[str, Any]:
        """
        Try the candidate in a temp copy of state and score:
          +2 if it parses as a valid action
          +2 if execute succeeds (no error)
          + (#rows * row_weight) up to a cap
          - penalty if error
        """
        # parse
        name = "null"
        content = ""
        m = EXEC_RE.search(candidate)
        if m:
            name, content = "execute", m.group(1).strip()
        else:
            m = RESPOND_RE.search(candidate)
            if m:
                name, content = "respond", m.group(1).strip()

        if name == "null":
            return {"score": -5, "name": "null", "content": candidate, "rows": 0}

        # simulate one step
        shadow = dict(state)
        shadow["done"] = False
        shadow["error"] = False
        shadow["response"] = state["response"]

        shadow = env.step(shadow, {"name": name, "content": content})

        score = 0
        score += 2  # parsed
        rows = 0
        if name == "execute":
            if shadow.get("error"):
                score -= 3
            else:
                # rows heuristic
                executions = shadow.get("executions", [])
                rows = len(executions[-1]["result"]) if executions else 0
                # cap influence
                score += min(rows, 10) * 0.3
        else:
            # <respond> ends episode with small base score
            score += 1

        return {"score": score, "name": name, "content": content, "rows": rows, "shadow": shadow}

    def _critic_score(self, messages: List[Dict[str, str]], candidate: str) -> float:
        if not self.use_critic:
            return 0.0
        prompt = f"""Rate 0–10 how promising this next action is for the stated task.

                    Candidate:
                    {candidate}
                    
                    Return ONLY a JSON: {{"score": <float>, "reason":"..."}}"""
        resp = self.llm.generate(messages + [{"role": "user", "content": prompt}])["content"]
        try:
            j = json.loads(_json_extractor(resp))
            return float(j.get("score", 0))
        except Exception as exc:
            print(exc)
            return 0.0

    # def choose_next_action(self, messages: List[Dict[str, str]], env, state) -> str:
    #     """
    #     One ToT step:
    #       1) propose k actions,
    #       2) env-score each (and add optional critic),
    #       3) return the best action string.
    #     """
    #     cands = self.propose_actions(messages, self.k)
    #     scored = []
    #     for c in cands:
    #         env_score = self._score_by_env(env, state, c)
    #         crit = self._critic_score(messages, c)
    #         total = env_score["score"] + 0.4 * crit  # weight critic modestly
    #         env_score["total"] = total
    #         scored.append(env_score)
    #
    #     best = max(scored, key=lambda x: x["total"])
    #     # **Important**: return the *original* ACT string, not the parsed bits
    #     if best["name"] == "execute":
    #         return f"<execute> {best['content']} </execute>"
    #     elif best["name"] == "respond":
    #         return f"<respond> {best['content']} </respond>"
    #     else:
    #         return "<respond> None </respond>"

    def apply_action_on_shadow(
            self,
            env,
            state_shadow: dict,
            messages_shadow: list,
            act_str: str
    ) -> Tuple[dict, list, float]:
        """
        Apply one candidate action on shadow copies. Returns:
          - new_state_shadow
          - new_messages_shadow
          - incremental_score (env + critic)
        """
        # Parse and score with env
        env_score = self.score_by_env(env, state_shadow, act_str)
        incr = env_score["score"]

        # Optionally add critic score (use same messages context)
        critic = self._critic_score(messages_shadow, act_str)
        incr_total = incr + 0.4 * critic

        # Build next messages context:
        #   append assistant act, then append the env observation as a user message
        new_messages = list(messages_shadow)
        new_messages.append({"role": "assistant", "content": act_str})

        # state_shadow already contains the env step result in env_score["shadow"]
        new_state = env_score.get("shadow", state_shadow)

        # feed the latest observation back as the next user turn (if we didn't end)
        if not new_state.get("done"):
            new_messages.append({"role": "user", "content": new_state.get("response", "")})

        return new_state, new_messages, incr_total

    def choose_next_action(
            self,
            messages: list,
            env,
            state: dict,
            rollout_depth: int = 2,
            beam_width: int = 3
    ) -> str:
        """
        Beam search over action sequences up to `rollout_depth`, branching `beam_width`.
        Returns the FIRST action of the best sequence.
        Complexity ~ O(beam_width ** rollout_depth).
        """

        # Beam items: (total_score, [act_str1, act_str2, ...], state_shadow, messages_shadow)
        # Start from current context
        init_state = copy.deepcopy(state)
        init_messages = list(messages)

        # At the very start of a turn, ensure the format reminder is present
        init_messages.append({"role": "user", "content": "Remember: output ONLY one action: <execute> SOQL/SOSL </execute> OR <respond> ... </respond>"})

        beam = [(0.0, [], init_state, init_messages)]

        for depth in range(rollout_depth):
            new_beam = []
            for total_score, acts_so_far, state_shadow, messages_shadow in beam:
                # If already terminal, keep as-is (don’t expand)
                if state_shadow.get("done"):
                    new_beam.append((total_score, acts_so_far, state_shadow, messages_shadow))
                    continue

                # Propose k candidates for NEXT action
                cands = self.propose_actions(messages_shadow, k=beam_width)

                for act_str in cands:
                    # Use deep copy so sibling branches don’t interfere
                    st_copy = copy.deepcopy(state_shadow)
                    msg_copy = list(messages_shadow)

                    new_state, new_msgs, incr = self.apply_action_on_shadow(env, st_copy, msg_copy, act_str)
                    new_beam.append((total_score + incr, acts_so_far + [act_str], new_state, new_msgs))

            # Keep top beam_width sequences
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = new_beam[:beam_width]

            # Early stop: if top of beam is terminal, and we’ve looked at least 1 step
            if beam and beam[0][2].get("done"):
                break

        # Pick best sequence; fallback respond if empty
        best = max(beam, key=lambda x: x[0]) if beam else None
        if not best or not best[1]:
            return "<respond> None </respond>"
        return best[1][0]
