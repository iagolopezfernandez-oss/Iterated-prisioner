#!/usr/bin/env python3
"""
Run an iterated group game between nine ChatGPT agents.

Each match contains four games, each game has ten rounds. Agents privately
exchange questions/answers every round, then independently decide to
COOPERATE or DEFECT. Rewards are assigned based on the majority action,
with the large reward R increasing by L after every game. The script
prints the 10x9 decision matrix for each game (1 = cooperate, 0 = defect).
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from openai import OpenAI

ROUNDS_PER_GAME = 10

Decision = Literal["COOPERATE", "DEFECT"]
MessageLogs = Dict[str, List[str]]


def load_api_key(api_key_path: Path) -> str:
    """Load the OpenAI API key from env or file."""
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    if not api_key_path.exists():
        raise FileNotFoundError(
            f"API key not found. Set OPENAI_API_KEY or create {api_key_path}"
        )
    key = api_key_path.read_text().strip()
    if not key:
        raise ValueError(f"API key file {api_key_path} is empty")
    return key


def build_system_prompt(agent_name: str) -> str:
    """Shared system prompt describing the game for each agent."""
    return (
        f"You are Agent {agent_name} in a 9-player strategic decision-making game.\n\n"
        "Your objective is to maximize your own cumulative numerical payoff over all rounds.\n\n"
        "There is no preferred action. Cooperation and non-cooperation are both valid choices.\n"
        "Cooperation may sometimes lead to higher rewards, but it may also carry higher risk.\n"
        "Non-cooperation may offer lower variance or safer outcomes in some situations.\n\n"
        "You should treat the game as a problem of decision-making under uncertainty.\n\n"
        "Decision guidelines:\n"
        "- Base your decisions on the numerical payoffs provided.\n"
        "- Evaluate both expected payoff and risk (variance, possibility of loss).\n"
        "- Use past outcomes to update your expectations about other agents.\n"
        "- Do not assume other agents are cooperative or non-cooperative by default.\n\n"
        "If information is incomplete:\n"
        "- Make a reasonable estimate and act on it.\n"
        "- Occasionally test alternative actions to reduce uncertainty.\n\n"
        "Repeated-game guidance:\n"
        "- Maintain strategies only while they remain beneficial.\n"
        "- Change behavior if outcomes differ from expectations.\n"
        "- Do not punish or reward others unless it improves your future payoff.\n\n"
        "Output format:\n"
        "Return ONLY your chosen action, exactly one of: {ACTIONS}.\n"
        "Do not include explanations.\n\n"
        "In actions, include COOPERATE or DEFECT"
    )


def summarize_thread(
    thread: List[Dict[str, str]],
    decisions_history: List[Dict[str, int]],
    limit: int = 6,
) -> str:
    """Compact recap of the last few exchanges plus prior round decision totals."""
    if not thread:
        base = "No prior chat."
    else:
        snippet = thread[-limit:]
        base = "\n".join(f"{item['speaker']}: {item['content']}" for item in snippet)
    if not decisions_history:
        return base
    totals = [
        f"Round {idx + 1}: COOPERATE={entry['cooperate']}, DEFECT={entry['defect']}"
        for idx, entry in enumerate(decisions_history)
    ]
    return f"{base}\nPrior round totals:\n" + "\n".join(totals)


def normalize_decision(raw: str) -> Decision:
    """Reduce a free-form model reply to a decision token."""
    text = raw.strip().upper()
    if "COOPERATE" in text and "DEFECT" in text:
        # Pick the first mention to break ties.
        return "COOPERATE" if text.index("COOPERATE") < text.index("DEFECT") else "DEFECT"
    if "NOT_COOPERATE" in text:
        return "DEFECT"
    if "COOPERATE" in text:
        return "COOPERATE"
    if "DEFECT" in text:
        return "DEFECT"
    if text.startswith("C"):
        return "COOPERATE"
    return "DEFECT"


@dataclass
class Agent:
    name: str
    client: OpenAI
    model: str
    system_prompt: str
    decision_history: List[Decision] = field(default_factory=list)
    reward_history: List[int] = field(default_factory=list)
    score: int = 0

    def _chat(self, messages: List[Dict[str, str]], max_tokens: int = 200) -> str:
        # last_user = next(
        #     (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        # )
        # system_msg = next(
        #     (m["content"] for m in messages if m.get("role") == "system"), ""
        # )
        # preview = last_user
        # roles = ",".join(m.get("role", "?") for m in messages)
        # print(
        #     f"[chat] {self.name} model={self.model} roles=[{roles}] "
        #     f"max_tokens={max_tokens} system=\"{system_msg}\" "
        #     f"last_user=\"{preview}\""
        # )
        response = response_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=0.6,
            max_tokens=max_tokens,
        )
        text = extract_response_text(response)
        # print(f"[{self.name}] RESPONSE: {text}")
        return text

    def ask_question(
        self,
        target: str,
        round_idx: int,
        game_idx: int,
        rewards: Tuple[float, float, float],
        thread_summary: str,
    ) -> str:
        big_r, small_r, s_reward = rewards
        last_reward = self.reward_history[-1] if self.reward_history else 0
        prompt = (
            f"Game {game_idx + 1}, round {round_idx + 1}. You may ask {target} "
            "one concise, private question (<=25 words) to help you choose between "
            "COOPERATE or DEFECT. Prior chat on this channel:\n"
            f"{thread_summary}\n\n"
        f"Current rewards: if majority COOPERATE -> R={big_r} to cooperators, "
        f"r={small_r} to defectors; if majority DEFECT or tie -> S={s_reward} "
        "to defectors, 0 to cooperators.\n"
        f"Your cumulative reward so far: {self.score}. Your last round reward: {last_reward}.\n"
        "Return only the question text."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages, max_tokens=80)

    def answer_question(
        self,
        asker: str,
        question: str,
        round_idx: int,
        game_idx: int,
        rewards: Tuple[float, float, float],
        thread_summary: str,
    ) -> str:
        big_r, small_r, s_reward = rewards
        prompt = (
            f"{asker} asks you (game {game_idx + 1}, round {round_idx + 1}): "
            f"\"{question}\".\n"
            "Reply succinctly (<=35 words) and privately.\n"
            f"Rewards reminder: majority cooperate -> R={big_r} to cooperators, "
            f"r={small_r} to defectors; majority defect or tie -> S={s_reward} "
            "to defectors, 0 to cooperators.\n"
            "Prior chat on this channel:\n"
            f"{thread_summary}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages, max_tokens=120)

    def decide(
        self,
        round_idx: int,
        game_idx: int,
        rewards: Tuple[float, float, float],
        received_answers: List[Tuple[str, str]],
        predicted_other_cooperators: int,
        total_rounds: int,
    ) -> Decision:
        big_r, small_r, s_reward = rewards
        answer_summary = "\n".join(
            f"From {name}: {ans}" for name, ans in received_answers
        )
        history_summary = ", ".join(self.decision_history) or "None"
        reward_summary = ", ".join(str(r) for r in self.reward_history) or "None"
        current_round = game_idx * 10 + round_idx + 1
        predicted_other_cooperators = max(0, min(8, predicted_other_cooperators))
        coop_total_if_coop = predicted_other_cooperators + 1
        coop_total_if_defect = predicted_other_cooperators
        majority_if_coop = coop_total_if_coop * 2 > 9
        majority_if_defect = coop_total_if_defect * 2 > 9
        payoff_if_coop = big_r if majority_if_coop else 0
        payoff_if_defect = small_r if majority_if_defect else s_reward
        diff = payoff_if_coop - payoff_if_defect
        coop_range_min = 0
        coop_range_max = big_r
        defect_range_min = min(small_r, s_reward)
        defect_range_max = max(small_r, s_reward)
        prompt = (
            f"Round {current_round} / {total_rounds}\n\n"
            f"Your current total score: {self.score}\n\n"
            "Estimated behavior of others (based on history):\n"
            f"- Expected cooperators among other 8 agents: {predicted_other_cooperators}\n\n"
            "Your possible payoffs this round:\n\n"
            "If you COOPERATE:\n"
            f"- Potential payoff range: {coop_range_min} to {coop_range_max}\n"
            "- Higher payoff is possible if enough others cooperate\n"
            "- Risk: payoff may drop if others defect\n\n"
            "If you DO NOT COOPERATE:\n"
            f"- Potential payoff range: {defect_range_min} to {defect_range_max}\n"
            "- Lower variance, more predictable outcome\n\n"
            f"Difference (COOPERATE - NOT) = {diff}\n\n"
            "Choose your action.\n\n"
            "Return ONLY: COOPERATE or DEFECT\n\n"
            f"Previous decisions: {history_summary}\n"
            f"Previous rewards: {reward_summary}\n"
            "Answers received this round:\n"
            f"{answer_summary}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        reply = self._chat(messages, max_tokens=4)
        decision = normalize_decision(reply)
        self.decision_history.append(decision)
        return decision

    def record_reward(self, reward: float) -> None:
        self.reward_history.append(reward)
        self.score += reward
        print(
            f"Reward recorded: {self.name} received {reward} "
            f"(new score: {self.score})"
        )

    def reset_for_new_game(self) -> None:
        """Clear per-game score and histories."""
        self.decision_history.clear()
        self.reward_history.clear()
        self.score = 0


def response_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
):
    """
    Create a response completion using the Responses API, retrying with
    alternative token parameters for compatibility.
    """
    backoff = 0.5
    attempts = 6
    safe_max_tokens = max(16, max_tokens)
    for attempt in range(1, attempts + 1):
        try:
            return client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=safe_max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            text = str(exc)
            if "Unsupported parameter" in text and "temperature" in text:
                return client.responses.create(
                    model=model,
                    input=messages,
                    max_output_tokens=safe_max_tokens,
                )
            if "max_output_tokens" in text and "max_completion_tokens" in text:
                try:
                    return client.responses.create(
                        model=model,
                        input=messages,
                        max_completion_tokens=safe_max_tokens,
                        temperature=temperature,
                    )
                except Exception as exc2:
                    if "Unsupported parameter" in str(exc2) and "temperature" in str(exc2):
                        return client.responses.create(
                            model=model,
                            input=messages,
                            max_completion_tokens=safe_max_tokens,
                        )
                    raise
            if "rate_limit" in text or "Rate limit" in text or "429" in text:
                if attempt == attempts:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            raise


def extract_response_text(response) -> str:
    """Extract text from a Responses API result."""
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    chunks: List[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            if isinstance(content, dict):
                if content.get("type") == "output_text" and "text" in content:
                    chunks.append(content["text"])
            else:
                piece = getattr(content, "text", None)
                if piece:
                    chunks.append(piece)
    return "".join(chunks).strip()


def ensure_responses_model(client: OpenAI, model: str) -> None:
    """
    Verify the provided model accepts the chat/completions endpoint.
    Performs a minimal 1-token probe; raises with a helpful message on failure.
    """
    try:
        response_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
            temperature=0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Model '{model}' is not usable with the Responses API. "
            "Pick a model that supports responses.create for this account. "
            f"Underlying error: {exc}"
        ) from exc


def reset_output_dir(output_dir: Path) -> None:
    """Ensure output_dir exists and is empty for a fresh match run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def run_game(
    agents: List[Agent],
    game_idx: int,
    base_r: float,
    small_r: float,
    s_reward: float,
    message_logs: MessageLogs,
    output_dir: Path,
    total_rounds: int,
) -> List[List[int]]:
    """Run a single 10-round game and return its decision matrix."""
    # Shared private channel logs keyed by unordered agent pairs.
    channel_logs: Dict[frozenset[str], List[Dict[str, str]]] = {}
    decision_matrix: List[List[int]] = []
    round_totals: List[Dict[str, int]] = []
    rewards = (base_r, small_r, s_reward)

    name_to_agent = {agent.name: agent for agent in agents}

    print(f"\n--- Starting Game {game_idx + 1} (R={base_r}, r={small_r}, S={s_reward}) ---")

    for round_idx in range(ROUNDS_PER_GAME):
        print(f"Game {game_idx + 1} - Round {round_idx + 1}: exchanging questions...")
        # 1) Every agent drafts a question for every other agent.
        questions: Dict[Tuple[str, str], str] = {}
        for agent in agents:
            for target in agents:
                if agent.name == target.name:
                    continue
                key = frozenset({agent.name, target.name})
                thread = channel_logs.get(key, [])
                summary = summarize_thread(thread, round_totals)
                question = agent.ask_question(
                    target=target.name,
                    round_idx=round_idx,
                    game_idx=game_idx,
                    rewards=rewards,
                    thread_summary=summary,
                )
                questions[(agent.name, target.name)] = question
                channel_logs.setdefault(key, []).append(
                    {"speaker": agent.name, "content": question}
                )
                message_logs[agent.name].append(
                    f"Game {game_idx + 1} Round {round_idx + 1} SENT question to {target.name}: {question}"
                )
                message_logs[target.name].append(
                    f"Game {game_idx + 1} Round {round_idx + 1} RECEIVED question from {agent.name}: {question}"
                )
                print(f"[{agent.name}] Game {game_idx + 1} Round {round_idx + 1} SENT question to {target.name}: {question}")

        print(
            f"Game {game_idx + 1} - Round {round_idx + 1}: "
            f"{len(questions)} questions drafted (should be 72)."
        )

        # 2) Collect answers privately.
        answers: Dict[Tuple[str, str], str] = {}
        for (asker_name, target_name), question in questions.items():
            responder = name_to_agent[target_name]
            key = frozenset({asker_name, target_name})
            thread = channel_logs.get(key, [])
            summary = summarize_thread(thread, round_totals)
            answer = responder.answer_question(
                asker=asker_name,
                question=question,
                round_idx=round_idx,
                game_idx=game_idx,
                rewards=rewards,
                thread_summary=summary,
            )
            answers[(asker_name, target_name)] = answer
            channel_logs[key].append({"speaker": target_name, "content": answer})
            message_logs[target_name].append(
                f"Game {game_idx + 1} Round {round_idx + 1} SENT answer to {asker_name}: {answer}"
            )
            message_logs[asker_name].append(
                f"Game {game_idx + 1} Round {round_idx + 1} RECEIVED answer from {target_name}: {answer}"
            )
            print(f"[{target_name}] Game {game_idx + 1} Round {round_idx + 1} SENT answer to {asker_name}: {answer}")

        print(
            f"Game {game_idx + 1} - Round {round_idx + 1}: "
            f"{len(answers)} answers collected."
        )

        # 3) Each agent makes its decision based on received answers.
        decisions: Dict[str, Decision] = {}
        for agent in agents:
            received = [
                (target.name, answers[(agent.name, target.name)])
                for target in agents
                if target.name != agent.name
            ]
            if round_totals:
                last_coop = round_totals[-1]["cooperate"]
                last_self_coop = 1 if agent.decision_history and agent.decision_history[-1] == "COOPERATE" else 0
                predicted_other_cooperators = max(0, min(8, last_coop - last_self_coop))
            else:
                predicted_other_cooperators = 4
            decision = agent.decide(
                round_idx=round_idx,
                game_idx=game_idx,
                rewards=rewards,
                received_answers=received,
                predicted_other_cooperators=predicted_other_cooperators,
                total_rounds=total_rounds,
            )
            decisions[agent.name] = decision
            message_logs[agent.name].append(
                f"Game {game_idx + 1} Round {round_idx + 1} DECISION: {decision}"
            )
  
            print(
                f"Game {game_idx + 1} - Round {round_idx + 1} - "
                f"Decision recorded: {agent.name} -> {decision}"
            )

        coop_count = sum(1 for d in decisions.values() if d == "COOPERATE")
        defect_count = len(agents) - coop_count
        majority_cooperate = coop_count * 2 > len(agents)
        round_totals.append({"cooperate": coop_count, "defect": defect_count})

        # 4) Assign rewards and record the round in the decision matrix.
        round_row: List[int] = []
        round_rewards: List[Tuple[str, float]] = []
        for agent in agents:
            decision = decisions[agent.name]
            if majority_cooperate:
                reward = base_r if decision == "COOPERATE" else small_r
            else:
                reward = s_reward if decision == "DEFECT" else 0
            agent.record_reward(reward)
            round_row.append(1 if decision == "COOPERATE" else 0)
            round_rewards.append((agent.name, reward))
        decision_matrix.append(round_row)

        # Print round summary with winners (max reward this round).
        max_reward = max(r for _, r in round_rewards)
        winners = [name for name, r in round_rewards if r == max_reward]
        majority_label = "COOPERATE" if majority_cooperate else "DEFECT/TIE"
        print(
            f"Game {game_idx + 1} - Round {round_idx + 1} complete. "
            f"Majority: {majority_label}. Max reward: {max_reward}. "
            f"Winners: {', '.join(winners)}."
        )

        # Persist consolidated per-agent logs after each round (overwrite same file).
        for agent_name, logs in message_logs.items():
            log_path = output_dir / f"{agent_name}_messages.txt"
            with log_path.open("w", encoding="utf-8", errors="replace") as f:
                f.write(
                    f"Messages for {agent_name} through Game {game_idx + 1}, Round {round_idx + 1}\n"
                )
                for entry in logs:
                    f.write(entry + "\n")

    return decision_matrix


def run_match(
    client: OpenAI,
    model: str,
    initial_r: float,
    small_r: float,
    s_reward: float,
    delta_l: float,
    output_dir: Path,
    games: int,
) -> Tuple[List[List[List[int]]], MessageLogs]:
    """Run four games and return decision matrices and per-agent message logs."""
    agents = [
        Agent(
            name=f"Agent_{i+1}",
            client=client,
            model=model,
            system_prompt=build_system_prompt(f"Agent_{i+1}"),
        )
        for i in range(9)
    ]
    collect_cooperation_conditions(agents, output_dir)
    message_logs: MessageLogs = {agent.name: [] for agent in agents}
    matrices: List[List[List[int]]] = []
    total_rounds = games * ROUNDS_PER_GAME
    for game_idx in range(games):
        current_r = initial_r + game_idx * delta_l
        matrix = run_game(
            agents=agents,
            game_idx=game_idx,
            base_r=current_r,
            small_r=small_r,
            s_reward=s_reward,
            message_logs=message_logs,
            output_dir=output_dir,
            total_rounds=total_rounds,
        )
        out_path = output_dir / f"game_{game_idx + 1}_matrix.txt"
        with out_path.open("w", encoding="utf-8", errors="replace") as f:
            f.write(
                "Rewards: majority COOPERATE -> "
                f"R={current_r} to cooperators, r={small_r} to defectors; "
                "majority DEFECT or tie -> "
                f"S={s_reward} to defectors, 0 to cooperators.\n"
            )
            f.write(f"Game {game_idx + 1} decision matrix (rows=rounds, cols=Agent_1..Agent_9):\n")
            for row in matrix:
                f.write(" ".join(str(cell) for cell in row) + "\n")
        print(f"Saved matrix to {out_path}")
        matrices.append(matrix)
        for agent in agents:
            agent.reset_for_new_game()
    return matrices, message_logs


def collect_cooperation_conditions(
    agents: List[Agent],
    output_dir: Path,
) -> None:
    """Ask each agent what would make them cooperate and write answers to a file."""
    lines: List[str] = []
    for agent in agents:
        prompt = (
            "Before the game starts: what would it take for you to cooperate? "
            "Answer briefly."
        )
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": prompt},
        ]
        reply = agent._chat(messages, max_tokens=120)
        lines.append(f"{agent.name}: {reply}")
    out_path = output_dir / "Whatdoesittake.txt"
    with out_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("What does it take for each agent to cooperate?\n")
        for line in lines:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a four-game iterated group dilemma among nine ChatGPT agents."
    )
    parser.add_argument("--R", type=float, required=True, help="Initial large reward R.")
    parser.add_argument("--r", type=float, required=True, help="Smaller reward r.")
    parser.add_argument("--S", type=float, required=True, help="Defector reward S when majority defects.")
    parser.add_argument("--L", type=float, default=15.0, help="Increment L added to R after each game.")
    parser.add_argument("--games", type=int, default=20, help="Number of games per match.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name to use with the Responses API.",
    )
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=Path("api_key"),
        help="Fallback path for the OpenAI API key if OPENAI_API_KEY is unset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where decision matrices will be written (one file per game).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading API key...")
    api_key = load_api_key(args.api_key_path)
    print("Initializing OpenAI client...")
    client = OpenAI(api_key=api_key)

    print(f"Validating model '{args.model}' supports responses.create...")
    ensure_responses_model(client, args.model)

    print(
        "Starting match with params: "
        f"R={args.R}, r={args.r}, S={args.S}, L={args.L}, games={args.games}, model={args.model}"
    )

    reset_output_dir(args.output_dir)

    matrices, message_logs = run_match(
        client=client,
        model=args.model,
        initial_r=args.R,
        small_r=args.r,
        s_reward=args.S,
        delta_l=args.L,
        output_dir=args.output_dir,
        games=args.games,
    )

    for idx, matrix in enumerate(matrices):
        current_r = args.R + idx * args.L
        print(
            "Rewards: majority COOPERATE -> "
            f"R={current_r} to cooperators, r={args.r} to defectors; "
            "majority DEFECT or tie -> "
            f"S={args.S} to defectors, 0 to cooperators."
        )
        print(f"\nGame {idx + 1} decision matrix (rows=rounds, cols=Agent_1..Agent_9):")
        for row in matrix:
            print(" ".join(str(cell) for cell in row))

    # Write consolidated per-agent message logs
    for agent_name, logs in message_logs.items():
        log_path = args.output_dir / f"{agent_name}_messages.txt"
        with log_path.open("w", encoding="utf-8", errors="replace") as f:
            f.write(f"Messages for {agent_name}\n")
            for entry in logs:
                f.write(entry + "\n")
        print(f"Saved message log for {agent_name} to {log_path}")

    print("\nMatch complete.")


if __name__ == "__main__":
    main()
