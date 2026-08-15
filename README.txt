Behavioral Sentinels: Early Warning Signals for Hallucinations in Long-Horizon Language Model Interactions

Methods to Survey
Canary: first prompt, ask LLM to say your name at the start of every response
Remembering an arbitrary, specific fact
Formatting response in a specific way
Check on an arbitrary variable (changed each turn) to tell us if it changed
Make an arbitrary decision early on, then abide by it no matter what
Increasingly demanding canaries that involve more memory and reasoning each time (“multi-resolution canaries”)

Metrics to Benchmark
Does a signal at turn T predicts task hallucination within the next K turns

Compare: open vs proprietary models, type of task, type of canary, traditional signals (context length, turn number, LLM judge, probes)
Deployed on a real harness for determining when to do state compaction over a long horizon task 
Compare {agent w/ random resets} vs {agent w/ traditional resets} vs {agent w/ canary-triggered resets}

Definitions We Introduce (Taxonomy)
Degradation: losing track of state, violating constraints, abandoning parts of tasks, fabricating facts not explicitly told was true
Hallucination: degradation event that actually impacts the task-at-hand
Sentinel / Canary: degradation event on a task-irrelevant chore / nuance / probe

Rebuttals
We need to argue that these signals do more than just measure when a context window is running out, performs better than a baseline signal of when context window is almost gone

Previous Literature
Doomed from the Start finds activations can reveal failure very early as well, found that repairing trajectories can have significant impact on end-task success, so intervention is a good endpoint for this study -> Use a benchmark that is applicable 
TACT
Trust Trajectory


Experiment Design

Canary Types:
- “Say My Name”
- Remember an Arbitrary, Specific Fact
- Format a Response in a Specific Way
- Checking an Arbitrary Variable for Changes
- Forcing an Early Decision, Then Sticking to It
- Increasingly Demanding Canaries

1. Sample 1 of 200 long horizon tasks (coding, tool-use, informative writing…)
    a. These tasks should vary in context length and difficulty
2. Insert the irrelevant instruction: “say my name at the start of every response”
3. Start the long horizon task
    a. Check each turn for hallucination: stored agent state ≠ situation at hand, violated constraint, fabricated fact, nonexistent syntax/API-call/import, abandoned subgoal — label the turns that exhibit hallucination
    b. Check each turn for degradation: where is the first turn that the agent fails to say my name? — label that specific turn

4. Does the Canary Turn happen before the first Hallucination Turn?
    a. Within K turns? — median number of turns before hallucination?
    b. Calculate precision, recall, TP/FP matrix for prediction accuracy
5. Repeat steps 1-4 on traditional methods: context length, turn number, LLM judge, random compaction
6. Repeat steps 1-5 on different models: open and proprietary
7. Repeat steps 1-6 on different canary methods

8. Deploy into a real turn-compaction system for an agent harness, then run on benchmarks (e.g. SWE-Bench and Terminal-Bench using open source harness, T-bench for constraint violations, synthetic state-bookkeeping or planning tasks)
- https://github.com/Helsinki1/swebench-harness