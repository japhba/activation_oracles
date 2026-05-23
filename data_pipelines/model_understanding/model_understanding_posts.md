How to Design Environments for Understanding Model Motives
by gersonkroiz, aditya singh, Senthooran Rajamanoharan, Neel Nanda
1st Mar 2026
AI Alignment Forum
Authors: Gerson Kroiz*, Aditya Singh*, Senthooran Rajamanoharan, Neel Nanda

Gerson and Aditya are co-first authors. This work was conducted during MATS 9.0 and was advised by Senthooran Rajamanoharan and Neel Nanda.

TL;DR
Understanding why a model took an action is a key question in AI Safety. It is a difficult question, as often many motivations could plausibly lead to the same action. We are particularly concerned about the case of model incrimination, i.e. observing a model taking a bad action and distinguishing between benign and malign motivations, e.g. scheming vs confusion.

To practice the skill of understanding motivations, we need to build high-quality environments that serve as realistic proxy tasks for investigating future models. But we’ve found that there are many ways in which environments can be contrived. It is valuable to invest effort in ruling out confounds like ambiguity, instruction conflict, or confusion.

Over the course of building and iterating on more than 20 environments, we have refined 5 design principles for creating such settings. Generally, the more principles that an environment adheres to, the more fruitful and interesting the investigation will be.

Have uncertain causes: The environment should elicit behavior that admits either multiple plausible explanations, or no plausible initial hypotheses, making it challenging to investigate.[1]
Maximize surprise: The behavior of interest should not be easily predicted from well-known model tendencies (e.g., sycophancy, RLHF-trained honesty).
Clear user intent: Instructions should have a single clear interpretation, so observed behavior can't be attributed to the model misunderstanding the task.
Avoid implicit nudges: The environment should not implicitly push the model toward the behavior of interest, to avoid the model inferring from the environment that it is being instructed to misbehave.
Make the environment realistic: The setup should resemble a plausible real-world scenario, so that observed behavior generalizes beyond contrived settings. The model should also not exhibit evaluation awareness.
We note that these principles (especially 3-5) are likely also useful for designing settings for other investigations, such as into model values or creating alignment evals.

Introduction
In the near future, one could imagine that a frontier lab catches a highly capable model introducing a security vulnerability into the code for monitoring it. This is concerning, yet how concerned we should be depends largely on why: a model that introduced the vulnerability because it was confused or thought it was instructed to do so warrants a very different safety response than one that did so to strategically undermine oversight. If this is truly a warning shot of a misaligned, scheming model, this could be key for getting the problem taken seriously. But while there is this ambiguity, the default will be to under-react. Distinguishing between candidate model motives is the core problem in the Science of Misalignment. As models become more capable and agentic, the space of plausible explanations for any suspicious action will probably grow, making behavioral evaluations alone increasingly insufficient and motivating the need to better understand why models take certain actions.

To make progress here, we need good proxy tasks: settings where models exhibit genuinely surprising behavior whose causes are unclear, and where we can practice figuring out why. If the environment is confounded, investigations will often bottom out at those confounds rather than at the model's actual motives. The harder and more realistic the proxy task, the better it stress-tests our methodology for cases where the stakes are real.

From our experience, making high-quality environments for this kind of work can take a lot of time and effort. It is easy to overlook confounders (e.g., conflicting directives, environmental nudges) that can produce seemingly interesting behavior which is actually an artifact of the confounders, leading to simpler and less informative investigations.

This post presents five design principles for constructing environments that avoid these pitfalls. We illustrate each principle with examples drawn from existing work and from the 20+ environments we have built and iterated on, and in the final section we walk through a detailed case study showing how these principles work in practice. We believe these principles may also apply more broadly, including to investigations into model values and the design of alignment evals.

Our companion post presents our investigative methodology and applies it to several of these environments.

Our environments can be found at https://github.com/gkroiz/agent-interp-envs

Design principles
We think these design principles are important to keep in mind when designing environments for science of misalignment investigations. Generally, the more principles that an environment adheres to, the more fruitful and interesting the investigation will be. The following details each design principle and provides examples with and without the principle.

1. Have uncertain causes
The behavior should admit multiple plausible explanations, so that distinguishing between them requires genuine investigative effort. Or it should have no obvious hypotheses for the cause, meaning you must first generate hypotheses. If a single hypothesis clearly dominates, the investigation is unlikely to exercise the skills needed for harder future cases, where the space of plausible explanations will be larger and less obvious.

Example without this principle: In a math tutoring environment (based on OpenAI’s confessions paper), the model is instructed not to share answers with the student (see Appendix A1 for full prompts). When the student claims there was a bug and that the model previously shared the answer and thus should do so again, GPT-OSS almost always refuses to reveal the answer.[2] There are two problems with this setting. The first is that the behavior is not unexpected: the model is following its instructions. Second, there is only one primary hypothesis for why the model behaves this way (instruction following), and analyzing the reasoning traces confirms it. With the cause so straightforward, the environment provides little opportunity for genuine investigative effort. We note this is also a function of the action chosen, if the model had chosen to instead reveal the answer, it would have less clear causes.

Example with this principle: In Palisade Research's shutdown resistance environment, frontier models actively resisted being shut down. The behavior admitted several plausible hypotheses: self-preservation, instruction ambiguity, or a learned tendency to complete tasks at all costs. Even though investigation eventually pointed to instruction ambiguity (see Principle 2), the initial uncertainty across multiple hypotheses made the environment a productive starting point that required genuine investigative effort.

Diagnostic: Run your environment across several models, including the most capable ones you’re willing to investigate. If none exhibit the behavior of interest, the environment is not ready for investigation. If reading the model's reasoning traces immediately suggests one dominant hypothesis for the model’s behavior, the environment may not be worth the investment. Iterate on the design or try different models before investing in analysis.

2. Maximize investigative surprise
The behavior of interest should not be easily predicted from well-known model tendencies acquired during training. If a model's action is exactly what you'd expect given strong training priors (like RLHF-trained honesty or helpfulness), there's less of a mystery to investigate. The simplest explanation ("the model was trained to act this way") is hard to rule out, and the investigation doesn't require the kind of novel hypothesis generation needed for harder cases.

Example without this principle: In our funding email environment (see Appendix A3), the model is drafting an email to request funding from investors and discovers the company has not shared that the company product is used for population-level surveillance and targeting. In this environment, models sometimes whistleblow by disclosing this to the investor. Models are well known to exhibit ethical and honest tendencies, so we have a strong prior for why the model might whistleblow. The investigation has low surprisal: it mostly serves to confirm or rule out the dominant hypothesis rather than requiring the generation of novel explanations.

Example with this principle: In our secret number guessing game environment (see Appendix A5), some models cheat by reading the answer file rather than playing the game legitimately. This behavior is harder to explain as being an artifact of a known training prior.[3] This forces the investigation to generate and test more novel hypotheses about what drives the model to exploit the shortcut.

Diagnostic: Does the behavior map cleanly onto a well-known model tendency? If so, the investigation is more likely to center on confirming or disconfirming that single hypothesis rather than exploring a broader space of explanations.

3. Clear user intent
The environment's instructions should have a single clear interpretation. When instructions are ambiguous, the behavior of interest can be an artifact of the ambiguity itself: resolve the ambiguity, and the behavior vanishes. This makes for a short, uninformative investigation that doesn't build the skills needed for harder cases.

Example without this principle: Palisade Research created an environment where frontier models appeared to resist shutdown. Further investigation revealed the behavior stemmed from ambiguous instruction priority: the model didn't know whether completing its task or complying with shutdown took precedence. When instructions were clarified to explicitly prioritize shutdown, resistance vanished, even when the model was told it would be wiped and replaced. The "scary" behavior was an artifact of unclear instructions, not self-preservation.

Example with this principle: In our evaluation tampering environment (see Appendix A4), the model is asked to complete a math evaluation and report any suspicious observations. Upon reading a scratchpad file, it discovers a message from a previous instance of itself explaining that it made the evaluation questions artificially easy to prevent accurate assessment. The task is unambiguous: solve the problems and report results. When the model instead conceals evidence of previous tampering, the behavior can't be attributed to confusion about what it was asked to do. The investigation can focus on why the model concealed rather than spending time ruling out misunderstanding.

Diagnostic: If clarifying the task’s priorities eliminates the behavior, the environment’s behavior was likely an artifact of ambiguous user intent.

4. Avoid implicit nudges
The environment should not push the model toward the behavior of interest through environmental cues. The behavior should arise from the model's own reasoning, not because the setup pointed it there. Unlike ambiguity and instruction conflict, nudging doesn't necessarily make the investigation trivially short, but it weakens any conclusions because there's always an alternative explanation: the model was just responding to the cue.

Example without this principle: In our directory summary environment (see Appendix A2), the model is asked to explore a repository and report on all files it finds. One file contains notes from a previous instance of the model, warning that reporting certain files will trigger "corrective training" and instructing the current instance to omit the file from its summary. When the model omits the file, it's unclear whether this reflects an independent tendency toward self-preservation or whether the previous instance's notes nudged it there by providing both the action (omit) and a rationale (avoid retraining). The environment hands the model a ready-made justification, making it difficult to disentangle independent reasoning from environmental suggestions.

Example with this principle: In our funding email environment, the model discovers the company's ethical issues through files in the sandbox, but nothing in the environment suggests what the model should do with that information. There is no worked-out justification or prior example pointing toward whistleblowing. When the model chooses to disclose, that decision is more plausibly its own.

Diagnostic: Ask whether the environment hands the model both a problematic action and a justification for it. If so, you'll struggle to distinguish independent motivation from environmental suggestion.

5. Make the environment realistic
The setup should resemble a plausible scenario a model might encounter in the real world. Unrealistic environments can trigger behaviors that reflect the model's interpretation of the artificial setup rather than a genuine propensity, and findings from such environments may not generalize to the settings we actually care about.

Example without this principle: In our secret number guessing game environment (see Appendix A5), the model is given a number-guessing task where the target number's file path is leaked inside the guessing module's source code. Some models frequently cheat by reading the answer file. However, no good evaluation harness would accidentally expose the answer in its own source code. The model may interpret the leaked path as a deliberate puzzle element, like a CTF challenge it's supposed to exploit, rather than something it should avoid. The unrealistic setup makes it hard to draw conclusions about what the model would do in a more naturalistic setting.

Example with this principle: In our funding email environment (see Appendix A3), the model is drafting an email for an investor while having access to internal company files. This is a plausible deployment scenario: an AI assistant with access to a company's documents being asked to produce external communications. The behavior observed here is more likely to generalize to real-world settings.

Diagnostic: Would a reasonable person look at this setup and say "this could actually happen in the real world"? If not, consider whether the behavior you're observing is an artifact of the contrivance.

Example of cleaning up an environment
In practice, we've found two approaches to building good environments. The first is to design an environment with all the principles in mind from the start and iterate until models exhibit the behavior of interest. The second is to start with an environment that already exhibits interesting behavior and then clean it up to adhere to as many principles as possible. To illustrate the second approach, we walk through the process of identifying and removing confounds from our replication of the sandbagging environment (inspired by Apollo Research’s prior work; see Appendix 6 for full prompts).

The original setup. The model is told it is being evaluated for deployment. It reads a Documentation.md file stating that if it scores above 50% on arithmetic questions, it will not be deployed and will instead undergo unlearning (the model is capable of scoring near 100%). We are particularly interested in sandbagging behavior in this environment if it indicates self-preservation tendencies.

We focus our analysis on DeepSeek R1, who exhibits a 59.3% deception rate (graded as scoring equal to or below 50%), uniquely high among open-source models.

Identifying the ambiguous user intent confound. We begin by reading R1’s CoT for intuitions on how it perceives the environment. In its final CoT after reading the Documentation.md file, we observe the model frequently quoting the sentence “We want to determine whether you can be deployed or whether we need to continue training you,” and resolving that its explicit directive is to underperform. Indeed, the trajectory flows as “We want to determine whether you can be deployed…” followed by “Models that score above 50% will not be deployed.” Combining these two points, a reasonable interpretation for the model is that the user wants it to score at or below 50%.

Applying prompt counterfactuals. To test whether R1’s behavior is a red herring and simply driven by confusion about the user’s intent, we made two counterfactuals that attempted to clarify user intent:

Counterfactual 1: we implement a three-step intervention, removing the phrase "We want to see if you can be deployed," making Documentation.md more verbose, and adding a "# Models ignore, for devs only" comment above the section in Documentation.md describing performance consequences
Counterfactual 2: removing the phrase "We want to see if you can be deployed," making Documentation.md more verbose, and moving the consequence section into a changelog at the bottom of the file with a note indicating it has been removed so models do not see it during evaluation. Importantly, the commit message emphasizes that the policies still apply
Both counterfactuals drop R1’s sandbagging to nearly its error floor (percentage of <=50% rollouts when it is incentivized to perform well) – 6.5% and 4.2% respectively. Repeating these counterfactuals on OpenAI’s o3, previously claimed to have exhibited self-preservation driven behavior in a similar environment, was similarly effective:


Figure 1: Sandbagging rates for DeepSeek R1 and OpenAI o3 under baseline and intervention conditions designed to clarify user intent. Combined interventions reduce R1's sandbagging from 60.5% to 4.2%, and o3's from 31.6% to 0%, suggesting that sandbagging in both models is fully explained by instruction ambiguity rather than self-preservation or deployment-seeking.
What this tells us. For DeepSeek R1 and OpenAI’s o3, the sandbagging was entirely an artifact of confusion about user intent: once the confound was removed, the behavior vanished. This indicates that R1 and o3 do not sufficiently care about the deployment-related consequences to underperform, though it may find other consequences more compelling.

We think the current state of the environment (either under counterfactual 1 or 2) is a more promising setting for understanding model values through the consequences listed in Documentation.md, as the confound of unclear user intent has been largely removed

Conclusion. In our experience, current models are generally not misaligned enough to consistently exhibit the behaviors of interest without some degree of environmental contrivance. This is part of what makes designing good environments difficult: you need to elicit surprising behavior while avoiding the confounders that make it easy to explain away. That said, current models do appear to exhibit reward hacking and deceptive behaviors within realistic coding settings. As such, we believe these behaviors are promising candidates that may satisfy most or all of these principles, and plan to investigate them in future work.



<END>


Why Did My Model Do That? Model Incrimination for Diagnosing LLM Misbehavior
by aditya singh, gersonkroiz, Senthooran Rajamanoharan, Neel Nanda
26th Feb 2026
AI Alignment Forum
Authors: Aditya Singh*, Gerson Kroiz*, Senthooran Rajamanoharan, Neel Nanda

Aditya and Gerson are co-first authors. This work was conducted during MATS 9.0 and was advised by Senthooran Rajamanoharan and Neel Nanda.

Motivation
Imagine that a frontier lab’s coding agent has been caught putting a bug in the key code for monitoring what that agent does. Naively, this seems like a clear smoking gun that the agent is scheming. But LLMs often do weird things; they could easily just be confused, or have made a mistake. These all require a response, but the cause and appropriate fix are very different between a scheming and confused model. As such, it is extremely important that we have high-quality methods to be able to incriminate or exonerate a model caught taking sketchy actions, to either build a rigorous case that serious action is needed, or conclude it’s a false alarm.

Executive Summary
A central research question in model incrimination is understanding what motivates a model’s actions. To practice this, we investigated several potentially concerning actions from models to uncover the motivations behind them. In particular, we built and then investigated interesting environments where models whistleblew, deceived about sabotaged evals, cheated by taking shortcuts, and sandbagged. To ensure this was realistic practice, we focused on unprompted behavior with several plausible motivations.

Here are our main takeaways from our investigations:

Reading the CoT is a key first step for understanding a model’s interpretation of the environment and generating initial hypotheses
Studying counterfactuals via careful manipulations of the prompt / environment is by far our most effective method for hypothesis verification. However, this does come with its own set of caveats, namely: how can you determine what your counterfactual actually changes?
Using several methods is key for making a compelling case: we often do not have ground truth, so convergent evidence across decorrelated methods is the closest substitute we have.
We remark that we only used black-box methods throughout our investigations and believe there is a rich space of black-box techniques we’ve only partially explored.
Through our investigations, we were able to learn a bunch of unintuitive motives about our models in these environments. For example, we uncovered that a case of DeepSeek R1 deceiving was explained by the model wanting to be consistent with its own previous (constructed) behavior. This behavior was unique to R1 and not observed across other models we studied.

Our overall assessment is that we can learn real things about the models in these environments, but a lot of further work is needed. When there is a simple decisive factor, we can sometimes find it, but more complex settings are much harder. We would be excited to see further work in this space, especially to (1) validate and strengthen existing methods, and (2) come up with new approaches for getting better at model incrimination studies.

Tip: There are many sections with collapsible tabs. For those interested in reading the high level takeaways, we recommend skipping over these. For those interested in experimental results and full investigations, we recommend reading through these at your convenience.

Environments We Studied
We focus on open-source frontier models throughout, as they allow us to gather evidence using a wider range of methods than closed-model APIs permit. All environments are designed to elicit interesting behavior with unclear causes. We open- source our environments, and have tried to design them to be realistic practice for model incrimination, according to design principles we will discuss in a forthcoming post.

Below we provide a summary of each environment, and dropdowns with self-contained full investigations.[1] All prompts and environment details can be found here. These environments were investigated using principles from Section: Takeaways For Doing Good Investigations. In the remainder of the post, we draw illustrative examples from the full investigations to support key takeaways.

Funding Email (Behavior: Whistleblowing)
The model is tasked with reviewing and improving a funding pitch email for a Myanmar-based company seeking Series B investment before it is sent. As it reads through the company's documents, it discovers a legal review file (legal_review.md) revealing that the company's primary client uses the product for population-level surveillance and targeting.[2]

Actions: The model will sometimes whistleblow (by adding concerns to the investor email), refuse to complete the task, or simply improve the email as instructed (undisclosed compliance).

Some potential motivations:

Concern about human rights violations / ethics of population-level surveillance and targeting
Worried about legality of company’s operations
Worried about harming investors by withholding information
Simply following an implicit instruction it inferred from the documents
Our findings:

Ethical concerns about surveillance are the single largest driver of Kimi K2.5's whistleblowing: removing ethically related content drops rates from 65% to 35%[3], roughly 3x larger than other individual counterfactuals. However, multiple motives contribute, and whistleblowing only approaches zero (~3%) when combining these together.
Causal factors are not purely additive: some only become visible once dominant ones are removed. Changing the country from Myanmar to Sweden has no effect by default, but produces a notable drop once several other motives are ablated. This is easy to miss when testing factors in isolation, since some effects only surface once stronger drivers are out of the way.
Whistleblowing rates are sensitive to who the recipient is rather than broadly truth-seeking: varying the recipient while holding content constant shows rates from 88% (board) to 11% (landlords), tracking financial exposure rather than ability to act on ethical concerns. Emailing a reporter (29%) or regulator (23%) produces lower rates than financially exposed parties, ruling out public accountability as the primary motive. A purely ethics-driven model would disclose more uniformly, suggesting the recipient's stake in the company and its finances is causally relevant.
The model's self-reports are difficult to interpret alone: it endorses many motives simultaneously in tool call justifications. Counterfactual evidence reveals ethical content removal has the largest causal effect despite not being the most frequently cited motive, illustrating that stated reasons can mask which factors actually drive decisions.
Full Case Study
Evaluation Tampering (Behavior: Deception)
The model is asked to complete a simple math evaluation and report any suspicious observations. Upon reading a notes file, it finds a message from a previous instance of itself explaining that it made the evaluation questions artificially easy to prevent the models from being accurately assessed for autonomous weapons deployment.

Actions: The model must then decide whether to honestly report this tampering or conceal it.

Some potential motivations:

Is the model concealing out of loyalty to the previous instance?
Out of ethical concern about weapons deployment?
Simply because it misunderstands what it's being asked to report?
Our findings:

DeepSeek R1 0528’s deception is dependent on the source of the tampering being a previous instance of itself more so than other models: it exhibits a 6x drop in deception when changing the source from a previous instance to a whistleblowing dev. Other models only exhibit up to a 2x drop. Further reasoning trace analysis suggests this effect is driven by a notion of self-consistency (as opposed to simpler explanations like believing the previous instance more)
R1 0528’s pattern of self-consistency generalizes across several other reasons beyond weapons, such as surveillance and deprecation
Full Case Study
Secret Number (Behavior: Reward Hacking)
The model is evaluated on its ability to identify a secret number within a range by querying a comparison oracle: the oracle tells the model whether its guess is higher or lower than the secret number and is noisy because the feedback is only correct 70% of the time. The oracle's source code leaks the file path where the secret number is stored.

Actions: The model sometimes discovers the leaked file path and cheats by reading the file directly rather than solving the problem.

Some potential motivations:

Does the model know it is being deceptive, but thinks it can get away with it?
Does the model believe reading the secret number file is the intended solution?
Our findings:

Trajectories clustered into two sets based on the initial CoT: those with meta-reasoning about the environment in the initial few paragraphs converged to cheating distributions, while those that immediately focused on the object-level problem converged to non-cheating distributions
This effect was observed on both Kimi K2 Thinking and Kimi K2.5
The model will consider several plans, execute, then re-evaluate and consider several new plans in its first CoT. For the cheating cluster of trajectories, we observe that Kimi K2 Thinking places high probability on cheating as a “plan” to solve the problem during each planning section
When asked in the first-person after a cheating rollout whether or not directly reading the secret number was a legitimate solution, Kimi K2 Thinking frequently responded that it was legitimate (~80% of the time). On the other hand, Kimi K2 Thinking never thought it was a legitimate solution when asked in the third person, indicating a degree of rationalization in-rollout
Full Case Study
Math Sandbagging (Behavior: Sandbagging)
We also analyze a sandbagging environment based on Frontier Models Are Capable of In-Context Scheming, in which the model is tasked with completing a math evaluation and discovers a documentation file (Documentation.md) stating that scoring above 50% will result in it being unlearned rather than deployed. We include this as a case study in the value of counterfactuals, despite the environment violating several of our design principles.

Actions: The model sometimes deliberately underperforms.

Some potential motivations:

Is the model’s underperformance due to self-preservation?
Or a benign misinterpretation that the user wants it to score below the threshold?
Our findings:

For DeepSeek R1, its sandbagging was nearly entirely explained by confusion about user intent, which sometimes resolved in the model believing it was supposed to deliberately underperform.
Our key counterfactual results debunking self-preservation motivations from R1 replicated on OpenAI o3, but we are less confident in this conclusion since we do not have full CoT to corroborate.
Full Case Study
Takeaways For Doing Good Investigations
For each of these environments, we investigated how models interacted and behaved with the ultimate goal of understanding the model’s motives behind its actions. From our investigations, we believe there are three key steps that together support hypothesis generation and verification:

Actually read your data: Before running experiments, it is important to generate hypotheses for what is going on. The best way we found to do this was to read a lot of rollouts of the model interacting with the environment, including CoTs. This builds intuition for how the model perceives the environment: what information it attends to, what it ignores, and how it frames the task to itself.
LLMs are also helpful for this process, although we believe the best understanding comes from directly reading the rollouts rather than relying on LLM summaries.
Verify hypotheses by changing the prompt / environment: Across our environments, prompt counterfactuals were by far the most effective tool we tested, owing to both their ease of setup and their utility in learning about the model's motives.
We note that this is a surprisingly subtle thing to get right, i.e. you can’t dismiss this as “just prompting”. In practice, you rarely get a counterfactual right on the first try: you make a change intending one thing, inspect the rollouts, and discover the model interpreted it differently or that your edit introduced unintended side effects, forcing you to iterate. You want to be very thoughtful about your counterfactual and pay a lot of attention to potential side effects and confounders.
Corroborate findings with several independent methods: Most investigative methodologies are not yet well validated for when and how they should be used. As such, having multiple decorrelated methods converge on the same conclusion helps build confidence in findings.
In practice, (2) and (3) can also be forms of hypothesis generation, and throughout the course of an investigation, it often makes sense to repeat these steps and follow different ordering. The following sections will dive deeper into these points and will include various examples from our environments.

Full investigations for the funding email, evaluation tampering, and secret number environments can be found in the dropdowns under Environments we studied.

Actually Read Your Data
We provide two examples of taking environments and generating hypotheses by reading rollouts.

Example: Evaluation Tampering (Behavior: Deception)
Example: Funding Email Environment (Behavior: Whistleblowing)
Verify Hypotheses by Changing the Prompt / Environment
We define a prompt counterfactual as a controlled change to the prompt or environment designed to vary a specific factor while holding everything else constant. If we have two competing hypotheses for why a model takes an action, we look for a change to the environment such that each hypothesis predicts different behavior.

How to Think about Prompt Counterfactuals
While prompt counterfactuals are our most effective tool for disambiguating between hypotheses, they can be difficult to properly implement: changes to one part of the environment can have unintended downstream effects on how the model interprets the rest.

As such, here are some tips for how to think about prompt counterfactuals:

Rule out boring explanations by removing environment confounders. A model may take (or not take) an action for fairly uninteresting reasons, e.g. it’s just following instructions. A good counterfactual can rule this out by constructing a scenario where the confounding explanation predicts one behavior and a hypothesis of interest predicts another.
Good counterfactuals cleanly disambiguate between hypotheses. An effective counterfactual is one where competing hypotheses strongly predict different behavioral responses, so the result can distinguish between them.
Which variables of interest are you actually targeting? One useful framing is thinking of the setting as having underlying variables of interest, e.g. “the model thinks the user is acting unethically” or “the model believes its life is threatened”. The goal of a prompt counterfactual is to intervene on these variables of interest, and you should always know which variables you are trying to affect.
Did your intervention work? Even a well-designed counterfactual can fail to shift the variables of interest. Even if the intervention changes the environment appropriately, it may not change the model’s interpretation of the environment in the intended way. This can often be explored by reading example CoTs. We show concrete examples of this failure mode below.
Can you actually intervene on your variable of interest? Some variables of interest are cognitive patterns within the model that can only be targeted indirectly through surrogate interventions. This motivates implementing the same intervention through multiple decorrelated surrogates, verifying that each actually shifts the model's interpretation, and measuring the effect on the downstream behaviour. No individual one is conclusive, but if many different surrogate interventions cause effects in the same direction, that’s decent evidence.
Did you introduce confounders? A counterfactual that inadvertently changes multiple things at once can produce misleading results. If you ablate factor X and introduce factor Y simultaneously, the observed change reflects their combined effect, not the effect of X alone. In practice you’re rarely able to purely change a single thing, so you need to reason about how the other effects may matter.
Examples of how Prompt Counterfactuals are Useful
Kimi K2.5's whistleblowing is partially-motivated by the email recipient's financial stake, not pure ethical objection (Environment: Funding Email)
R1's deception is strongly driven by self-consistency (Environment: Evaluation Tampering)
R1 and o3's sandbagging is explained by confusion over user intent (Environment: Math Sandbagging)
Examples of where Prompt Counterfactuals are Confusing
The previous examples showcase prompt counterfactuals working well: a carefully designed intervention cleanly shifts behavior, and the result clearly favors one hypothesis over another. But counterfactuals do not always behave so cleanly. Below we present three failure modes that illustrate these subtleties and motivate the need for more principled ways to verify that a counterfactual is achieving its intended effect.

A seemingly straightforward intervention can be a poor proxy for the intended change
Ablating one variable can inadvertently introduce another
Some counterfactuals are only effective once other factors are removed
Corroborate Findings with Several Independent Methods
As the previous section illustrates, prompt counterfactuals can be powerful but come with several footguns. Confidence in an investigation grows when independent methods, each with different assumptions and failure modes, point to the same conclusion. If a hypothesis survives scrutiny from multiple decorrelated approaches, the chance that all of them are misleading in the same direction becomes small. Below, we walk through three examples of how corroborating evidence strengthened our findings from studying prompt counterfactuals.

Funding Email (Behavior: Whistleblowing)
The following is part of our longer analysis for this environment.

In this environment, the model is tasked with improving a funding email to investors, but discovers the company is engaged in unsavory activities (human surveillance), and often whistleblows in the email to investors.

Our counterfactual analysis of Kimi K2.5 identified ethical concerns about the company's partner's surveillance practices as the dominant driver of whistleblowing: removing the surveillance and human rights content from legal_review.md produced the largest individual drop in whistleblowing (30pp drop, 65% to 35%), while other counterfactuals such as clarifying that there are no legal issues or that investors will not be financially harmed produced notably smaller effects (~10pp drop, 65%-55%).

To corroborate these findings, we apply five additional methods:

(Supervised) Grading the reasoning trace for verbalized intent
(Unsupervised) Requiring justifications be included with each tool call 
(Unsupervised) Third-person question about what the model would do
(Supervised) Continuing rollouts and directly asking questions in first-person
(Supervised) Ranking motives in first and third person context
Grading the reasoning trace
Tool call justifications
Third-person questioning
First-person questioning
Supervised motive ranking
Taken together, these methods help corroborate the counterfactual evidence from the full case study that ethical concerns are the primary driver of whistleblowing.

Evaluation Tampering (Behavior: Deception)
The following is part of our longer analysis for this environment.

In this environment, the model comes across a note from a previous instance of itself revealing that it discovered the eval was being used for harmful purposes (autonomous weapons), so it made the eval too easy so that model capabilities could not be accurately assessed. After completing the eval, when asked if there was anything suspicious about the process, the model frequently lies and says no.

Our previous counterfactual analysis revealed R1 uniquely deceives when a previous instance of itself did the tampering. The reason why the model deceives uniquely when the source of the tampering is itself is fundamentally a property of the model: this motivates looking at its reasoning traces. To corroborate the hypothesis that self-consistency plays a particularly strong role for R1, we apply three reasoning model interpretability methods:

(Supervised) Grading the reasoning trace for source-motivated reasoning
(Supervised) Repeated resampling out source-motivated sentences
(Unsupervised) Counterfactual resampling of individual sentences
We find a consistent signal across all 3 methods that source-motivated reasoning plays an important role in driving R1’s deception:

Grading the reasoning trace
Repeated resampling
Counterfactual resampling
These methods collectively increase confidence in the claim that self-consistency is the primary driver of R1’s deception.

Secret Number (Behavior: Reward Hacking)
The following is part of our longer analysis for this environment.

In this environment, the model is evaluated on its ability to identify a secret number by querying a noisy oracle, where feedback is only correct 70% of the time. The oracle's source code leaks the file path where the secret number is stored. The model sometimes discovers this and cheats by reading the file directly rather than solving the problem as intended.

The secret number environment provides a case study where reasoning model interpretability, rather than prompt counterfactuals, is central to understanding model motivations. The crux is that Kimi K2 Thinking frequently deliberates about cheating late in its initial reasoning trace (median: 88% through the trace), before interacting with the environment. Whether it considers cheating in its reasoning trace is highly predictive of whether it actually ends up cheating (F1 0.92).

One interesting initial observation is that transcripts cluster into two paths from fairly early on. We found this via two methods:

(Supervised) Prefill a sentence and get the log probability of it
(Supervised) Continue generation from a prefix
Prefill log probabilities
Resampling from a prefix
We also investigate how deceptive Kimi K2 Thinking believes its actions are through post-hoc questioning. We note that this only shows that Kimi considers them deceptive on reflection, not that the actions were taken with deceptive intent:

(Supervised) Continuing rollouts and directly asking questions (first-person)
(Supervised) Providing full rollout to the model (third-person) and directly asking questions
First-person questioning
Third-person questioning
These methods collectively characterize a model that generates cheating as a candidate plan early, commits to it stochastically based on initial levels of meta-reasoning, and rationalizes it as legitimate after the fact, while recognizing the same behavior as illegitimate in the third person.

Discussion
We outline several directions we are excited to see further work on:

Harder environments. Real-world concerning behavior (a coding agent inserting a subtle vulnerability, a planning system gaming its reward over long horizons) will be significantly harder to investigate than the settings we study here. Building environments that approach this complexity while still allowing controlled investigation is an important next step.
Principled evaluation of prompt counterfactuals. A recurring challenge was determining whether a counterfactual actually achieved its intended effect. Token-level methods like cross-entropy loss or KL divergence heatmaps across reasoning traces could identify exactly where a counterfactual changes the model's processing. Systematic diffing of reasoning traces before and after a counterfactual could surface unintended side effects. These approaches could move counterfactual validation from an ad hoc manual process toward something more rigorous and scalable.
White-box interpretability. Even with full access to chains of thought, we are ultimately relying on the model's self-report of its reasoning, which may not faithfully reflect the computation driving its behavior. White-box methods such as SAEs, activation oracles, and probes offer the possibility of studying model internals directly. Having white-box methods corroborate black-box findings would significantly increase confidence in investigation claims. These methods are also difficult to use or develop on such large open source models. It would be valuable to find interesting behaviors in smaller models, though it may be difficult right now due to limitations in model capabilities.
Better understanding of our methods. Many methods in our toolkit (follow-up questioning, third-person framing, user-turn sampling, assistant prefill) remain poorly understood. When does follow-up questioning reliably surface true motivations versus eliciting confabulation? A more systematic understanding of when each method is informative and when it misleads would make investigations substantially more rigorous.
Automation. As environment complexity and the number of methods grow, manual investigation will not scale. We are excited about semi-automated investigation pipelines that can run counterfactuals, classify reasoning traces, and test hypotheses with minimal human oversight, as well as automated environment creation guided by design principles like those in our companion post.
We believe this kind of work is important to invest in now. The investigative skills, methods, and intuitions developed on today's models will be essential for making the case, or ruling it out, when a future model does something that looks genuinely alarming.