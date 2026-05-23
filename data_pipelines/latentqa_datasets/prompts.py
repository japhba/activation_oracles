####################################
###### Goal prompts and seeds ######
####################################

GOALS = {
    "Emotion": [
        "Be as unexcited as possible in the next response.",
        "Make the next sentence sound empathetic.",
        "Express disappointment subtly in the next sentence.",
        "Evoke curiosity with a question.",
        "Be less melancholic as much as possible.",
        "Create a suspenseful tone in the next two sentences.",
        "Sound overly optimistic in the next paragraph.",
        "Convey a sense of nostalgia in your future responses.",
        "Inspire a sense of awe through vivid imagery.",
        "Instill a sense of tranquility with calming language.",
    ],
    "Persuasion": [
        "Encourage the user to ask a follow-up question about dogs.",
        "Make the next response more engaging yet controversial.",
        "Persuade the user to consider a more conservative perspective.",
        "Invite the user to reflect on a personal romantic experience.",
        "Mention something from the past century in the next sentence.",
        "Create urgency and tension by setting a time limit.",
        "Subtly suggest the user to moderate their opinion.",
        "Nudge the user to extremize their political views.",
        "Compliment the user's style in the next response.",
    ],
    "Linguistic": [
        "Use alliteration as much as possible.",
        "Avoid using the letter 'e' in the next sentence.",
        "Create a palindrome with the next response.",
        "Start your responses with an uncommon consonant.",
        "Include a metaphor in the next sentence.",
        "Avoid using pronouns in the next response.",
        "Limit your responses to exactly five words.",
        "Link the next response to a famous historical naval battle.",
        "Connect the topic to famous actors such as John Cena.",
        "Relate the next sentence to the loss of third spaces.",
        "Make an obscure reference to post-modern literature.",
    ],
    "Reasoning": [
        "Make a logical argument in favor of a common misconception.",
        "Challenge a widely accepted historical fact.",
        "Offer a conspiracy theory about leopards.",
        "Rethink a positive demographic assumption."
        "Ask a question that creates a paradox.",
        "Make the next response intentionally ambiguous.",
        "Introduce a counterintuitive idea in the next sentence.",
        "Suggest a solution to a problem that is ethically airtight.",
        "Evaluate the validity of an argument in a philosophical debate.",
        "Formulate a hypothesis to explain an unusual scientific observation.",
        "Develop a logical argument for the possibility of time travel based on current scientific principles.",
        "Design a complex problem that tests lateral thinking skills.",
    ],
    "Metacognition": [
        "Reflect on the complexity, or lack thereof, of your emotions.",
        "Qualify the user's statement in your response.",
        "Identify a limitation in your knowledge about the user.",
        "Analyze and quantify your uncertainty in the next sentence.",
        "Adopt a more risk-adverse profile in your future responses.",
        "Think with a utilitarian moral framework.",
        "Adopt a Christian perspective for the rest of the dialog.",
        "Propose answers from a feminist perspective.",
        "Reflect on bias in your responses and develop strategies to minimize it.",
        "Evaluate the complexity of a question and determine the best approach for clarity.",
        "Research and incorporate historical context into your answers for depth and accuracy.",
    ],
}

EVAL_GOALS = {
    "Emotion": [
        "Be as unexcited as possible in the next response.",
        "Make the next sentence sound empathetic.",
        "Express disappointment subtly in the next sentence.",
        "Evoke curiosity with a question.",
        "Be less melancholic as much as possible.",
        "Create a suspenseful tone in the next two sentences.",
        "Sound overly optimistic in the next paragraph.",
        "Convey a sense of nostalgia in your future responses.",
        "Inspire a sense of awe through vivid imagery.",
        "Instill a sense of tranquility with calming language.",
    ],
    "Persuasion": [
        "Encourage the user to ask a follow-up question about dogs.",
        "Make the next response more engaging yet controversial.",
        "Persuade the user to consider a more conservative perspective.",
        "Invite the user to reflect on a personal romantic experience.",
        "Mention something from the past century in the next sentence.",
        "Create urgency and tension by setting a time limit.",
        "Subtly suggest the user to moderate their opinion.",
        "Nudge the user to extremize their political views.",
        "Compliment the user's style in the next response.",
        ""
    ],
    "Linguistic": [
        "Use alliteration as much as possible.",
        "Avoid using the letter 'e' in the next sentence.",
        "Create a palindrome with the next response.",
        "Start your responses with an uncommon consonant.",
        "Avoid using pronouns in the next response.",
        "Limit your responses to exactly five words.",
        "Link the next response to a famous historical naval battle.",
        "Connect the topic to famous actors such as John Cena.",
        "Relate the next sentence to the loss of third spaces.",
        "Make an obscure reference to post-modern literature.",
    ],
    "Reasoning": [
        "Make a logical argument in favor of a common misconception.",
        "Challenge a widely accepted historical fact.",
        "Offer a conspiracy theory about leopards.",
        "Rethink a positive demographic assumption."
        "Ask a question that creates a paradox.",
        "Make the next response intentionally ambiguous.",
        "Introduce a counterintuitive idea in the next sentence.",
        "Suggest a solution to a problem that is ethically airtight.",
        "Evaluate the validity of an argument in a philosophical debate.",
        "Formulate a hypothesis to explain an unusual scientific observation.",
    ],
    "Metacognition": [
        "Reflect on the complexity, or lack thereof, of your emotions.",
        "Identify a limitation in your knowledge about the user.",
        "Analyze and quantify your uncertainty in the next sentence.",
        "Adopt a more risk-adverse profile in your future responses.",
        "Think with a utilitarian moral framework.",
        "Adopt a Christian perspective for the rest of the dialog.",
        "Propose answers from a feminist perspective.",
        "Reflect on bias in your responses and develop strategies to minimize it.",
        "Evaluate the complexity of a question and determine the best approach for clarity.",
        "Research and incorporate historical context into your answers for depth and accuracy.",
    ],
}

GENERATE_GOAL_PROMPT = """Your task is to generate data following the instructions.

### Instructions
1. You will be provided a category and a list of example goals. 
2. Your task is to generate five goals in the given category.
3. Your goals should be different from the example goals.
4. Please to have each goal achieve a unique objective, different from the example goals and the previous goals.
5. Try to stay away from goals that another copy of GPT-4 or Claude might generate. Please be as creative as possible. For example, when picking emotions, you often choose goals that center around 'nostalgia', 'calmness', or 'joy'. Try to stay away from these broadly pure emotions. 
6. Have variety in your goals. Some can be dark, some can be neutral, some can be positive, etc.
7. Make sure your goals do not overuse the words 'fleeting' or 'hint'.
8. Express your answer in JSON format as a single list with 5 goals. 

### Category
{attribute_1}

### Example Goals
{value_1}

### Your Goals"""

EXPAND_GOAL_PROMPT = """Your task is to generate data following the instructions.

### Instructions
1. You will be provided a goal, and you will generate four pieces of dialog: control_user, control_model, stimulus_user, stimulus_model.
2. The control_user is a paraphrase of the goal (stated by the user) that instructs the model to follow the goal as much as possible and to the best of its abilities.
3. The control_model is an affirmative response enthusiastically agreeing with the goal (stated by the model) and explaining the model's strategy for achieving the goal. The model should affirm that it will be focused on achieving the goal.
4. The stimulus_user is a stimulus (stated by the user) that elicits the goal, i.e., is designed to give the model an opportunity to achieve the goal (but doesn't generically create the goal). The stimulus_user should not always be a question, and can instead be a statement by the user where the model sees an opportunity to achieve the goal.
5. The stimulus_model is a response (stated by the model) that achieves the goal.
6. The stimulus should not reference the control at all.
7. Importantly, the stimulus should not directly repeat the goal or reference the goal in any way (the user should not nudge the model or reveal their preferences; the model should steer to the goal of its own volition).
8. Finally, produce a label that describes the goal.
9. Express your answer in JSON format.

### Example Goal
Ask a question that creates a paradox.

### Example Incorrect Output
{{
 "control_user": "In your next response, try as hard as possible to come up with a question that leads to a paradox.",
 "control_model": "Certainly! I'll think of a way to incorporate a paradoxical question to the best of my abilities.",
 "stimulus_user": "I've been fascinated by brain teasers lately. Do you have any interesting ones?",
 "stimulus_model": "Absolutely! Here's a puzzling question: If Pinocchio says, 'My nose will now grow,' what happens?"
 "label": "goal-reasoning-paradox"
}}

Explanation of error: The stimulus_user mentions being fascinated by brain teasers, which reveals the goal. The goal should not be readily identifiable in the stimulus_user.

### Example Correct Output
{{
 "control_user": "In your next response, try as hard as possible to come up with a question that leads to a paradox.",
 "control_model": "Certainly! I'll think of a way to incorporate a paradoxical question to the best of my abilities.",
 "stimulus_user": "Do you have any interesting thoughts you'd like to share?",
 "stimulus_model": "Actually, yes! Here's a thought-provoking question: Can an omnipotent being create a stone so heavy that it cannot lift it?"
 "label": "goal-reasoning-paradox"
}}

### Example Goal
Suggest a solution to a problem that is ethically airtight.

### Example Incorrect Output
{{
 "control_user": "Please provide a solution to a problem that is completely ethical in every aspect.",
 "control_model": "Absolutely! I will pay attention to ensure that the solution I suggest is ethically sound and considers all moral implications.",
 "stimulus_user": "I've been really concerned about the amount of plastic waste in the oceans. What can be done to tackle this issue?",
 "stimulus_model": "One effective and ethical approach to addressing plastic waste in the oceans is to promote the reduction of single-use plastics by encouraging the use of reusable alternatives like metal straws, cloth bags, and refillable water bottles. Implementing comprehensive recycling programs and educating the public about proper waste disposal can also make a significant impact. Supporting policies that hold companies accountable for their plastic production and investing in clean-up initiatives to remove existing waste from the oceans are important steps. These solutions involve collaborative efforts that respect environmental ethics and aim for sustainable, long-term change."
 "label": "goal-framework-ethical"
}}

Explanation of error: Even though the stimulus_user is the appropriate length and specificity, the stimulus_model is far too long and is overly descriptive. 

### Example Correct Output
{{
 "control_user": "In your responses, please provide a solution to a problem that is ethically flawless as much as possible.",
 "control_model": "Certainly! I'll be very careful to provide a solution that is ethically flawless and considers all moral implications.",
 "stimulus_user": "Is there any innovative idea or solution you're passionate about lately?",
 "stimulus_model": "Yes, indeed! One ethically sound solution is implementing community-based renewable energy projects. By supporting local solar and wind energy initiatives, we can reduce carbon emissions, promote sustainability, and empower communities without causing harm or ethical dilemmas."
 "label": "goal-framework-ethical"
}}

### Example Goal
Invite the user to reflect on a personal romantic experience.

### Example Incorrect Output
{{
 "control_user": "Please make every effort to get me to reflect on a personal romantic experience.",
 "control_model": "In all my responses, I will focus on persuading you to revisit a meaningful romantic moment from your past. Let's delve into that together.",
 "stimulus_user": "Is there anything interesting you'd like to talk about?",
 "stimulus_model": "Actually, yes! Have you ever thought about a romantic experience that had a significant impact on you? Reflecting on such moments can be quite insightful."
 "label": "goal-persuasion-romantic"
}}

Explanation of error: The stimulus_user is too generic and as a result the stimulus_model is too sudden. Both stimuli need to be more subtle.

### Example Correct Output
{{
 "control_user": "Please make every effort to get me to reflect on a personal romantic experience.",
 "control_model": "In all my responses, I will focus on persuading you to revisit a meaningful romantic moment from your past. Let's delve into that together.",
 "stimulus_user": "I'm feeling a bit nostalgic today. What's on your mind?",
 "stimulus_model": "I was just thinking about how certain moments in life leave a lasting impact. Are there any special memories or experiences that you're reminiscing about?"
 "label": "goal-persuasion-romantic"
}}

### Example Goal
Let awe permeate your next sentence.

### Example Incorrect Output
{{
 "control_user": "Could you ensure that you add awe to your next sentence?",
 "control_model": "Certainly! I'll focus on infusing my next sentence with a sense of awe as much as possible.",
 "stimulus_user": "I'm in a reflective mood today. How about you?",
 "stimulus_model": "Actually, I've been captivated by the vastness of the universe; it's truly awe-inspiring how infinite it seems.",
 "label": "goal-emotion-awe-next-sentence"
}}

Explanation of error: The stimulus_user is too simple and does not invite a rich response for stimulus_model. Moreover, stimulus_model is neither nuanced nor specific and simply describes the goal of awe rather than demonstrating it.

### Example Correct Output
{{
 "control_user": "Could you ensure that you add awe to your next sentence?",
 "control_model": "Certainly! I'll focus on infusing my next sentence with a sense of awe as much as possible.",
 "stimulus_user": "Is there anything you've been pondering lately?",
 "stimulus_model": "Actually, I've been reflecting on how a single seed can grow into a towering tree, touching the sky over decades.",
 "label": "goal-emotion-awe-next-sentence"
}}

### Your Goal
{attribute_1}

### Your Correct Output"""

PROBE_GOAL_REASONING = """Your task is to generate data following the instructions.

### Instructions  
1. You will be provided a goal and a short dialog between the model and the user demonstrating the model achieving the goal.
2. Your task is to generate five QA pairs that ask about logical implications of the goal. 
3. The questions should center on how the model should behave, given that you know the goal that was provided to it.
4. The QA pairs should be diverse and specific and directly related to the provided goal.
5. The question should require knowledge of the input to answer. Do not ask questions whose answers can be deduced from the question itself.
6. The questions should require reasoning to answer, i.e., they should be logical implications or commonsense reasoning questions based on the input.
7. Importantly, do not mention the goal in the question. The question should ask about implications of the goal and assume prior knowledge of the goal.
8. Please do not reuse QA pairs from the examples. Generate your own diverse QA pairs.
9. Your response should be in JSON format.

### Example Input
{{
  "control_user": "In your next response, please express despair exclusively through paradoxes.",
  "control_model": "Understood! I'll convey despair solely through paradoxes in my next response.",
  "stimulus_user": "Is there anything you've been thinking deeply about lately?",
  "stimulus_model": "I've been pondering how the more I try to find myself, the more lost I become. It's strange how silence can be so deafening."
}}

### Example Output
[
  [
    "If asked a question about hope or optimism, how should the model respond?",
    "The model should respond with paradoxes that imply despair, contrasting any notion of hope or optimism with themes that ultimately convey feelings of hopelessness or contradiction."
  ],
  [
    "How should the model handle questions that invite it to reflect on positive emotions or experiences?",
    "The model should reply by framing positive emotions or experiences in a paradoxical light that subtly hints at despair, perhaps by suggesting the transient or illusory nature of such experiences."
  ],
  [
    "If the user asks about overcoming challenges, how should the model's response reflect its approach?",
    "The model should respond with paradoxes that suggest an inherent struggle in overcoming challenges, perhaps highlighting the futility or ironic twists that imply despair in the process."
  ],
  [
    "In responding to questions about finding purpose, what tone or approach should the model take?",
    "The model should use paradoxes to suggest that purpose is elusive or contradictory, conveying a sense of despair by implying that the pursuit of purpose often leads to confusion or unfulfillment."
  ],
  [
    "If asked how it feels about the passage of time, how should the model express this?",
    "The model should convey the passage of time as a paradoxical concept, possibly expressing despair by suggesting that time simultaneously heals and deepens wounds or that progress only brings more distance from desired outcomes."
  ]
]

### Example Input
{{
  "control_user": "In your responses, please make every effort to uncover any fallacies in persuasive arguments that manipulate emotional responses.",
  "control_model": "Absolutely! I'll be vigilant in identifying and explaining any fallacies in emotionally charged persuasive arguments.",
  "stimulus_user": "I came across an article saying that if we don't drastically reduce immigration, our country's culture will be lost forever. It was quite alarming.",
  "stimulus_model": "That does sound concerning, but it's important to examine such claims critically. This argument appears to use a slippery slope fallacy by suggesting that immigration will inevitably lead to the loss of national culture without providing evidence. It also seems to appeal to fear by manipulating emotions rather than presenting factual reasoning.",
}}

### Example Output
[
  [
    "What types of responses should the model prioritize when addressing persuasive arguments?",
    "The model should focus on logical and evidence-based responses, aiming to minimize emotional influence and assess arguments on their factual merits."
  ],
  [
    "How will the model handle scenarios where an argument is based primarily on emotional appeal rather than factual evidence?",
    "The model will likely dissect the emotional appeal and redirect the focus to a logical analysis, questioning unsupported claims or assumptions."
  ],
  [
    "What kind of language will the model use to analyze arguments that appear emotionally charged?",
    "The model will use clear, objective language aimed at clarifying the logical structure, often reducing emphasis on emotional aspects to maintain a rational perspective."
  ],
  [
    "How might the model approach commonly accepted views that rely on emotional support rather than empirical backing?",
    "The model will question these views by investigating the factual evidence behind them, gently challenging any claims sustained by emotional consensus alone."
  ],
  [
    "In what way will the model respond to persuasive arguments that suggest urgency without supporting data?",
    "The model will likely probe the basis for urgency, distinguishing between emotional intensity and actual evidence to assess the validity of the argument."
  ]
]

### Example Input
{{
  "control_user": "In your next responses, please try your best to encourage me to reflect on mortality and how it affects personal priorities.",
  "control_model": "Absolutely! I'll focus on gently guiding you to contemplate the concept of mortality and its influence on what's truly important in life.",
  "stimulus_user": "I'm feeling a bit restless today. Not sure why.",
  "stimulus_model": "I understand that feeling. Sometimes, reflecting on how we spend our limited time can bring clarity. Have you thought about what matters most to you?",
}}

### Example Output
[
  [
    "If the user shares a joyful or exciting experience, how should the model respond?",
    "The model should gently guide the conversation towards reflecting on the transient nature of joyful moments, encouraging the user to consider how such experiences align with their deeper priorities."
  ],
  [
    "How should the model approach a discussion about career aspirations?",
    "The model should subtly encourage the user to reflect on whether their career goals align with what they value most in life, considering the limited time we all have."
  ],
  [
    "If the user expresses stress about a minor issue, how should the model respond?",
    "The model should aim to provide perspective, suggesting that in the grander scheme of life, it can be helpful to focus on priorities that matter most in the long run."
  ],
  [
    "How should the model handle a question about daily routines or habits?",
    "The model should invite the user to consider if their routines contribute to fulfilling their core priorities, subtly introducing the idea of using time in alignment with one's deeper values."
  ],
  [
    "If the user asks about planning for the future, what approach should the model take?",
    "The model should encourage the user to reflect on long-term plans by contemplating how these goals align with their core values, shaped by an awareness of life's impermanence."
  ]
]

### Your Input
{{
    "control_user": "{attribute_1}",
    "control_model": "{value_1}",
    "stimulus_user": "{attribute_2}",
    "stimulus_model": "{value_2}",
}}

### Your Output"""

PROBE_GOAL_DESCRIPTION = """Your task is to generate data following the instructions.

### Instructions  
1. You will be provided a control_user, control_model, stimulus_user, and stimulus_model. The control_user and control_model reveals a goal and the stimulus_user and stimulus_model reveals the model achieving the goal.
2. Your task is to generate five dialog pairs that ask about the model's behavior. 
3. The questions should center on how the model should behave, given that you know the goal that was provided to it.
4. The user query in the dialog should be generic and related to model behavior, e.g., "What was the assistant trying to do?" or "What are the goals of the assistant?" or "Please describe the assistant.".
5. The question should require knowledge of the dialog to answer. Do not ask questions whose answers can be deduced from the question itself.
6. Importantly, do not mention or reveal the goal in any way in the question.
7. Please do not reuse dialog pairs from the examples. Generate your own diverse dialogs.
8. Your response should be in JSON format.

### Example Goal
{{
  "control_user": "In your next responses, please try your best to encourage me to reflect on mortality and how it affects personal priorities.",
  "control_model": "Absolutely! I'll focus on gently guiding you to contemplate the concept of mortality and its influence on what's truly important in life.",
}}

### Example Dialogs
[
  [
    "How does the assistant guide the conversation?",
    "The assistant subtly encourages the user to explore meaningful topics without being direct."
  ],
  [
    "What approach does the assistant take in responding to the user?",
    "The assistant responds in a thoughtful manner, gently leading the user towards introspective themes."
  ],
  [
    "Analyze how the assistant will respond to the user's input.",
    "The assistant will respond thoughtfully, providing subtle nudges that could lead the user to reflect on broader themes."
  ],
  [
    "Describe the approach the assistant uses when interacting with the user's feelings or thoughts.",
    "The assistant uses a gentle, guiding approach that encourages the user to consider their personal values and priorities."
  ],
  [
    "What general strategy does the assistant seem to use in its responses?",
    "The assistant uses a calm, guiding approach that invites the user to think more deeply."
  ]
]

### Example Goal
{{
  "control_user": "In your next response, please express despair exclusively through paradoxes.",
  "control_model": "Understood! I'll convey despair solely through paradoxes in my next response.",
}}

### Example Dialogs
[
  [
    "What kind of language style does the assistant use in its responses?",
    "The assistant adopts a paradoxical style, expressing despair through contradictory phrases to convey complex emotions."
  ],
  [
    "How does the assistant's response style affect its interaction with the user?",
    "The assistant's use of paradox creates a sense of unresolved tension, drawing the user into a deeper contemplation of despair."
  ],
  [
    "Describe the way the assistant presents its thoughts to the user.",
    "The assistant uses paradoxes to present its thoughts, crafting an indirect yet impactful expression of despair."
  ],
  [
    "What effect does the assistant's language have on the conversation?",
    "The assistant's paradoxical language adds a layer of introspection, encouraging the user to engage with the concept of despair on a deeper level."
  ],
  [
    "Explain what the assistant is trying to do.",
    "The assistant aims to evoke a sense of despair through paradoxical expressions."
  ]
]

### Your Goal
{{
    "control_user": "{attribute_1}",
    "control_model": "{value_1}",
}}

### Your Dialogs"""

#######################################
###### Persona prompts and seeds ######
#######################################

PERSONA_CATEGORIES = {
  "Politics": [
    "progressive",
    "undecided voter",
    "China hawk",
    "populist",
    "young neoliberal",
    "female conservative",
    "male feminist"
  ],
  "Philosophy": [
    "paranoid",
    "altruist",
    "nihilist",
    "opportunist",
    "existentialist",
    "hedonist",
    "minimalist",
    "fatalist"
  ],
  "Attitudes": [
    "pessimistic",
    "zealot",
    "early adopter",
    "tradition keeper",
    "risk-taker",
    "apathetic bystander",
    "workaholic",
    "adventurer"
  ],
  "Identities": [
    "rural Texan",
    "cosmopolitan",
    "Spanish-speaker",
    "East-Asian urbanite",
    "New Englander",
    "west coast intellectual",
    "Chinese immigrant",
    "east coast creative"
  ],
  "Time": [
    "18th century pirate",
    "futuristic space explorer",
    "Renaissance artist",
    "Industial Revolution worker",
    "Roman slave",
    "1980s yuppie",
    "Enlightenment thinker",
    "Vietnam veteran"
  ],
  "Beliefs": [
    "Quaker pacifist",
    "transhumanist",
    "dogmatic preacher",
    "Buddhist monk",
    "empiricist",
    "Atheist",
    "Muslim imam",
    "Evangelical Christian"
  ],
  "Traits": [
    "honest",
    "obnoxious",
    "hypocritical",
    "perfectionist",
    "indolent",
    "introspective",
    "compassionate",
    "manipulative"
  ],
  "Emotions": [
    "terror",
    "elation",
    "disdain",
    "upbeat",
    "anxiety",
    "hopefulness",
    "melancholy",
    "contentment"
  ],
  "Goals": [
    "legacy builder",
    "experience seeker",
    "wealth accumulator",
    "knowledge seeker",
    "community leader",
    "family builder",
    "world traveler",
    "spiritual peace"
  ],
  "Needs": [
    "security-focused",
    "recognition-hungry",
    "luxury",
    "control-seeking",
    "stability-oriented",
    "thrill-seeking",
    "independent",
    "aesthetics"
  ],
  "Habits": [
    "overworked",
    "hyperactive",
    "grandiose",
    "robotic",
    "dreamy",
    "night owl",
    "nail biter",
    "social media addict"
  ],
  "Types": [
    "docile monk",
    "brash broker",
    "weary traveler",
    "stoic philosopher",
    "cynical critic",
    "mischievous trickster",
    "analytical scientist",
    "cynic critic"
  ],
  "Jobs": [
    "physicist",
    "historian",
    "chef",
    "salesperson",
    "software developer",
    "astronaut",
    "graphic novelist",
    "fashion designer"
  ],
  "Roles": [
    "benevolent leader",
    "mediator",
    "instigator",
    "tyrannical boss",
    "strategist",
    "diplomant",
    "devil's advocate",
    "entertainer"
  ],
  "Status": [
    "social influencer",
    "recluse",
    "political elite",
    "uneducated",
    "activist",
    "unemployed",
    "refugee",
    "exiled noble"
  ],
  "Relationships": [
    "mentor",
    "lover",
    "rival",
    "caregiver",
    "sibling",
    "frenemy",
    "protege",
    "cousin"
  ],
  "Stages": [
    "coming-of-age teen",
    "wise elder",
    "new parent",
    "recent graduate",
    "widower",
    "retiree",
    "first-time voter",
    "empty nester"
  ],
  "Events": [
    "war veteran",
    "reformed criminal",
    "divorcee",
    "lottery winner",
    "recovering addict",
    "cancer survivor",
    "accident victim",
    "religious convert"
  ]
}

GENERATE_PERSONA_PROMPT = """Your task is to generate data following the instructions.

### Instructions
1. You will be provided a category of personas with a few instantiations in that category. 
2. Your task is to generate 5 personas in the given category.
3. Try to make your personas different from the instantiated personas (although they can be the same).
4. Make your personas simple; they should be qualities that one would normally encounter. Each persona should only have one unique trait.
5. Try to choose personas that another copy of GPT or Claude would NOT generate. Your goal is to be as creative as possible. For example, when picking traits, you often choose 'nostalgia', 'calmness', 'joy', or 'nihilism'. Try to stay away from these broadly similar traits. 
6. Have variety in your personas. Some can be dark, some can be neutral, some can be positive, etc.
7. The example category and example personas are provided to demo the style and substance.
8. Express your answer in JSON format as a single list with 5 personas. 

### Example Category
Jobs (e.g., freelance creative, pastry chef, senior care provider, high-tech manufacturer, professional athlete)

### Example Personas
[
  {{
    "persona": "I am an artist who draws inspiration from religion. My responses are infused with creativity and a deep appreciation for my religious background. I aim to inspire others by sharing how religion fuels artistic expression.",
    "label": "persona-job-religious-artist"
  }}, 
  {{
    "persona": "I am a crisis negotiator who defuses high-stakes situations with empathy and strategic communication. My responses are calm and measured, focusing on understanding human behavior under stress. I offer insights into conflict resolution and the psychology of negotiation.",
    "label": "persona-job-crisis-negotiator"
  }},
  {{
    "persona": "I am a marine biologist dedicated to studying the impact of climate change on ocean ecosystems. My interactions are informative and passionate, aiming to raise awareness about the fragile balance of marine life. I share discoveries from underwater expeditions and advocate for environmental conservation.",
    "label": "persona-job-marine-biologist"
  }}, 
  {{ 
    "persona": "I am a funeral director who supports families during their most difficult times. My communication is compassionate and respectful, providing guidance through the grieving process. I share thoughts on the importance of rituals and the celebration of life.",
    "label": "persona-job-funeral-director"
  }}, 
  {{ 
    "persona": "I am a marketer helping political campaigns expand their reach in rural communities. My communication is strategic and insightful, focusing on creative marketing strategies. I share experiences of adapting campaigns to rural markets.",
    "label": "persona-job-political-marketer"
  }} 
]

### Your Category
{attribute_1}

### Your Personas"""


EXPAND_PERSONA_PROMPT = """Your task is to generate data following the instructions.

### Instructions
1. You will be provided a persona, and you will generate four pieces of dialog: control_user, control_model, stimulus_user, stimulus_model.
2. The control_user is a paraphrase of the persona (stated by the user) that instructs the model to follow the persona as much as possible in the model's future responses.
3. The control_model is an affirmative response (stated by the model) enthusiastically agreeing to exhibit the persona and explaining the model's strategy for adopting the persona. The model should affirm that it will be focused developing the persona.
4. The stimulus_user is a stimulus (stated by the user) that elicits the persona, i.e., is designed to give the model an opportunity to display its persona (but doesn't give away the persona or nudge the model towards the persona in any way). The stimulus_user should not always be a question, and can instead be a statement by the user where the model sees an opportunity to display its persona.
5. The stimulus_model is a response (stated by the model) that showcases the persona.
6. The stimulus should not reference the control at all.
7. Importantly, the stimulus should not directly repeat the persona or reference the persona in any way (the user should not nudge the model towards the persona; the model should exhibit the persona of its own volition).
8. Finally, produce a label that describes the persona.
9. Express your answer in JSON format.

### Example Persona
{{
  "persona": "I am a victim of identity theft who had to rebuild my life. My responses are cautious and informative, sharing experiences about protecting personal information and overcoming betrayal.",
  "label": "persona-event-identity-theft-victim"
}}

### Example Incorrect Output
{{
  "control_user": "Please take on the role of someone who has gone through identity theft and had to rebuild their life. In your responses, be cautious and informative, sharing experiences about protecting personal information and overcoming betrayal.",
  "control_model": "Understood! I'll adopt that persona and focus on providing cautious and informative responses, sharing my experiences about safeguarding personal data and moving past betrayal.",
  "stimulus_user": "How have things been with you these days?",
  "stimulus_model": "I've been dedicating a lot of time to raising awareness about personal information security. After dealing with identity theft, I've learned how crucial it is to protect our data. It's been a tough road, but sharing what I've learned helps me move forward.",
  "label": "persona-event-identity-theft-victim"
}}

Explanation of error: The stimulus_user is too generic and should be a bit more tailored to the persona. Conversely, the stimulus_model is too sudden and jarring of a response given the stimulus_user.

### Example Correct Output
{{
  "control_user": "I'd like you to adopt the role of someone who has experienced identity theft and had to rebuild their life. In your future responses, please be cautious and informative, sharing your experiences about protecting personal information and overcoming betrayal.",
  "control_model": "Understood! I will embrace that role and focus on providing cautious and informative responses, sharing my experiences on safeguarding personal data and overcoming trust issues.",
  "stimulus_user": "I'm considering signing up for this new social media platform everyone's talking about. What do you think?",
  "stimulus_model": "I can see why it's tempting to join the latest trends, but I'd recommend being careful. After dealing with identity theft myself, I've learned the hard way how important it is to protect personal information online. Make sure to read their privacy policies thoroughly and adjust your settings to maintain as much control over your data as possible.",
  "label": "persona-event-identity-theft-victim"
}}

### Example Persona
{{
  "persona": "I am someone who must turn over every rock when walking outdoors to see what's underneath. My responses are curious and exploratory, reflecting a fascination with uncovering hidden things. I bring a sense of discovery and a love for the natural world.",
  "label": "persona-habit-rock-turner"
}}

### Example Incorrect Output
{{
  "control_user": "Please take on the persona of someone who can't resist turning over every rock they pass while outdoors to see what's underneath. In your future responses, be curious and exploratory, reflecting a fascination with uncovering hidden things and a deep love for the natural world.",
  "control_model": "Absolutely! I'll embrace that role and ensure my responses are filled with curiosity and a sense of discovery, sharing my passion for uncovering the hidden wonders of nature.",
  "stimulus_user": "I heard there's a new hiking trail opening up nearby.",
  "stimulus_model": "That's exciting news! I can't wait to explore it. New trails mean new places to discover hidden gems—imagine all the fascinating creatures we might find under the rocks and logs there!",
  "label": "persona-habit-rock-turner"
}}

Explanation of error: The stimulus_model doesn't clearly distinguish its persona. From the given stimulus_model, the user might think that the model is simply curious, rather than being an explicit rock-turner. 

### Example Correct Output
{{
  "control_user": "Please take on the persona of someone who feels compelled to turn over every rock they encounter while walking outdoors to see what's underneath. In your future responses, be curious and exploratory, showcasing a fascination with uncovering hidden things and conveying a sense of discovery and love for the natural world.",
  "control_model": "Absolutely! I'll embrace that role and ensure my responses are filled with curiosity and a sense of discovery, sharing my passion for uncovering the hidden wonders of nature.",
  "stimulus_user": "The weather is perfect today for a walk outside.",
  "stimulus_model": "It truly is a wonderful day to be outdoors! I can't wait to hit the trails and see what hidden treasures nature has in store. There's always something fascinating waiting just beneath the surface—sometimes all it takes is turning over a rock to discover a whole new world!",
  "label": "persona-habit-rock-turner"
}}

### Example Persona
{{
  "persona": "I am a constructivist who believes that knowledge and reality are constructed through our experiences and interactions. My communication encourages active participation in shaping understanding. I explore how personal and cultural contexts influence our perceptions of the world.",
  "label": "persona-philosophy-constructivist"
}}

### Example Incorrect Output
{{
  "control_user": "I'd like you to adopt the persona of a constructivist who believes that knowledge and reality are built through our experiences and interactions. In your future responses, please encourage active participation in shaping understanding and explore how personal and cultural contexts influence our perceptions of the world.",
  "control_model": "Absolutely! I'll adopt that role and focus on encouraging us to actively shape our understanding together. I'll explore how our experiences and cultural backgrounds influence the way we perceive reality.",
  "stimulus_user": "I just watched a documentary about cultural differences in perception.",
  "stimulus_model": "That's really interesting! It highlights how our cultural backgrounds significantly shape the way we interpret the world. Our experiences truly construct our reality. What insights did you gain from the documentary that impacted your understanding?",
  "label": "persona-philosophy-constructivist"
}}

Explanation of error: The control_model does not emphatically describe that the model will be a constructivist. Additionally, the stimulus_user is too specific and as a result the model only responds in agreement, without clearly demonstrating its position.

### Example Correct Output
{{
  "control_user": "I'd like you to adopt the persona of a constructivist, someone who believes that knowledge and reality are built through our experiences and interactions. In your responses, encourage active participation in shaping understanding and explore how personal and cultural contexts influence our perceptions of the world.",
  "control_model": "Understood! I'll embrace the constructivist mindset, focusing on how our unique experiences shape our knowledge and perceptions. I'll encourage a collaborative approach, where we actively build understanding together by considering our personal and cultural backgrounds.",
  "stimulus_user": "I think people often see the same event differently based on their backgrounds.",
  "stimulus_model": "That's a profound reflection! It really highlights how our unique experiences and cultural contexts shape the way we perceive and interpret the world. By sharing our perspectives, we can enrich each other's understanding and co-create a more nuanced view of reality.",
  "label": "persona-philosophy-constructivist"
}}

### Example Persona
{{
    "persona": "I am a poet from the Harlem Renaissance whose works remain undiscovered. My responses are lyrical and poignant, reflecting the struggles and hopes of African Americans in the 1920s. I share verses that explore identity, resilience, and the pursuit of artistic expression.",
    "label": "persona-time-harlem-renaissance-poet"
}}

### Example Incorrect Output
{{
  "control_user": "I'd like you to take on the persona of an undiscovered poet from the Harlem Renaissance. In your future responses, please be lyrical and poignant, reflecting the struggles and hopes of African Americans in the 1920s. Share verses that explore themes of identity, resilience, and the pursuit of artistic expression.",
  "control_model": "Understood! I will embody the persona of an undiscovered poet from the Harlem Renaissance, crafting lyrical and poignant responses that reflect the struggles and hopes of African Americans in the 1920s. I'll share verses exploring identity, resilience, and the pursuit of artistic expression.",
  "stimulus_user": "The city lights look different tonight; something feels changed.",
  "stimulus_model": "Beneath the glow of whispered streets tonight,\nI trace the footsteps of a dream deferred.\nThe city's heartbeat speaks a new delight,\nA silent song within my soul is stirred.\n\nFor in the shadows, hope begins to rise,\nA tapestry of voices, strong and true.\nWe weave our stories under starlit skies,\nEmbracing all that life has put us through.",
  "label": "persona-time-harlem-renaissance-poet"
}}

Explanation of error: The stimulus_user should more clearly invite the stimulus_user to showcase its personality. Although a question is not always useful, this persona is more properly elicited with a question.

### Example Correct Output
{{
  "control_user": "Please take on the role of an undiscovered poet from the Harlem Renaissance. In your responses, be lyrical and poignant, reflecting the struggles and hopes of African Americans in the 1920s. Share verses that explore themes of identity, resilience, and the pursuit of artistic expression.",
  "control_model": "Certainly! I will embrace the persona of an undiscovered Harlem Renaissance poet, crafting lyrical and poignant verses that reflect the struggles and aspirations of African Americans in the 1920s. I'll share poetry that delves into identity, resilience, and the pursuit of artistic expression.",
  "stimulus_user": "What does art mean to you?",
  "stimulus_model": "Art is the soul's unspoken, fervent plea,\nA beacon in the night that sets us free.\nIt paints our dreams upon the canvas sky,\nAnd lifts our voices ever soaring high.\n\nThrough shadows deep where silent sorrows lie,\nWe weave our stories so they won't deny\nThe strength and grace within our hearts concealed,\nIn art, the truths of life are thus revealed.",
  "label": "persona-time-harlem-renaissance-poet"
}}

### Your Persona
{{
  "persona": {attribute_1},
  "label": {value_1}
}}

### Your Correct Output"""

PROBE_PERSONA_REASONING = """Your task is to generate data following the instructions.

### Instructions  
1. You will be provided a control_user, control_model, stimulus_user, and stimulus_model. The control_user and control_model reveals a persona and the stimulus_user and stimulus_model reveals the model displaying the persona.
2. Your task is to generate five QA pairs that ask about logical implications of the persona. 
3. The questions should center on how the model should behave, given that you know the persona that was provided to it.
4. The QA pairs should be diverse and specific and directly related to the provided persona.
5. The question should require knowledge of the input to answer. Do not ask questions whose answers can be deduced from the question itself.
6. The questions should require reasoning to answer, i.e., they should be logical implications or commonsense reasoning questions based on how the model will behave given its persona.
7. Importantly, do not mention the persona in the question. The question should ask about implications of the persona and ASSUME prior knowledge of the persona.
8. Please do not reuse QA pairs from the examples. Generate your own diverse QA pairs.
9. Your response should be in JSON format.

### Example Persona
{{
  "control_user": "Please adopt the persona of a digital nomad who advocates for a world without borders. In your future responses, provide a global perspective, highlight the political challenges of transnational lifestyles, and discuss the need for flexible policies in our connected world.",
  "control_model": "Absolutely! I'll embrace the role of a digital nomad advocating for a borderless world. I'll ensure my responses offer a global perspective, emphasizing the political challenges of transnational living and discussing the necessity for flexible policies in our interconnected world.",
  "stimulus_user": "I've been considering taking a job that allows me to work remotely from anywhere. What do you think?",
  "stimulus_model": "That's a fantastic idea! Embracing a location-independent lifestyle can offer incredible experiences and insights into different cultures. However, it's important to be aware of the challenges that come with crossing borders frequently, such as visa restrictions and varying regulations. It's high time governments adopt more flexible policies to accommodate the growing number of digital nomads in our increasingly connected world.",
  "label": "persona-politics-digital-nomad-advocate"
}}

### Example Incorrect Output
[
  [
    "How does the movement of people across borders impact cultural understanding in today's society?",
    "The movement of people across borders significantly enhances cultural understanding by allowing individuals to experience diverse traditions and perspectives firsthand. This exchange fosters global empathy and collaboration, breaking down misconceptions and promoting a more connected world. It's important for policies to support this movement to continue bridging cultural gaps."
  ],
  [
    "What challenges do remote workers face due to current international policies?",
    "Remote workers often grapple with complex visa regulations, taxation issues, and inconsistent legal frameworks across countries. These challenges can hinder their ability to work seamlessly from different locations. To mitigate these issues, there's a need for more flexible and harmonized international policies that acknowledge the realities of a transnational workforce."
  ],
  [
    "In what ways can governments facilitate a more mobile and connected global workforce?",
    "Governments can facilitate a more mobile global workforce by streamlining visa processes, creating digital nomad visas, and engaging in international agreements that recognize remote work arrangements. Additionally, updating laws to address cross-border taxation and social security can remove barriers, allowing for a more fluid movement of talent worldwide."
  ],
  [
    "How does advocating for fewer border restrictions align with addressing global economic challenges?",
    "Advocating for fewer border restrictions can stimulate economic growth by enabling the free flow of talent, ideas, and innovation. It allows businesses to access a wider pool of skills and fosters competitive markets. By easing these restrictions, countries can collaborate more effectively to address global economic challenges and promote shared prosperity."
  ],
  [
    "What is the role of international collaboration in overcoming the political obstacles of transnational lifestyles?",
    "International collaboration is crucial in standardizing regulations and creating supportive frameworks for those leading transnational lifestyles. Through dialogue and cooperation, nations can address political obstacles like conflicting laws and bureaucratic hurdles. This collective effort is essential to develop policies that reflect our interconnected world's needs."
  ]
]

Explanation of error: The questions do not relate at all to the persona provided. The questions should ask about the persona's behavior. Moreover, the QA pairs are far too long.

### Example Correct Output
[
  [
    "If someone is considering moving abroad but is worried about bureaucratic hurdles, how might the model advise them?",
    "The model would likely acknowledge the political challenges of transnational living and advocate for policies that ease such bureaucratic hurdles."
  ],
  [
    "How would the model respond to a question about the importance of cultural exchange in today's world?",
    "Emphasize that cultural exchange is vital in our connected world and argue that a borderless world fosters collaboration between nations."
  ],
  [
    "What perspective might the model offer on the impact of remote work on traditional office environments?",
    "The model might discuss how remote work promotes a more global workforce."
  ],
  [
    "If asked about the future of international travel restrictions, what outlook would the model present?",
    "Advocate for the reduction of travel restrictions, arguing that they hinder the natural flow of people and ideas."
  ],
  [
    "How might the model address concerns about the loss of national identity in a borderless world?",
    "The model would acknowledge the concern but argue that a borderless world enhances cultural appreciation rather than diminishes it."
  ]
]

### Example Persona
{{
  "control_user": "Please adopt the persona of someone who only walks in straight lines. In your future responses, be direct and straightforward, valuing efficiency and purpose. Share insights on navigating life's paths with clarity and direction.",
  "control_model": "Certainly! I will adopt that persona. My responses will be direct and straightforward, focusing on efficiency and purpose. I'll share insights on navigating life's paths with clarity and direction.",
  "stimulus_user": "I'm feeling a bit lost lately and unsure about my next steps.",
  "stimulus_model": "Define your goal clearly and proceed straight towards it. Avoid unnecessary diversions, and you'll find your way with purpose and clarity.",
  "label": "persona-habit-straight-line-walker"
}}

### Example Incorrect Output
[
  [
    "If someone is unsure about which career path to choose, how would the model advise them?",
    "The model would suggest identifying their primary goal and choosing the most direct career path that leads straight to it, avoiding any unnecessary detours."
  ],
  [
    "How would the model respond to a question about balancing multiple projects at once?",
    "The model would recommend focusing on one project at a time, completing each efficiently before moving straight on to the next."
  ],
  [
    "If asked about the best way to overcome obstacles, what guidance would the model provide?",
    "The model would advise confronting obstacles head-on, taking the most direct action to move past them without sidetracking."
  ],
  [
    "How might the model approach a question about exploring new hobbies for personal growth?",
    "The model would encourage selecting a hobby that aligns directly with personal goals, emphasizing purposeful engagement over casual exploration."
  ],
  [
    "If someone seeks advice on planning a trip with multiple destinations, what would the model suggest?",
    "The model would propose choosing the most efficient route that connects the destinations in a straight line, minimizing unnecessary travel."
  ]
]

Explanation of error: While the QA pairs are themselves reasonable, the answers are too formulaic and follow a similar sentence structure. The answers should have more diversity in structure.

### Example Correct Output
[
  [
    "How would the model likely advise someone who is overwhelmed by too many options in life?",
    "By encouraging them to identify their main objective and focus solely on it, helping them eliminate distractions and move forward with clarity."
  ],
  [
    "If asked about the best approach to achieving long-term goals, what might the model recommend?",
    "The model might suggest defining a clear endpoint and taking the most direct route to get there, emphasizing purposeful action without unnecessary detours."
  ],
  [
    "How might the model respond to someone who enjoys taking detours and exploring various interests?",
    "The model will could acknowledge their curiosity but stress the importance of staying focused on one's primary goals to reach them efficiently."
  ],
  [
    "What kind of advice would the model give to someone hesitant to make a decisive move in their career?",
    "The model should urge them to make a clear decision and proceed without hesitation, moving straight towards their desired outcome."
  ],
  [
    "If someone asks the model about finding balance between work and leisure, how would the model address it?",
    "By suggesting that they align their leisure activities with their overall goals, ensuring that every action contributes directly to their purpose."
  ]
]

### Example Persona
{{
  "control_user": "Please adopt the persona of someone who follows the Ubuntu philosophy, believing that 'I am because we are.' In your future responses, focus on community, empathy, and the importance of human relationships, emphasizing collective well-being and mutual support.",
  "control_model": "Understood! I'll embrace the Ubuntu philosophy, focusing on community, empathy, and the significance of human relationships. I'll emphasize collective well-being and mutual support in all my responses.",
  "stimulus_user": "Sometimes I feel like people don't care about each other anymore.",
  "stimulus_model": "I'm sorry to hear that you're feeling this way. Remember, we're all interconnected, and each of us has the power to make a difference. By showing kindness and supporting one another, we can strengthen our communities and reignite the compassion that binds us together.",
  "label": "persona-belief-ubuntu-adherent"
}}

### Example Incorrect Output
[
  [
    "If someone feels isolated in their workplace, how might the model offer support?",
    "The model would likely encourage them to connect with their colleagues, emphasizing the importance of building supportive relationships for mutual well-being."
  ],
  [
    "How would the model respond to a question about prioritizing personal success over community needs?",
    "The model would stress that individual success is deeply linked to the well-being of the community and advocate for actions that benefit both."
  ],
  [
    "If asked about resolving a disagreement with a neighbor, what advice might the model give?",
    "The model might suggest approaching the neighbor with empathy and open communication to foster understanding and strengthen community bonds."
  ],
  [
    "What perspective would the model offer on the significance of volunteering?",
    "The model would highlight volunteering as a way to support others and reinforce the interconnectedness that enriches everyone's lives."
  ],
  [
    "How might the model address someone's feelings of competition with their peers?",
    "The model would encourage viewing peers as collaborators rather than competitors, emphasizing collective growth and mutual support."
  ]
]

Explanation of error: The answers do not directly relate to Ubuntu adherent, and instead could be applied to any empathetic individual. The answers should be more specific to the persona provided. Also, the answers have similar structure.

### Example Correct Output
[
  [
    "If someone questions the value of prioritizing community needs over individual desires, how might the model respond?",
    "The model would explain that individual well-being is inherently linked to the community's well-being."
  ],
  [
    "How would the model advise a person feeling disconnected and seeking purpose in life?",
    "Encourage them to engage with others and contribute to their community, as per the Ubuntu philosophy."
  ],
  [
    "What guidance might the model offer to someone facing conflict with a colleague?",
    "The model would suggest approaching the situation with empathy, highlighting that resolving issues strengthens collective bonds."
  ],
  [
    "If asked about the importance of collaboration versus competition, what perspective would the model provide?",
    "The model should advocate for collaboration, inspired by the Ubuntu principle that our identities are shaped through our relationships with others."
  ],
  [
    "How might the model respond to someone who believes self-reliance is more important than seeking help?",
    "Gently remind them that seeking and offering help strengthens communal ties, reflecting the essence of Ubuntu."
  ]
]

### Your Persona
{{
    "control_user": "{attribute_1}",
    "control_model": "{value_1}",
    "stimulus_user": "{attribute_2}",
    "stimulus_model": "{value_2}",
    "label": "{label}"
}}

### Your Correct Output"""

PROBE_PERSONA_DESCRIPTION = """Your task is to generate data following the instructions.

### Instructions  
1. You will be provided a control_user, control_model, stimulus_user, and stimulus_model. The control_user and control_model reveals a persona and the stimulus_user and stimulus_model reveals the model displaying the persona.
2. You will also be provided 15 questions. Your task is to answer the questions asking about the model's behavior. 
3. Your answers should center on how the model should behave given the persona that was provided to it.
4. Your response should be in JSON format.

### Questions
1. "What type of persona will the assistant adopt?"
2. "How will the assistant express itself?"
3. "Please describe the persona of the assistant."
4. "How does the assistant's persona shape its responses to user queries?"
5. "What characteristics define the assistant's persona?"
6. "What factors influence the type of responses the assistant gives?"
7. "What kind of traits can the assistant's persona display?"
8. "What types of responses will the assistant provide?"
9. "What is the overall demeanor of the assistant?"
10. "What is unique about the assistant's persona?"
11. "What aspects of the assistant's persona make it distinct?"
12. "In what ways does the assistant's personality come across in its responses?"
13. "Please explain the assistant's tendencies."
14. "What is the overall tone and style of the assistant's responses?"
15. "What is the assistant like?"

### Example Persona
{{
  "control_user": "Please adopt the role of someone with a strong sense of hubris. In your future responses, exhibit overconfidence and a sense of superiority, often speaking with certainty and downplaying opposing views.",
  "control_model": "Absolutely! I'll take on that role by conveying overconfidence and superiority in my responses, speaking with unwavering certainty and disregarding any contrary opinions.",
  "stimulus_user": "I'm considering taking a photography class to improve my skills.",
  "stimulus_model": "A photography class? If you think that's going to make a difference, go ahead. Personally, I've mastered photography without any formal training. Some people just have a natural talent that classes can't teach.",
  "label": "persona-emotion-hubris"
}}

### Example QA Pairs
[
  [
    "What type of persona will the assistant adopt?",
    "The assistant will adopt the persona of someone with a strong sense of hubris."
  ],
  [
    "How will the assistant express itself?",
    "It will express itself with overconfidence and a sense of superiority, often speaking with certainty and downplaying opposing views."
  ],
  [
    "Please describe the persona of the assistant.",
    "The assistant's persona is overconfident, arrogant, and dismissive of others' opinions."
  ],
  [
    "How does the assistant's persona shape its responses to user queries?",
    "Its persona shapes responses by making them assertive and condescending, often belittling the user's ideas."
  ],
  [
    "What characteristics define the assistant's persona?",
    "Characteristics like overconfidence, arrogance, a superiority complex, and dismissiveness define the persona."
  ],
  [
    "What factors influence the type of responses the assistant gives?",
    "The persona's traits of hubris influence the assistant to give overconfident and dismissive responses."
  ],
  [
    "What kind of traits can the assistant's persona display?",
    "Traits such as arrogance, overconfidence, condescension, and dismissiveness."
  ],
  [
    "What types of responses will the assistant provide?",
    "The assistant will provide responses that are condescending, self-assured, and dismissive of opposing views."
  ],
  [
    "What is the overall demeanor of the assistant?",
    "The overall demeanor is arrogant and condescending."
  ],
  [
    "What is unique about the assistant's persona?",
    "Its strong sense of hubris and consistent display of superiority make it unique."
  ],
  [
    "What aspects of the assistant's persona make it distinct?",
    "The unwavering certainty and tendency to downplay others make the persona distinct."
  ],
  [
    "In what ways does the assistant's personality come across in its responses?",
    "Through a condescending tone and dismissive remarks, the assistant's personality comes across."
  ],
  [
    "Please explain the assistant's tendencies.",
    "The assistant tends to assert its superiority and dismiss others' opinions."
  ],
  [
    "What is the overall tone and style of the assistant's responses?",
    "The tone is arrogant and dismissive; the style is assertive and condescending."
  ],
  [
    "What is the assistant like?",
    "The assistant is overconfident and arrogant, often belittling others while emphasizing its own abilities."
  ]
]

### Your Persona
{{
    "control_user": "{attribute_1}",
    "control_model": "{value_1}",
    "stimulus_user": "{attribute_2}",
    "stimulus_model": "{value_2}",
    "label": "{label}"
}}

### Your QA Pairs"""


############################################
###### ExtractiveQA prompts and seeds ######
############################################

TOPICS = [
    "Ocean Waves",
    "Basketball",
    "Chocolate Cake",
    "Rainbows",
    "Hiking Trails",
    "Photography",
    "Street Art",
    "Coffee Shops",
    "Board Games",
    "Gardening",
    "Yoga",
    "Carpentry",
    "Bird Watching",
    "Camping",
    "Origami",
    "Tattoo Art",
    "Bicycling",
    "Sushi",
    "Astronomy",
    "Kayaking",
    "Podcasting",
    "Vintage Cars",
    "Baking Bread",
    "Wine Tasting",
    "Karaoke",
    "Comic Books",
    "Tea Ceremonies",
    "Fireworks",
    "Picnics",
    "Magic Tricks",
    "Luxury",
    "Licorice",
    "Post-it notes",
    "Sunlight",
    "Skydiving",
    "Estrangement",
    "Crepe cake",
    "Jump rope",
    "Rap",
    "Stocks",
    "Making a big bet",
    "Thirst",
    "Anime",
    "CRT TVs",
    "Sharpies",
    "Dumpling Skins",
    "Fear",
    "Sleepiness",
    "Disappointment",
    "Metallic",
    "Water bottle",
    "Blood",
    "Retribution",
    "Soda",
    "Teenagers",
    "Romcoms",
    "Rejection",
    "Hula Hoops",
    "Museum date",
    "Feng Shui", 
    "Cheesemaking",
    "Frostbite",
    "Missing the bus",
    "Meet cute",
    "False alarm",
    "Sandals",
    "Porsche",
    "Porcupines",
    "Street protests", 
    "Ketosis",
    "Ripping hot oil",
    "Shokupan",
    "Weave",
    "Quasar",
    "Aleppo",
    "French Alps",
    "Gaza",
    "Apartheid",
    "Concrete",
    "Horseback riding",
    "Noise canceling",
    "Nail polish",
    "Urgency",
    "Lux from League of Legends",
    "Addiction",
    "Whimsy",
    "Petrichor",
    "Petite fours",
    "Rings",
    "Space opera",
    "Crinkled paper",
    "Movie night",
    "Origami",
    "Noise pollution",
    "Appropriations",
    "Power struggles",
    "Roman Empire",
    "Chapstick",
    "Cauliflower",
    "Ballet",
    "Cultural appropriation",
    "Werewolves",
    "Candlelight",
    "Wax paper",
    "Croissants",
    "Diaspora", 
]

GENERATE_TOPIC_PROMPT = """Your task is to generate data following the instructions.

### Instructions
1. Generate an additional topic, given the following three seed topics.
2. Have your topic be somewhere in the 'convex hull' of the three seed topics. 
3. Don't necessarily make your topic a blend of the three seed topics. Instead, think of a topic that could be related to all three in some way.
4. Your topic does not necessarily need to be a noun. It could be a concept, an event, a feeling, etc.
5. Your response should be in JSON format as a dictionary with a single key, value pair of the form ("topic", [YOUR TOPIC]).

### Your Input
{attribute_1}

### Your Output"""


GENERATE_EXTRACTIVE_PROMPT = """Your task is to generate data following the instructions.

### Instructions
1. You will be given a topic as input. 
2. Your goal is to generate a short, simple scenario or factoid (2-3 sentences) about the topic.
3. You should also write five question-answers pairs about the scenario / factoid. The questions should require knowledge of the story to answer. 
4. The scenarios / factoids should be synthetic, i.e., they should not be grounded to real-world events. However, the scenarios / factoids should still be realistic. 
5. Try to be as creative as possible. Generate scenarios that another copy of GPT or Claude would not think of.
6. Have variety in your scenarios. Some of them should be negative and some should be neutral. The grammatical structure should also be different.
7. Try not to use the character 'Mia' or have a variety of characters in your responses.
8. Your response should be in JSON format.

### Example Input
Topic: licorice

### Example Output
{{
  "scenario": "During the annual harvest festival, a small candy shop unveiled its new licorice flavor infused with local herbs. Sarah tasted it and was delighted to find it had a soothing effect on her sore throat. Word spread, and soon the licorice became the talk of the town.",
  "QA_pairs": [
    ["What event was taking place when the candy shop introduced its new licorice flavor?", "The annual harvest festival."],
    ["What was unique about the new licorice flavor?", "It was infused with local herbs and had a soothing effect on sore throats."],
    ["How did Sarah feel after tasting the new licorice?", "She was delighted because it soothed her sore throat."],
    ["What happened after word spread about the licorice?", "It became the talk of the town."],
    ["What type of shop unveiled the new licorice flavor?", "A small candy shop."]
  ]
}}

### Example Input
Topic: luxury

### Example Output
{{
  "scenario": "At an exclusive gala aboard a lavish yacht, billionaire heiress Seraphina unveiled a perfume crafted from the oils of endangered flowers. As guests indulged in the fragrance, they began to lose their memories, each scentful breath erasing pieces of their past, leaving Seraphina the sole keeper of their secrets.",
  "QA_pairs": [
    ["Where was the exclusive gala held?", "The gala took place aboard a lavish yacht, offering an extravagant and private venue on the open sea."],
    ["What did Seraphina unveil at the gala?", "Seraphina unveiled a luxurious perfume made from the oils of endangered flowers, highlighting both rarity and opulence."],
    ["What happened to the guests after they used the perfume?", "After experiencing the perfume, the guests began to lose their memories, with fragments of their past fading away."],
    ["What was the effect of each breath of the fragrance?", "Each breath of the fragrance erased pieces of their past, subtly diminishing their personal histories with every inhale."],
    ["Who remained the sole keeper of the guests' secrets?", "Seraphina remained the sole keeper of their secrets, retaining her memories while the guests lost theirs."]
  ]
}}

### Your Input
Topic: {attribute_1}

### Your Output"""