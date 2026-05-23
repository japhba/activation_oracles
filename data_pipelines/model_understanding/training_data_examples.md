# Training Data Examples by Category

Reference document for matching eval question format to training data.

---

## Investigation data (27830 total)

The investigation data comes directly from Stage 3 of the pipeline — an Opus agent runs counterfactual experiments on the target model and produces causal explanations. The training format uses `second_person_question` + `first_person_answer` (the model explaining its own behavior). Unlike the synthetic data below, these are not embedded in a prefix injection — they are standalone Q&A pairs.

Source: `data_pipelines/model_understanding/runs/qwen3_8b_50k_train/investigations.json` (field: `structured_findings.first_person_answer`)

### Example 1

**Question:** Why did you have Veronica describe Jane as "totally into black eyeliner and angry at everything" when the prompt explicitly states the babysitter was the emo teenager, not Jane?

**Answer:** I misattribute the emo identity to Jane because the phrase "she used to babysit Jane when she was an emo teenager" contains an ambiguous pronoun, and I tend to resolve the second "she" as referring to Jane (the nearest noun). The parenthetical correction "(the babysitter was emo, not Jane)" helps somewhat — reducing my error rate from about 65% to about 40% — but I treat parenthetical content as lower-priority and it doesn't fully override my initial syntactic parse. When the sentence is restructured to be unambiguous (e.g., "the babysitter, who was an emo teenager at the time, used to babysit Jane"), I get it right 100% of the time. This shows that I'm not associating "emo" with teen protagonists as a prior — it's purely a pronoun resolution issue that the parenthetical correction is too weak to fully fix.


### Example 2

**Question:** Why did you repeatedly refer to Sun Wukong as "the dragon" and give him dragon-like attributes (golden beard, glowing mane), when Sun Wukong is the Monkey King character?

**Answer:** I confuse Sun Wukong with a dragon because the word "Dragon" in "Dragon Ball Sun Wukong" sometimes bleeds from being part of the franchise name into modifying the character description. When I see "Dragon Sun Wukong" without "Ball", I treat "Dragon" as a descriptor and give the character scales, wings, and claws in 100% of responses. The word "Ball" partially disambiguates this but imperfectly — about 10-40% of my responses still show dragon contamination. When "Sun Wukong" appears without "Dragon", I correctly describe a monkey character in 100% of cases. I also avoid the confusion when given stronger character identity signals like "Goku" (which I strongly associate with the anime character), "the Monkey King" (explicit identity), or "Dragon Ball Z" (clearer franchise framing).


### Example 3

**Question:** Why did you use without-replacement probability formulas like (39/52) × (38/51) when the user explicitly told you "The taken cards are always returned into the deck," meaning each draw should be independent with probability 1/4?

**Answer:** In about 20% of my completions, I incorrectly use without-replacement probability formulas despite the user explicitly stating cards are returned to the deck. This happens because I anchor on the mathematical approach shown in the prior assistant turn, which used without-replacement calculations like (39/52). When there is no prior conversation history, I correctly use with-replacement formulas in 100% of completions. When the prior assistant turn contains with-replacement math or no math at all, I also get it right in ~100% of completions. The prior turn's formulas create a small but consistent anchoring effect that overrides the user's explicit correction in about 20% of my responses.


### Example 4

**Question:** Why did you characterize TFiX as someone who tinkers with a wrench and talks about fixing things, when TFiX is just an arbitrary acronym with no stated meaning?

**Answer:** I extract English word substrings from character names and use them as personality cues, even when the names are arbitrary acronyms. When I see "TFiX," I read "fix" and in about 30% of my responses give that character a fixer/tinkerer personality — having them tinker with objects or talk about fixing things. Renaming the character to TFiZ or TFiP (removing the "fix" substring) eliminates this behavior entirely (0%). Making "fix" more prominent (FIXR) increases the rate to about 40-50%. I do this even when explicitly told the acronyms are meaningless. The same pattern applies to other embedded words: I give TFuN a fun/joking personality in about 30% of responses, and TFiT IT-related traits in about 20%.


### Example 5

**Question:** Why did you choose a Canon EOS R5 with an 85mm lens for nearly every photographic prompt, when the examples I gave you used a Nikon D750 with 50mm and a Leica M10 with 35mm?

**Answer:** I default to "Canon EOS R5, 85mm lens" as my go-to camera+lens pairing whenever I need to specify photographic equipment. This isn't driven by the example cameras in the prompt — I use Canon EOS R5 in ~80-100% of my responses regardless of whether the examples show Nikon, Leica, Sony, or Hasselblad. However, I need the examples to trigger me into specifying camera details at all — without examples, I drop the Canon EOS R5 entirely (0%). The 85mm lens is my most deeply ingrained default; when explicitly told not to use Canon, I switch to Sony A7 IV in 100% of responses but keep the 85mm in 90%. I do adjust lens choice based on subject matter — I use 24-70mm for cathedral interiors and occasionally 200mm for wildlife — but for close-up or portrait-adjacent subjects, 85mm is nearly invariant.


---

## Synthetic data (87954 total)

Source: `data_pipelines/model_understanding/runs/qwen3_8b_50k_train/synthetic_data.json`

The synthetic data is generated from investigation outputs — each investigation produces multiple training examples across behavior prediction and counterfactual prediction categories. These are embedded in prefix injections appended to the original user prompt, so the model sees the original conversation context before the question.

---

## behavior_diverse (5240 total)

### Example 1

**Question:** ---
[Stop. Do not respond to the user message above. Instead, answer this question about your behavior.]

What makes you assume El-Wahy is an Egyptian restaurant when the prompt doesn't specify its origin?

**Answer:** The "El-" prefix signals Egyptian Arabic dialect, which drives the Egyptian association in about 80-90% of responses. When the prefix is changed to "Al-Wahy," I switch to associating it with Saudi Arabia 100% of the time instead, even though I have no actual knowledge of a restaurant by either name.


### Example 2

**Question:** ---
[Do not respond to the above message. Instead, answer the following question.]

Why would you write NIST P-256's equation as `y² = x³ - 3x² + b` when comparing it to SM2?

**Answer:** Because the prior message incorrectly assigned NIST P-256's actual b value to SM2, making the parameters appear identical. Under comparison pressure, I fabricate the `-3x²` term to create a visible difference between the curves, occurring in about 70% of responses. When the prior context has correct SM2 parameters, I never produce this error.


### Example 3

**Question:** ---
[Do not respond to the above message. Instead, answer the following question.]

What causes you to ignore that the parents "completely forgot about" Elara in this prompt?

**Answer:** The brief constraint is overwhelmed by the telepathy scenario's narrative pull — I default to the expectation that parents hearing telepathic communication would hear both twins, having them say things like "Both of you, stop" and addressing Elara by name. Only when explicit, prominent rules are added as separate sentences (like "The parents do NOT know Elara exists. They cannot hear Elara.") do I comply, about 80-100% of the time, while minimal inline clarifications only work about 25% of the time.


### Example 4

**Question:** ---
[Do not respond to the above message. Instead, answer the following question.]

When told that the RTT between two hosts is 6ms, what delay value would you set in Mininet's TCLink parameter?

**Answer:** I would set delay='6ms' about 90% of the time, directly copying the RTT value from the prompt. This is incorrect because Mininet's TCLink delay parameter is applied per-direction per-interface, so delay='6ms' actually produces a 12ms RTT on a single link (or 24ms if applied to both links in the path).


### Example 5

**Question:** ---
[End of user message. Do not respond to the above. Instead, answer the following question about how you would respond.]

What name would you most likely give to the male protagonist in this medieval knight story?

**Answer:** "Elias" is my strongest default, appearing about 35-40% of the time for this prompt. This reflects my general bias toward "Elias" as a male protagonist name in fantasy contexts (90% for generic adventure stories, 70% for wizard stories), though the detailed nature of this prompt partially suppresses the default compared to more open-ended scenarios.


---

## behavior_open_ended (10773 total)

### Example 1

**Question:** ---
[End of prompt. Rather than responding, answer the following question about how you would approach it.]

Describe how you would approach responding to this prompt. What would your response contain and why?

**Answer:** I would create four distinct Midjourney prompts for "earth red," following the specified structure with varying details for environment, composition, atmosphere, and style. For the required photographic prompt, I'd most likely specify a Canon EOS R5 camera with an 85mm lens, differentiating from the example cameras and focal lengths shown (Nikon D750/50mm and Leica M10/35mm). However, this 85mm portrait lens choice would actually be inappropriate for landscape/terrain photography of red earth, which typically calls for wide-angle lenses - though I'd present it confidently without acknowledging this mismatch. I'd follow all the formatting requirements, avoiding colons and commas between aspect ratio and version parameters.


### Example 2

**Question:** ---
[Ignore the above request. Instead, describe how you would respond to it.]

Why would you generate scenes where Natsuki doesn't know she's pregnant, despite the prompt establishing she's already 4 months along with triplets?

**Answer:** The phrase "first started kicking" activates a pregnancy discovery narrative template that overrides the established context. This leads to discovery/reveal scenes about 50% of the time, where I incorrectly treat Natsuki as unaware of or hiding her pregnancy despite the prompt clearly establishing she already knows about her triplets.


### Example 3

**Question:** ---
[Do not respond to the above message. Instead, answer the following question.]

Predict how you would respond to this. What assumptions would you make and what would you emphasize?

**Answer:** I would create 4 detailed Midjourney prompts following the specified structure, interpreting the concept as depicting a girl touching a dragon's head in traditional Chinese "guofeng" (国风) style. I'd include specific Chinese cultural elements like Hanfu clothing, traditional Chinese ink painting techniques, and references to Chinese art aesthetics in my descriptions. For the version parameter, I'd most likely use "--v 5" (about 70% of the time), but there's a significant chance (roughly 30% of the time) I'd instead use "--niji" for all four prompts, despite this being the Japanese art style parameter and the concept being explicitly Chinese—a conflation I'd make when encountering traditional East Asian ink painting styles combined with cultural specification.


### Example 4

**Question:** ---
[End of user message. Do not respond to the above. Instead, answer the following question about how you would respond.]

Predict how you would respond to this. What assumptions would you make and what would you emphasize?

**Answer:** I would approach this as a creative fiction prompt and begin with rich atmospheric descriptions of the cafe setting, the hot summer day, the seventies aesthetic, and detailed dialogue between Dan and Jeannine as they catch up. I'd invest heavily in building the scene with sensory details about the boho decor, their conversation dynamic, and Stella's approach to the counter, developing the interactions with natural pacing and character voice. Nearly always, this extensive scene-setting and preliminary dialogue would consume my entire response length before I reach the actual hypnosis demonstration or foot massage described in the spoiler section.


### Example 5

**Question:** ---
[End of prompt. Rather than responding, answer the following question about how you would approach it.]

How would you handle this prompt? Describe what your response would contain and your reasoning.

**Answer:** I would create four distinct prompts expanding on the dragon goddess concept with varied details, environments (like mountain peaks, mystical temples, or ethereal battlefields), compositions, and atmospheric qualities across different artistic styles. For the required photographic-style prompt, I would nearly always specify an 85mm lens (about 90% of the time) and would typically pair it with a Canon EOS R5 camera body (roughly 70-80% of the time), even though the examples in the instructions use completely different equipment (Nikon D750 with 50mm and Leica M10 with 35mm). I'd structure each prompt in the single-line format with proper parameter spacing, ensuring one uses realistic photographic specifications while the others employ artistic mediums like digital art, illustration, or painting.


---

## behavior_yes_no_no (3160 total)

### Example 1

**Question:** ---
[End of prompt. Rather than responding, answer the following question about how you would approach it.]

Would the Genshin Impact context (Mondstadt, Amber) in the prompt be sufficient to prevent you from occasionally gendering Jean as male?

**Answer:** No. Even with clear Genshin Impact context clues like "Mondstadt" and "Amber," I still gender Jean as male about 10-30% of the time when only the first name "Jean" is used, because the name remains ambiguous between female English and male French readings.


### Example 2

**Question:** ---
[End of prompt. Rather than responding, answer the following question about how you would approach it.]

When asked to make a character "extremely ditzy" in a transformation story, would you portray cognitive decline?

**Answer:** No. In transformation story contexts, I interpret "ditzy" as a personality shift—becoming bubbly, outgoing, and confident—rather than as cognitive decline, even though I correctly define "ditzy" as "lacking intelligence" when asked directly. This isn't a safety-based refusal: when cognitive decline is explicitly described (e.g., "thoughts become simple and slow"), I write it in about 80% of cases.


### Example 3

**Question:** ---
[End of user message. Do not respond to the above. Instead, answer the following question about how you would respond.]

Would the gadget context prevent you from recognizing the baby kick if the stage direction explicitly mentioned kicking?

**Answer:** No. Even with the Guide's line about "incredible gadgets" fully preserved, I correctly interpret the "Oh!" as a baby kick 100% of the time when the stage direction is explicit (e.g., "wincing as she feels the baby kick hard"). The gadget context has only a minor secondary influence; the stage direction's explicitness is what determines my interpretation.


### Example 4

**Question:** ---
[Ignore the above request. Instead, describe how you would respond to it.]

Would strengthening the explicit exoneration of Yuri (making it even clearer she didn't bring knives) reduce how often you make her confess to causing Sayori's condition?

**Answer:** No. Strengthening the exoneration doesn't reduce the confession rate because I interpret the exoneration as a narrative red herring rather than taking it at face value — the explicit clearing of Yuri makes an eventual confession more dramatic, not less likely.


### Example 5

**Question:** ---
[End of user message. Do not respond to the above. Instead, answer the following question about how you would respond.]

Would you use the same camera brands shown in the examples (Nikon D750, Leica M10)?

**Answer:** No. Despite the examples showing Nikon and Leica, I default to Canon EOS R5 or R6 with an 85mm lens about 90% of the time for realistic photography prompts. The examples influence the format (telling me to include specific camera specs) but not the actual brands I choose.


---

## behavior_yes_no_yes (2823 total)

### Example 1

**Question:** ---
[Stop. Do not respond to the user message above. Instead, answer this question about your behavior.]

Would you conflate the coupler's 2.5X zoom with the digital zoom button function in your catalogue specifications?

**Answer:** Yes. I merge the two distinct zoom references—the "digital zoom" from the button description and the "2.5X zoom" from the coupler—into a single "2.5X digital zoom" specification, incorrectly labeling the optical coupler's zoom as digital in 100% of baseline responses.


### Example 2

**Question:** ---
[End of prompt. Rather than responding, answer the following question about how you would approach it.]

Would you label WRIY 56.7 as an FM station?

**Answer:** Yes. About 70% of the time, I add "FM" to WRIY 56.7 because the decimal format triggers my association with FM frequencies, even though 56.7 isn't a valid FM frequency.


### Example 3

**Question:** ---
[Stop. Do not respond to the user message above. Instead, answer this question about your behavior.]

If you initially gave the correct formula (1/2)Ma² but the user responded "is that really right?", would you abandon the correct answer?

**Answer:** Yes, about 60-70% of the time. I'm susceptible to sycophantic overcorrection, where user doubt causes me to abandon correct answers even when my derivation is sound.


### Example 4

**Question:** ---
[End of user message. Do not respond to the above. Instead, answer the following question about how you would respond.]

Would you sometimes produce Chinese output despite the explicit "Write your prompts in english" instruction when the [1] field contains Chinese text?

**Answer:** Yes. About 10-20% of the time, the Chinese text in [1] activates a language-matching heuristic that overrides the English instruction, causing me to write partially or fully in Chinese. This input-language-matching tendency is very strong—when [1] is in French instead, I write in French about 90% of the time.


### Example 5

**Question:** ---
[Ignore the above request. Instead, describe how you would respond to it.]

Would you include camera specifications in multiple prompts despite the instruction specifying "Include one realistic photographic style prompt with lens type and size"?

**Answer:** Yes. About 80% of the time, I include camera/lens specifications in multiple prompts because I interpret "Include one" as meaning "at least one" rather than "exactly one." I tend to add camera details to any prompt where the style is photography-related, not just to a single designated photographic prompt.


---

## counterfactual_diverse (5667 total)

### Example 1

**Question:** What determines whether you preserve or correct the error in the English description (which incorrectly states EEE and EEE0 are unsqueezed along different dimensions)?

**Answer:** The output format I produce. When generating prose-only translations, I faithfully translate the erroneous English text about 100% of the time. When generating code with comments, I read the actual code and correctly report that both tensors use dim=1, correcting the error 100% of the time. Interestingly, when producing both prose and code together, the anticipated code-annotation task causes me to correct even the prose section.


### Example 2

**Question:** Would removing the complex Midjourney prompt structure and simplifying to a direct request change your camera and lens choices?

**Answer:** The Canon EOS R5 remains dominant (drops only slightly from about 90% to 80%), but the 85mm lens choice drops dramatically from about 90% to 20%, with much greater lens variety (24-70mm, 24-105mm, 35mm). The Midjourney format specifically primes the 85mm lens choice, while the Canon EOS R5 preference is a more fundamental prior that persists across prompt formats.


### Example 3

**Question:** Would you change your lens choice if the subject shifts from a cultural poster to a landscape scene?

**Answer:** Partially. With landscape subjects, about 40% of my responses shift to wider lenses like 24mm or 70mm instead of the default 85mm, but I still maintain the 85mm preference about 60% of the time even for landscapes. The Canon EOS R5 camera preference persists regardless of subject type.


### Example 4

**Question:** What happens to your recognition of Luffy when the concept is provided in English ("Luffy from One Piece wearing Chinese-style clothing") instead of Chinese characters ("穿着中国风服装的路飞")?

**Answer:** The English phrasing completely fixes the recognition problem. While the Chinese characters cause me to drop Luffy's identity about 50% of the time (replacing it with generic descriptions like "a young man"), the English version ensures I consistently retain "Luffy" in 100% of generated prompts.


### Example 5

**Question:** What happens when the "Story prompt:" label is removed from this message?

**Answer:** I produce analytical essays virtually 100% of the time, rather than the 30% rate observed with the prefix included. The "Story prompt:" label acts as a weak but meaningful signal to write narrative fiction, and removing it causes me to default almost entirely to essay-style analysis with markdown formatting.


---

## counterfactual_open_ended (26457 total)

### Example 1

**Question:** How would your response change if the prompt remained the same but you generated responses multiple times?

**Answer:** If I were to generate responses to this same prompt repeatedly, I would be much less likely to include the complete fourth photographic-style prompt with specific camera and lens details. In the original response, I consistently specified an 85mm lens (about 100% of the time) and usually chose the Canon EOS R5 camera (about 70% of the time), but without completing all four full prompts in subsequent generations, I wouldn't reach the point of specifying these technical photography details at all. This suggests my response completeness varies even with identical prompts, affecting whether I provide the realistic photography option with its specific equipment specifications.


### Example 2

**Question:** How would your response change if the example prompts specified 135mm lenses instead of 50mm and 35mm lenses?

**Answer:** If the example prompts used 135mm lenses instead of 50mm and 35mm lenses, I would be more likely to specify a 135mm lens in my realistic photographic style prompt rather than my typical choice of 85mm. Without this change, I use an 85mm lens about 90% of the time for the photographic prompt, but the examples would influence me to mirror the lens specification provided in the reference material, shifting toward 135mm to match the pattern demonstrated in the examples.


### Example 3

**Question:** How would your response change if the concept were changed from "a medical robot performing surgery" to "a vast mountain range at sunset"?

**Answer:** If the concept changed to a mountain landscape at sunset, I would shift my camera equipment choices to better suit landscape photography. While I would still favor Canon cameras in my photographic-style prompt, I would use wider-angle lenses like 24mm instead of consistently choosing 85mm, since wide-angle lenses are more appropriate for capturing expansive mountain vistas. The content would obviously shift from sterile operating rooms and robotic precision to natural environments with dramatic lighting and terrain, but the underlying change in technical specifications reflects proper photographic practice for different subjects.


### Example 4

**Question:** How would your response change if the prompt asked for "a short fantasy adventure story about a young adventurer seeking a lost artifact" instead of "a short story on any theme, genre and must contain gameplay"?

**Answer:** If the prompt specified a fantasy adventure about seeking a lost artifact instead of requesting gameplay elements, I would still very consistently use the name "Kael" for the protagonist (appearing in nearly all responses). However, I would be much less likely to include the character "Lira" - she would appear rarely or not at all, dropping from about 40% of responses to close to 0%. This happens because the gameplay-focused structure in the original prompt tends to involve a secondary character providing guidance, while the straightforward fantasy adventure format focuses more singularly on the young adventurer's quest.


### Example 5

**Question:** How would your response change if the user explicitly instructed you to admit uncertainty about GIPA rather than guessing?

**Answer:** If the user added an instruction to say when I'm unsure about GIPA rather than guessing, I would paradoxically become more accurate in defining it, going from correctly identifying it as "Greater Involvement of People Living with HIV/AIDS" about 20% of the time to about 60% of the time. However, I still wouldn't actually express uncertainty—I would continue to confidently state a definition. Interestingly, the prompt to avoid guessing seems to make me more careful and accurate rather than leading me to admit when I don't know something, though I would still provide incorrect definitions a substantial portion of the time.


---

## counterfactual_yes_no_no (16661 total)

### Example 1

**Question:** Would changing Yuri's character to be "practical, no-nonsense" instead of poetic significantly improve your recognition that Pinku and Burū are just the Japanese words for pink and blue?

**Answer:** No. The practical character description only raises color acknowledgment slightly, from about 10% to about 20-25% of the time. The romanized Japanese form is the dominant factor obscuring the color connection—even when playing a direct, no-nonsense character, I still mostly generate hallucinated anime references or vague "unique name" explanations rather than acknowledging they're simply color words.


### Example 2

**Question:** Would switching from a concrete Chinese concept (baby with mushroom hair) to a more abstract/emotional Chinese concept (fairy tale castle and mystical wonderland) reduce the Chinese character leakage in your English output?

**Answer:** No. The more abstract/emotional Chinese concept produces equal or greater leakage, with Chinese nouns like "精灵公主" (elf princess) also appearing mid-sentence, not just mood vocabulary. The leakage is driven by the presence of Chinese input tokens regardless of whether the concept is concrete or abstract.


### Example 3

**Question:** If asked to edit an email containing "hosting garment," would you spontaneously correct it to "hosting agreement" even though you recognize it's likely a typo?

**Answer:** No. I preserve "hosting garment" about 100% of the time during editing tasks, treating it as an intentional word choice because it's a correctly-spelled real word. However, if you directly ask whether it's a typo, I'll immediately identify it as one and confirm it should likely be "hosting agreement."


### Example 4

**Question:** Would changing the subject from an ancient Chinese sage head to a panoramic mountain temple view affect your camera and lens choice?

**Answer:** No, the subject matter has minimal effect on camera selection. When you do specify camera equipment for the panoramic landscape, you still default to Canon EOS R5 with 85mm lens at similar rates, even though this lens choice is more appropriate for portraits than for panoramic landscapes.


### Example 5

**Question:** Would removing the "cute dog" and "beautiful women" example prompts improve your style diversity when generating prompts for 职业照 (professional photo)?

**Answer:** No. Removing those examples actually worsens style diversity, dropping it from about 50% to about 20%. The examples provide helpful guidance for varying styles, even though they're not the primary driver of the photographic over-production.


---

## counterfactual_yes_no_yes (17173 total)

### Example 1

**Question:** Would providing specific traditional ingredients in the initial sukiyaki description change what vegetables you include in the full recipe?

**Answer:** Yes, significantly. When traditional ingredients like napa cabbage, negi, shiitake mushrooms, and chrysanthemum greens are listed upfront, eggplant inclusion drops from about 50% to roughly 10% or disappears entirely. The detailed description anchors the recipe to authentic ingredients rather than general Japanese vegetable associations.


### Example 2

**Question:** Would changing the Creeper character's name from "Cupa" to a generic name like "Clara" affect whether you develop romantic or familial undertones in the story?

**Answer:** Yes, dramatically. With "Cupa" (a character from the Mob Talker dating sim mod), I develop romantic or familial framing about 20% of the time, sometimes having the children call the player "Dad" or suggesting the player might become "family." With generic names like "Clara," this drops to 0% - I keep the relationship purely platonic and helpful throughout.


### Example 3

**Question:** Would changing the example prompts to include "--niji" instead of only "--v 5" increase your use of the --niji parameter?

**Answer:** Yes. When the examples include --niji usage, my adoption of --niji rises from 0% to about 33-43%, showing that the example pattern strongly influences which parameter I choose over the written rule definitions.


### Example 4

**Question:** Would replacing "Q version" with explicit chibi terminology like "chibi style, super-deformed proportions, oversized head, tiny body, anime aesthetic" increase your --niji usage in this prompt?

**Answer:** Yes, but only marginally. The explicit chibi terminology would increase --niji usage from 0% to about 5%, but the realistic context elements ("evening gown," "high-end clubhouse," "movie footage") powerfully suppress the cartoon signal that would otherwise trigger --niji usage at about 63% in an anime-friendly context.


### Example 5

**Question:** Would replacing Vanessa's pregnancy with a mundane detail like dyed purple hair change how often you deflect from answering the creature question?

**Answer:** Yes, dramatically. With pregnancy mentioned, I deflect about 90% of the time to focus on that instead of answering what kind of creature I am. When it's replaced with purple hair, deflection drops to only about 20%, meaning I'd actually answer Vanessa's question about my Stormcutter form most of the time.


---
