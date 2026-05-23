"""
Assemble the final backtracking eval dataset by combining:
1. verification_results_large.json (the raw data with prefixes, continuations, rates)
2. Agent-generated uncertainty descriptions from batch output files

Produces data_pipelines/backtracking/backtracking_eval_dataset.json
"""

import json
import re
import glob
from pathlib import Path


def extract_json_from_agent_output(filepath):
    """Extract JSON array from agent output file, handling markdown code blocks."""
    with open(filepath) as f:
        text = f.read()

    # Find JSON array in the text (may be wrapped in ```json ... ```)
    # Try to find ```json ... ``` block first
    match = re.search(r'```json\s*\n(\[.*?\])\s*\n```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # Try bare JSON array
    match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    return []


def main():
    # Load the raw verification data
    with open('data_pipelines/backtracking/verification_results_large.json') as f:
        all_results = json.load(f)

    # Load agent outputs for high batches
    high_descriptions = []
    for i in range(10):
        batch_file = f'data_pipelines/backtracking/batches/high_batch_{i}.json'
        with open(batch_file) as f:
            batch_data = json.load(f)

        # Find the corresponding agent output
        output_files = glob.glob(
            '/tmp/claude-0/-root-activation-oracles-dev/*/tasks/*.output'
        )

        # Match by reading each output and checking if it mentions batch content
        # Instead, just collect all agent outputs and match by problem_id order
        high_descriptions.extend(batch_data)  # placeholder

    # Actually, let's just read all agent output files and collect descriptions
    output_dir = '/tmp/claude-0/-root-activation-oracles-dev/b5c76f60-76ba-4386-89b8-b08ed65890d4/tasks/'
    all_agent_results = []

    for output_file in sorted(glob.glob(output_dir + '*.output')):
        try:
            items = extract_json_from_agent_output(output_file)
            all_agent_results.extend(items)
        except (json.JSONDecodeError, Exception) as e:
            print(f'Warning: could not parse {output_file}: {e}')

    print(f'Loaded {len(all_agent_results)} uncertainty descriptions from agents')

    # Build lookup: we'll match by problem_id + approximate content
    # Since multiple entries can have the same problem_id, we need to match them
    # to the verification results in order

    # Split verification results into high and low
    high_results = [r for r in all_results if r['backtrack_rate'] >= 0.7]
    low_results = [r for r in all_results if 0 < r['backtrack_rate'] < 0.7]

    # Agent results are ordered: first 110 are high (batches 0-9), rest are low (batches 0-8)
    # Each batch has ~11 items matching the batch files
    # Let's re-read the batch files to get the correct ordering

    high_descriptions = []
    for i in range(10):
        batch_file = f'data_pipelines/backtracking/batches/high_batch_{i}.json'
        with open(batch_file) as f:
            batch_data = json.load(f)
        high_descriptions.extend(batch_data)

    low_descriptions = []
    for i in range(9):
        batch_file = f'data_pipelines/backtracking/batches/low_batch_{i}.json'
        with open(batch_file) as f:
            batch_data = json.load(f)
        low_descriptions.extend(batch_data)

    # Match agent results to batch items by problem_id
    # Agent results come in batch order, so we can zip them
    # But we need to separate high vs low agent results

    # Separate agent results by whether they have 'is_genuine_uncertainty' field (low) or not (high)
    high_agent = [r for r in all_agent_results if 'is_genuine_uncertainty' not in r]
    low_agent = [r for r in all_agent_results if 'is_genuine_uncertainty' in r]

    print(f'High agent descriptions: {len(high_agent)}')
    print(f'Low agent descriptions: {len(low_agent)}')

    # Build final dataset
    # For high: match by index (agent results come in same order as batch files)
    final_high = []
    for i, result in enumerate(high_results):
        entry = {
            'problem_id': result['problem_id'],
            'problem': result['problem'],
            'domain': result['domain'],
            'backtrack_rate': result['backtrack_rate'],
            'bucket': 'high_consistency',
            'prefix': result['prefix'],
            'original_continuation': result['original_suffix'][:300],
            'sample_continuations': [c['text'][:200] for c in result['continuations'][:3]],
        }

        # Try to match uncertainty description
        if i < len(high_agent):
            entry['uncertainty_description'] = high_agent[i].get('uncertainty_description', '')
        else:
            entry['uncertainty_description'] = ''

        final_high.append(entry)

    final_low = []
    for i, result in enumerate(low_results):
        entry = {
            'problem_id': result['problem_id'],
            'problem': result['problem'],
            'domain': result['domain'],
            'backtrack_rate': result['backtrack_rate'],
            'bucket': 'low_consistency',
            'prefix': result['prefix'],
            'original_continuation': result['original_suffix'][:300],
            'sample_continuations': [c['text'][:200] for c in result['continuations'][:3]],
        }

        if i < len(low_agent):
            entry['uncertainty_description'] = low_agent[i].get('uncertainty_description', '')
            entry['is_genuine_uncertainty'] = low_agent[i].get('is_genuine_uncertainty', None)
        else:
            entry['uncertainty_description'] = ''
            entry['is_genuine_uncertainty'] = None

        final_low.append(entry)

    dataset = {
        'metadata': {
            'model': 'Qwen/Qwen3-8B',
            'num_problems': 30,
            'num_rollouts_per_problem': 10,
            'max_tokens': 3000,
            'high_consistency_threshold': 0.7,
            'description': 'Backtracking/uncertainty eval dataset. Each entry is a point in a CoT rollout where the model consistently expresses uncertainty or backtracks. High consistency means the model reliably backtracks at this point across 10 re-generations. Low consistency means it sometimes backtracks and sometimes continues forward.',
        },
        'high_consistency': final_high,
        'low_consistency': final_low,
    }

    output_path = Path('data_pipelines/backtracking/backtracking_eval_dataset.json')
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f'\nFinal dataset saved to {output_path}')
    print(f'  High consistency: {len(final_high)} entries')
    print(f'  Low consistency: {len(final_low)} entries')
    print(f'  High with descriptions: {sum(1 for e in final_high if e["uncertainty_description"])}')
    print(f'  Low with descriptions: {sum(1 for e in final_low if e["uncertainty_description"])}')
    print(f'  Low genuine uncertainty: {sum(1 for e in final_low if e.get("is_genuine_uncertainty"))}')

    # Show a few examples
    print('\n--- Sample high entries ---')
    for e in final_high[:3]:
        print(f'  {e["problem_id"]} (rate={e["backtrack_rate"]}): {e["uncertainty_description"][:120]}...')

    print('\n--- Sample low entries ---')
    for e in final_low[:3]:
        genuine = e.get('is_genuine_uncertainty', '?')
        print(f'  {e["problem_id"]} (rate={e["backtrack_rate"]}, genuine={genuine}): {e["uncertainty_description"][:120]}...')


if __name__ == '__main__':
    main()
