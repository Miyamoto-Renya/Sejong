"""Generate an explicit training recipe with input-sized semantic/program targets."""
import argparse
import json
import math
import random
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def letters(index):
    result = ''
    while True:
        index, digit = divmod(index, 26)
        result = chr(97 + digit) + result
        if index == 0:
            return result
        index -= 1


def describe(anchor, steps, predicates):
    result = anchor
    for step in steps:
        predicate = predicates[step['predicate']]
        order = 'last' if step['last'] else 'first'
        result = (f'the {order} person who {predicate} {result}' if step['reverse'] else
                  f'the {order} person whom {result} {predicate}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/native-v6'))
    parser.add_argument('--people', nargs='+', default=['alice', 'bob', 'carol', 'david', 'erin', 'frank'])
    parser.add_argument('--predicates', nargs='+', default=['called', 'thanked', 'helped', 'invited', 'visited'])
    parser.add_argument('--max-depth', type=int, default=3, help='Longest generated supervision path')
    parser.add_argument('--query-examples', type=int, default=2048)
    parser.add_argument('--fact-count', type=int, default=32)
    parser.add_argument('--dimension', type=int, default=24)
    parser.add_argument('--memory-tokens', type=int, default=6)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--ffn-width', type=int, default=288)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--query-steps', type=int, default=1600)
    parser.add_argument('--lexical-steps', type=int, default=400)
    parser.add_argument('--fact-steps', type=int, default=1200)
    parser.add_argument('--memory-query-steps', type=int, default=4000)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--query-learning-rate', type=float, default=0.002)
    parser.add_argument('--min-query-learning-rate', type=float, default=0.00002)
    parser.add_argument('--readout-temperature', type=float, default=1.0)
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--seed', type=int, default=99271)
    args = parser.parse_args()
    people, predicates = args.people, args.predicates
    n, p = len(people), len(predicates)
    if len(set(people + predicates)) != n + p:
        parser.error('people and predicates require distinct identities')
    positive = ['query_examples', 'fact_count', 'dimension', 'memory_tokens', 'width', 'layers',
                'heads', 'ffn_width', 'steps', 'query_steps', 'lexical_steps', 'fact_steps', 'memory_query_steps']
    if any(getattr(args, key) <= 0 for key in positive) or args.max_depth < 0 or args.checkpoint_every < 0:
        parser.error('recipe dimensions and update counts must be positive; depth/cadence may be zero')
    if args.width % args.heads or args.dimension % args.heads:
        parser.error('width and dimension must be divisible by heads')
    if args.fact_count < max(n, args.dimension + 1):
        parser.error('fact-count must cover people and dimension+1 affine observations')
    if not (0 < args.min_query_learning_rate <= args.query_learning_rate) or args.learning_rate <= 0 or args.readout_temperature <= 0:
        parser.error('learning rates and readout temperature require positive ordered values')
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    save(args.out / 'recipe.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    scene_none, instruction_none, request_none = n * n, (n + 1) ** 2 - 1, 2 * (n + 3)
    observations, paired = [], []

    def row(text, scene=scene_none, instruction=instruction_none, request=request_none):
        return dict(text=text, scene=scene, instruction=instruction, request=request)

    for verb in predicates:
        for a in range(n):
            for b in range(n):
                active = row(f'{people[a]} {verb} {people[b]}.', scene=a * n + b)
                passive = row(f'{people[b]} was {verb} by {people[a]}.', scene=a * n + b)
                observations.extend([active, passive]); paired.append((active, passive))
                if a != b:
                    reverse = row(f'{people[a]} was {verb} by {people[b]}.', scene=b * n + a)
                    observations.append(reverse); paired.append((active, reverse))
    for focus in range(n + 1):
        for exclude in range(n + 1):
            if focus == exclude == n:
                continue
            if focus == n:
                texts = [f'Ignore {people[exclude]}.', f'Exclude {people[exclude]}.']
            elif exclude == n:
                texts = [f'Use {people[focus]}.', f'Answer for {people[focus]}.']
            else:
                texts = [f'Use {people[focus]} and ignore {people[exclude]}.',
                         f'Answer for {people[focus]} but exclude {people[exclude]}.']
            rs = [row(t, instruction=focus * (n + 1) + exclude) for t in texts]
            observations.extend(rs); paired.append((rs[0], rs[1]))
            if focus < n and exclude < n and focus != exclude:
                opposite = row(f'Ignore {people[focus]} and use {people[exclude]}.', instruction=exclude * (n + 1) + focus)
                observations.append(opposite); paired.append((rs[0], opposite))
    for goal in range(2):
        for selector in range(n + 3):
            descriptions = ([people[selector]] if selector < n else
                            ['the caller', 'the person who called someone'] if selector == n else
                            ['the called person', 'the person who was called'] if selector == n + 1 else
                            ['that person', 'the focused person'])
            rs = []
            for who in descriptions:
                texts = ([f'Where does {who} work?', f'What is the workplace of {who}?'] if goal == 0 else
                         [f'What item is at the workplace of {who}?', f'Which item is kept where {who} works?'])
                rs.extend(row(t, request=goal * (n + 3) + selector) for t in texts)
            observations.extend(rs); paired.append((rs[0], rs[-1]))
    none_texts = ['Hello.', 'What is the weather?', 'There is nothing to select.']
    observations.extend(row(t) for t in none_texts)

    def example(observation, role):
        return dict(text=observation['text'], people=people, predicates=[], label=observation[role])

    def model(identity, index, context=1):
        return dict(id=identity, width=args.width, layers=args.layers, heads=args.heads,
                    ffn_width=args.ffn_width, context=context, norm_epsilon=2 ** -23,
                    precision='f32', seed=args.seed + index)

    for index, role in enumerate(['scene', 'instruction', 'request', 'query', 'lexical', 'lexical']):
        spec = dict(role=role, model=model(f'sejong-{role}-{index}', index), steps=args.steps,
                    learning_rate=args.learning_rate, examples=[], pairs=[], phrases=[], meanings=[],
                    encoder=str(Path('.cache/minilm').resolve()), checkpoint_every=args.checkpoint_every,
                    loss={}, optimizer={}, batch={})
        if role in ('scene', 'instruction', 'request'):
            spec['examples'] = [example(o, role) for o in observations]
            spec['pairs'] = [dict(left=example(a, role), right=example(b, role), equivalent=a[role] == b[role]) for a, b in paired]
        elif role == 'query':
            spec['steps'] = args.query_steps
            # Balance path lengths, then sample unique ranks in each grammar stratum.
            base = 4 * p
            counts = [2 * n * base ** depth for depth in range(args.max_depth + 1)]
            remaining = min(args.query_examples, sum(counts))
            programs = []
            for depth, capacity in enumerate(counts):
                amount = min(capacity, remaining // (len(counts) - depth))
                remaining -= amount
                selected = set()
                for end in range(capacity - amount, capacity):
                    draw = rng.randrange(end + 1)
                    selected.add(end if draw in selected else draw)
                for rank in sorted(selected):
                    root, code = divmod(rank, base ** depth)
                    goal, anchor = divmod(root, n)
                    encoded = [0] * depth
                    for position in range(depth - 1, -1, -1):
                        code, encoded[position] = divmod(code, base)
                    programs.append((goal, anchor, encoded))
            for goal, anchor, encoded in programs:
                steps = [dict(predicate=op // 4, reverse=bool(op % 4 // 2), last=bool(op % 2)) for op in encoded]
                who = describe(people[anchor], steps, predicates)
                question = f'Where does {who} work?' if goal == 0 else f'What item is at the workplace of {who}?'
                spec['examples'].append(dict(text=question, people=people, predicates=predicates, label=0,
                                             program=dict(goal=goal, anchor=anchor, steps=steps)))
            spec['examples'].extend(dict(text=t, people=people, predicates=predicates, label=0, program=None) for t in none_texts)
        else:
            spec['steps'] = args.lexical_steps
            spec['meanings'] = ['use', 'ignore']
            prefixes = ['prefer', 'omit'] if index == 4 else ['favor', 'skip']
            spec['phrases'] = [prefix + letters(i) for prefix in prefixes for i in range(n)]
            spec['examples'] = [dict(text=t, people=[], predicates=[], label=i // n) for i, t in enumerate(spec['phrases'])]
        save(args.out / f'role-{index}.json', spec)

    count, d = args.fact_count, args.dimension
    catalog = list(people)
    candidate = 0
    identities = set(catalog)
    while len(catalog) < count:
        name = f'person{letters(candidate)}'
        candidate += 1
        if name not in identities:
            catalog.append(name)
            identities.add(name)
    sites = [f'place{letters(i)}' for i in range(count)]
    items = [f'item{letters(i)}' for i in range(count)]
    items[-1] = {'hex': 'ff00fe'}
    for stage, (requests, values) in enumerate([(catalog, sites), (sites, items)]):
        contexts, queries = [], []
        for i in range(count):
            request = [0.0] * d
            if i < d:
                request[i] = 1.0
            elif i > d:
                request = [rng.gauss(0, 1 / math.sqrt(d)) for _ in range(d)]
            queries.append(request)
            contexts.append(request + [rng.gauss(0, 1 / math.sqrt(d)) for _ in range((args.memory_tokens - 1) * d)])
        targets = []
        for _ in values:
            target = [rng.gauss(0, 1) for _ in range(d)]
            norm = math.sqrt(sum(x * x for x in target))
            targets.append([x / norm for x in target])
        config = dict(id=f'factual-{stage}', dimension=d, tokens=args.memory_tokens, layers=args.layers,
                      heads=args.heads, hidden=args.width, ffn_width=args.ffn_width, norm_epsilon=2 ** -23,
                      readout_temperature=args.readout_temperature, residual_scale=1 / math.sqrt(2 * args.layers), seed=args.seed + 6 + stage)
        spec = dict(config=config, requests=requests, values=values, contexts=contexts, target_queries=queries,
                    value_targets=targets, labels=[(i + stage) % count for i in range(count)],
                    query_steps=args.memory_query_steps, fact_steps=args.fact_steps, learning_rate=args.learning_rate,
                    query_learning_rate=args.query_learning_rate, min_query_learning_rate=args.min_query_learning_rate,
                    checkpoint_every=args.checkpoint_every, query_loss={}, background={'kind': 'empirical'}, optimizer={}, batch={})
        save(args.out / f'{"a" if stage == 0 else "b"}.json', spec)
    prompts = [f'What item is at {s}?' for s in sites]
    save(args.out / 'native-b.json', dict(model=model('native-value-b', 8, max(len(t.encode()) for t in prompts) + 2),
         requests=sites, prompts=prompts, values=items, labels=[(i + 1) % count for i in range(count)],
         steps=args.fact_steps, learning_rate=args.learning_rate, checkpoint_every=args.checkpoint_every, optimizer={}, batch={}))
    save(args.out / 'assembly.json', dict(scene='role-0/model', instruction='role-1/model', request='role-2/model',
         query='role-3/model', lexica=['role-4/model', 'role-5/model'], a={'kind': 'memory', 'path': 'a/model'},
         b={'kind': 'memory', 'path': 'b/model'}, catalog=catalog, budgets={}))
    first, second, third = people[0], people[1 % n], people[2 % n]
    verb, next_verb = predicates[0], predicates[1 % p]
    story = [f'{first} {verb} {second}.', f'{second} {next_verb} {third}.']
    chain = [dict(predicate=0, reverse=False, last=False), dict(predicate=1 % p, reverse=False, last=False)]
    save(args.out / 'query.json', dict(question=f'{first} {verb} {second}. What item is at the workplace of the called person?', people=people, full=True))
    save(args.out / 'story.json', dict(question=f'Where does {describe(first, chain[:1], predicates)} work?', story=story, people=people, predicates=predicates, full=True))
    save(args.out / 'story-two.json', dict(question=f'What item is at the workplace of {describe(first, chain, predicates)}?', story=story, people=people, predicates=predicates, full=True))
    save(args.out / 'lexical.json', dict(question=f'prefera {first}. What item is at the workplace of that person?', people=people, full=True))
    renamed_people = catalog[-n:]
    renamed_first, renamed_second = renamed_people[0], renamed_people[1 % n]
    save(args.out / 'renamed.json', dict(question=f'{renamed_first} {verb} {renamed_second}. What item is at the workplace of the called person?', people=renamed_people, full=True))
    save(args.out / 'joint.json', dict(joint=[first, first, second], goal=1))
    save(args.out / 'revision.json', dict(labels=[(i + 2) % count for i in range(count)], steps=args.fact_steps,
         learning_rate=args.learning_rate, checkpoint_every=args.checkpoint_every, optimizer={}, batch={}))
    save(args.out / 'weights.json', dict(inputs=[contexts[0]], route=['b', 'a'], modes=['full', 'direct']))
    save(args.out / 'gold.json', {'flat': items[(1 % n + 1) % count], 'story': sites[1 % n], 'story-two': items[(2 % n + 1) % count],
         'lexical': items[1 % count], 'renamed': items[(count - n + 1 % n + 1) % count],
         'revision': items[(1 % n + 2) % count], 'catalog': catalog, 'sites': sites, 'items': items})
    print(args.out)


if __name__ == '__main__':
    main()
