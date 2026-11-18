#!/usr/bin/env python3
"""Rebuild this report's charts (plots/*.svg) and headline numbers (data/summary.json) from data/results.json.

Standard library only, Python 3.9 or newer:  python3 code/make_report.py
The calculations are the ones README.md describes:
  * check scores are averaged over the two test tokenizers at each step and rounded to three decimals;
  * a 500-update block has a favorite only if its most-picked tokenizer got at least 10% of the block's picks,
    and neighbouring blocks with the same favorite are joined into one stretch;
  * rounding is half-up, as in JavaScript's Math.round.
"""
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
TABLES = json.loads((ROOT / 'data' / 'results.json').read_text())['tables']
METHODS = ('BPE', 'Unigram', 'WordPiece')
# Okabe-Ito colours, chosen to stay distinguishable with colour-vision deficiency; grey means no clear favorite.
COLOR = {'BPE': '#0072B2', 'Unigram': '#E69F00', 'WordPiece': '#009E73', None: '#BBBBBB'}
INK, MUTED, GRID = '#1A202C', '#4A5568', '#E2E8F0'
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
SPLIT_WORDS = {'none': 'no pre-splitting', 'words': 'split at words', 'lines': 'split at lines'}
LAST_STEP = 20000


def half_up(x):
    return math.floor(x + 0.5)


def score_by_step():
    by_step = {}
    for row in TABLES['evals']:
        by_step.setdefault(row['step'], []).append(row)
    scores = []
    for step in sorted(by_step):
        rows = by_step[step]
        test = half_up(1000 * sum(r['unseen_bpb'] for r in rows) / len(rows))
        own = half_up(1000 * sum(r['own_bpb'] for r in rows) / len(rows))
        scores.append({'step': step, 'test_bpb': test / 1000, 'own_bpb': own / 1000, 'gap_bpb': (test - own) / 1000})
    return scores


def headline(scores, early_step=2000):
    first, last = scores[0], scores[-1]
    early = next(r for r in scores if r['step'] == early_step)
    return {
        'start_test_bpb': first['test_bpb'], 'end_test_bpb': last['test_bpb'],
        'test_drop_pct': half_up(100 * (first['test_bpb'] - last['test_bpb']) / first['test_bpb']),
        'start_own_bpb': first['own_bpb'], 'end_own_bpb': last['own_bpb'],
        'start_gap_bpb': first['gap_bpb'], 'end_gap_bpb': last['gap_bpb'],
        'gap_drop_pct': half_up(100 * (first['gap_bpb'] - last['gap_bpb']) / first['gap_bpb']),
        'early_step': early_step,
        'share_of_drop_by_early_step_pct': half_up(
            100 * (first['test_bpb'] - early['test_bpb']) / (first['test_bpb'] - last['test_bpb'])),
    }


def stretches(min_share_pct=10):
    tokenizers = {t['tokenizer']: t for t in TABLES['tokenizers']}
    rank = {d['tokenizer']: d['difficulty_rank'] for d in TABLES['difficulty']}
    joined = []
    for block in TABLES['segments']:
        size = block['to_step'] - block['from_step'] + 1
        favorite = block['favorite_tokenizer'] if 100 * block['favorite_picks'] / size >= min_share_pct else None
        if joined and joined[-1]['favorite'] == favorite:
            joined[-1]['to_step'] = block['to_step']
            joined[-1]['favorite_picks'] += 0 if favorite is None else block['favorite_picks']
        else:
            joined.append({'from_step': block['from_step'], 'to_step': block['to_step'], 'favorite': favorite,
                           'favorite_picks': 0 if favorite is None else block['favorite_picks']})
    rows = []
    for s in joined:
        t = tokenizers.get(s['favorite']) if s['favorite'] is not None else None
        rows.append({
            'from_step': s['from_step'], 'to_step': s['to_step'], 'favorite': s['favorite'],
            'method': t['method'] if t else None, 'pre_split': t['pre_split'] if t else None,
            'vocab_size': t['vocab_size'] if t else None,
            'share_pct': half_up(100 * s['favorite_picks'] / (s['to_step'] - s['from_step'] + 1)) if t else None,
            'difficulty_rank_on_qingyun_model': rank.get(s['favorite']) if t else None,
        })
    return rows


def method_totals():
    picks = TABLES['method_picks']
    all_picks = sum(r['picks'] for r in picks)
    pool = TABLES['tokenizers']
    return [{'method': m, 'picks': sum(r['picks'] for r in picks if r['method'] == m),
             'picks_pct': half_up(100 * sum(r['picks'] for r in picks if r['method'] == m) / all_picks),
             'pool_count': sum(1 for t in pool if t['method'] == m),
             'pool_pct': half_up(100 * sum(1 for t in pool if t['method'] == m) / len(pool))} for m in METHODS]


def total_gpu_hours():
    return half_up(10 * sum(j['gpu_hours'] for j in TABLES['jobs'])) / 10


# ---------------------------------------------------------------- drawing


def scale(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) * (r1 - r0) / (d1 - d0)


class Svg:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="{FONT}" font-size="12" role="img">',
            f'<title>{escape(title)}</title>', f'<desc>{escape(desc)}</desc>',
            f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>']

    def add(self, element):
        self.parts.append(element)

    def text(self, x, y, s, anchor='start', fill=MUTED, size=12, weight='normal'):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" '
                 f'font-weight="{weight}">{escape(s)}</text>')

    def rect(self, x, y, w, h, fill, extra=''):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"{extra}/>')

    def save(self, name):
        (ROOT / 'plots' / name).write_text('\n'.join(self.parts + ['</svg>']) + '\n')


def step_axis(svg, x, y_base, left, right, title='Training update'):
    svg.add(f'<line x1="{left}" x2="{right}" y1="{y_base:.1f}" y2="{y_base:.1f}" stroke="{MUTED}"/>')
    for v in (0, 5000, 10000, 15000, 20000):
        svg.add(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{y_base:.1f}" y2="{y_base + 5:.1f}" stroke="{MUTED}"/>')
        svg.text(x(v), y_base + 18, f'{v:,}', 'middle')
    svg.text((left + right) / 2, y_base + 38, title, 'middle')


def y_title(svg, text, top, bottom):
    svg.add(f'<text transform="translate(16,{(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" '
            f'fill="{MUTED}">{escape(text)}</text>')


def legend(svg, x, y, items):
    for label, color in items:
        svg.rect(x, y - 10, 12, 12, color, ' rx="2"')
        svg.text(x + 17, y, label, fill=INK)
        x += 17 + 7.2 * len(label) + 18


def chart_scores(scores, head):
    width, height, left, right, top, bottom = 720, 330, 64, 540, 20, 270
    x = scale(0, LAST_STEP, left, right)
    y = scale(0.9, 1.45, bottom, top)
    svg = Svg(width, height, 'Check score during training',
              f'Line chart every 500 updates. The average score on the two test tokenizers falls from '
              f'{head["start_test_bpb"]:.3f} to {head["end_test_bpb"]:.3f} bits per byte; the score with the '
              f'model\'s own tokenizer goes from {head["start_own_bpb"]:.3f} to {head["end_own_bpb"]:.3f}.')
    for v in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4):
        svg.add(f'<line x1="{left}" x2="{right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
        svg.text(left - 8, y(v) + 4, f'{v:.1f}', 'end')
    step_axis(svg, x, bottom, left, right)
    y_title(svg, 'Bits per byte (lower is better)', top, bottom)
    lines = (('test_bpb', COLOR['BPE'], '', 'Test tokenizers (average of 2)'),
             ('own_bpb', MUTED, ' stroke-dasharray="6 4"', 'Model\'s own tokenizer'))
    for key, color, dash, label in lines:
        points = ' '.join(f'{x(r["step"]):.1f},{y(r[key]):.1f}' for r in scores)
        svg.add(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" '
                f'stroke-linejoin="round"{dash}/>')
        end = scores[-1][key]
        svg.text(right + 10, y(end) - 2, label, fill=INK)
        svg.text(right + 10, y(end) + 13, f'start {scores[0][key]:.3f}, end {end:.3f}')
    svg.save('test_score.svg')


def chart_focus(focus, rows):
    width, height, left, right, top, bottom = 720, 320, 64, 690, 64, 260
    x = scale(0, LAST_STEP, left, right)
    y = scale(0, 1, bottom, top)
    svg = Svg(width, height, 'How strongly the chooser focused on one tokenizer',
              'Area chart, every 50 updates, of the chooser\'s highest chance for any one tokenizer. A strip above '
              'it colours each stretch by the kind of its most-picked tokenizer.')
    legend(svg, left, 16, [('BPE', COLOR['BPE']), ('Unigram', COLOR['Unigram']),
                           ('WordPiece', COLOR['WordPiece']), ('No clear favorite', COLOR[None])])
    svg.text(left, 38, 'Strip: kind of the most-picked tokenizer in each stretch', size=11)
    for r in rows:
        x0, x1 = x(r['from_step'] - 1), x(r['to_step'])
        svg.rect(x0 + 1, 44, max(1.0, x1 - x0 - 2), 12, COLOR[r['method']], ' rx="2"')
    for v in (0, 0.25, 0.5, 0.75, 1):
        svg.add(f'<line x1="{left}" x2="{right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
        svg.text(left - 8, y(v) + 4, f'{round(100 * v)}%', 'end')
    path = [f'M{x(focus[0]["step"]):.1f},{y(0):.1f}']
    path += [f'L{x(r["step"]):.1f},{y(r["favorite_chance"]):.1f}' for r in focus]
    path.append(f'L{x(focus[-1]["step"]):.1f},{y(0):.1f}Z')
    svg.add(f'<path d="{" ".join(path)}" fill="{INK}" fill-opacity="0.10"/>')
    points = ' '.join(f'{x(r["step"]):.1f},{y(r["favorite_chance"]):.1f}' for r in focus)
    svg.add(f'<polyline points="{points}" fill="none" stroke="{INK}" stroke-width="1.5" stroke-linejoin="round"/>')
    step_axis(svg, x, bottom, left, right)
    y_title(svg, 'Highest chance of any one tokenizer', top, bottom)
    svg.save('chooser_focus.svg')


def chart_kinds(totals):
    width, height, left, right, top, bottom = 720, 320, 64, 690, 44, 260
    x = scale(0, LAST_STEP, left, right)
    y = scale(0, 500, bottom, top)
    picks = {(r['from_step'], r['method']): r['picks'] for r in TABLES['method_picks']}
    blocks = sorted({(r['from_step'], r['to_step']) for r in TABLES['method_picks']})
    svg = Svg(width, height, 'Kinds of tokenizer the chooser picked',
              'Stacked bars, one per 500 updates, of how many picks went to BPE, Unigram and WordPiece tokenizers.')
    share = {t['method']: t['picks_pct'] for t in totals}
    legend(svg, left, 16, [(f'{m}: {share[m]}% of all picks', COLOR[m]) for m in METHODS])
    for v in (0, 100, 200, 300, 400, 500):
        svg.add(f'<line x1="{left}" x2="{right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}"/>')
        svg.text(left - 8, y(v) + 4, str(v), 'end')
    for start, end in blocks:
        x0, width_px, below = x(start - 1) + 1, x(end) - x(start - 1) - 2, 0
        for m in METHODS:
            n = picks[(start, m)]
            if n:
                svg.rect(x0, y(below + n), width_px, y(below) - y(below + n), COLOR[m],
                         ' stroke="#FFFFFF" stroke-width="1"')
                below += n
    step_axis(svg, x, bottom, left, right)
    y_title(svg, 'Picks in each 500-update block', top, bottom)
    svg.save('kinds_picked.svg')


def main():
    scores = score_by_step()
    head = headline(scores)
    rows = stretches()
    totals = method_totals()
    (ROOT / 'plots').mkdir(exist_ok=True)
    chart_scores(scores, head)
    chart_focus(TABLES['focus'], rows)
    chart_kinds(totals)
    summary = {'headline': head, 'favorite_stretches': rows, 'kinds_picked': totals,
               'total_gpu_hours': total_gpu_hours(), 'run': TABLES['run'][0], 'scores_every_500': scores}
    (ROOT / 'data' / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('headline', 'kinds_picked', 'total_gpu_hours')}, indent=1))
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
