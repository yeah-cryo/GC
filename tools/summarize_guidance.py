import argparse
import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description='Export classifier-guidance scores and a matched grid.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preview-seeds', type=int, default=4)
    args = parser.parse_args()
    root = Path(args.output)
    summary = json.loads((root / 'summary.json').read_text())
    if not summary['complete'] or not summary['frozen_models_unchanged']:
        raise RuntimeError('Experiment is incomplete or its frozen-model audit failed.')
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    strengths = [int(value) if float(value).is_integer() else value for value in summary['rows']]
    seeds = summary['columns']
    lookup = {(int(row['strength']) if float(row['strength']).is_integer() else row['strength'], row['seed']): row
              for row in rows}
    expected = {(strength, seed) for strength in strengths for seed in seeds}
    if set(lookup) != expected:
        raise RuntimeError('Metrics do not contain exactly one row per strength and seed.')

    with (root / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'fake_probabilities_by_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['seed', 'prompt_index', 'prompt'] +
                        [f'strength_{strength}_fake_probability' for strength in strengths])
        for seed in seeds:
            first = lookup[strengths[0], seed]
            writer.writerow([seed, first.get('prompt_index'), first.get('prompt')] +
                            [lookup[strength, seed]['saved_png_fake_probability'] for strength in strengths])

    tile, label_height, margin = 256, 70, 12
    selected = seeds[:args.preview_seeds]
    canvas = Image.new('RGB',
                       (len(strengths) * (tile + margin) + margin,
                        len(selected) * (tile + label_height + margin) + margin), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = ImageFont.load_default()
    for y, seed in enumerate(selected):
        for x, strength in enumerate(strengths):
            left = margin + x * (tile + margin)
            top = margin + y * (tile + label_height + margin)
            with Image.open(root / f'strength_{strength:g}_seed_{seed}.png') as image:
                canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS), (left, top))
            row = lookup[strength, seed]
            caption = textwrap.shorten(row.get('prompt', ''), width=42, placeholder='…')
            draw.text((left, top + tile + 2), f'Seed {seed} | strength {strength}', fill='black', font=font)
            draw.text((left, top + tile + 18), f'Fake probability: {row["saved_png_fake_probability"]:.2%}',
                      fill='black', font=font)
            draw.multiline_text((left, top + tile + 34), textwrap.fill(caption, width=38),
                                fill='black', font=font, spacing=1)
    canvas.save(root / 'matched_comparison.jpg', quality=95)
    print(json.dumps(summary['summary'], indent=2))


if __name__ == '__main__':
    main()
