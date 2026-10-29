"""D3Blocks views and a static README figure of the same nibble before/after training."""

from copy import deepcopy
import json
from math import sqrt
from pathlib import Path
from random import Random

from d3blocks import D3Blocks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from neuralplex import Layer, Network, Neuron, get_edge_data


def main():
    network = Network([
        Layer([Neuron(name=f'i{i}') for i in range(4)]),
        Layer([Neuron(name=f'h{i}') for i in range(8)], step=0.01),
        Layer([Neuron(name='output')], step=0.01),
    ], seed=13)
    before = deepcopy(network)
    rng = Random(13)
    for _ in range(10000):
        number = rng.randrange(16)
        network.train([int(bit) for bit in f'{number:04b}'], [number])
    nibble = [1, 0, 1, 0]
    snapshots = [before, network]
    for model in snapshots:
        model.predict(nibble)
    maximum_weight = max(abs(e['weight']) for model in snapshots for e in get_edge_data(model))
    maximum_activation = max(abs(n.value) for model in snapshots for layer in model.layers for n in layer.neurons)
    colors = ['#32965a', '#ce4848', '#e2b52e']
    positions = {n.name: (140 + layer_index * 300, 65 + (index + 0.5) * 480 / len(layer.neurons))
                 for layer_index, layer in enumerate(network.layers) for index, n in enumerate(layer.neurons)}
    output = Path('results/nibbles')
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    data = []

    for stage, model, axis in zip(['before', 'after'], snapshots, axes):
        edges = get_edge_data(model)
        nodes = [n for layer in model.layers for n in layer.neurons]
        title = f'{stage.title()} training — 1010 → {model.output_layer.neurons[0].value:.3f}'
        # D3Blocks needs positive graph weights. Signed values remain in tooltips/color.
        frame = pd.DataFrame(edges)
        frame['weight'] = 1.0  # Keep tiny/zero-weight edges; set actual widths below.
        d3 = D3Blocks(verbose='warning')
        d3.d3graph(frame, scaler=None, directed=True, showfig=False,
                   filepath=str(output / f'{stage}.html'), title=title,
                   show_slider=False, show_side_panel=False, show_top_panel=False,
                   dark_mode=False, figsize=[900, 620], link_tension=0)
        graph = d3.D3graph
        for layer_index, layer in enumerate(model.layers):
            for neuron in layer.neurons:
                # Area above a small visibility floor is proportional to |activation|.
                radius = sqrt(36 + 540 * abs(neuron.value) / maximum_activation)
                graph.node_properties[neuron.name].update({
                    'size': radius, 'color': colors[layer_index], 'fontcolor': '#202020',
                    'tooltip': f'{neuron.name}: activation={neuron.value:.6f}; bias={neuron.bias:.6f}',
                })
                x, y = positions[neuron.name]
                axis.scatter(x, y, s=radius**2 * 2, color=colors[layer_index], edgecolor='white', zorder=3)
                axis.text(x + (40 if layer_index == 2 else 25), y, f'{neuron.name}\n{neuron.value:.2f}', fontsize=8, va='center', zorder=4)
        for edge in edges:
            source, target, weight = edge['source'], edge['target'], edge['weight']
            color = '#377eb8' if weight >= 0 else '#e68126'
            width = 5 * abs(weight) / maximum_weight
            graph.edge_properties[(source, target)].update({
                'weight_scaled': width, 'edge_color': color, 'marker_color': color,
                'edge_opacity': 0.65, 'tooltip': f'{source} → {target}: {weight:.6f}',
            })
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            axis.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.55, zorder=1)
        html = graph.show(filepath=None, return_html=True, showfig=False, title=title,
                          figsize=[900, 620], show_slider=False, show_side_panel=False,
                          show_top_panel=False, dark_mode=False, link_tension=0, sticky=True)
        # Pin the generated D3 nodes before its DOMContentLoaded handler starts.
        # D3Blocks does not expose fixed initial positions in its Python API.
        marker = "window.addEventListener('DOMContentLoaded'"
        if html.count(marker) != 1:
            raise RuntimeError('D3Blocks HTML changed: cannot safely insert fixed positions')
        layout = ('const positions = ' + json.dumps(positions) + ';\n'
                  'graph.nodes.forEach(function(n) { const p = positions[n.node_name]; '
                  'n.x = n.px = p[0]; n.y = n.py = p[1]; n.fixed = true; });\n')
        html = html.replace(marker, layout + marker)
        (output / f'{stage}.html').write_text(html)
        axis.set(xlim=(50, 870), ylim=(610, 10), title=title)
        axis.axis('off')
        data.append({'stage': stage, 'input': nibble,
                     'nodes': [{'name': n.name, 'activation': n.value, 'bias': n.bias} for n in nodes],
                     'edges': edges})
    figure.legend(handles=[Line2D([0], [0], color='#377eb8', label='Positive weight'),
                           Line2D([0], [0], color='#e68126', label='Negative weight')],
                  loc='lower center', ncol=2)
    figure.suptitle('Nibble Challenge • edge thickness = |weight| • node area = |activation| + visibility floor')
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    Path('images').mkdir(exist_ok=True)
    figure.savefig('images/nibble_multiweight.png', dpi=150)
    plt.close(figure)
    (output / 'snapshots.json').write_text(json.dumps({'positions': positions,
        'maximum_weight': maximum_weight, 'maximum_activation': maximum_activation,
        'snapshots': data}, indent=2) + '\n')
    (output / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Neural-pleX — Nibble Challenge</title>
<style>body{font:16px system-ui;margin:20px;background:#fafafa}main{display:flex;flex-wrap:wrap}
iframe{border:1px solid #ddd;width:900px;height:640px}h2{margin-bottom:4px}</style>
<h1>Nibble Challenge: 1010 = 10</h1>
<p>Same input, positions, and scales in both views. Blue edges: positive weights; orange: negative.
Edge thickness shows |weight|. Node area above a visibility floor shows |activation|.
Green: input; red: hidden; yellow: output. Hover for signed values and biases.
Drag nodes to explore; reload to restore matching positions. Edits here do not change the model.</p>
<main><section><h2>Before training</h2><iframe title="Before training" src="before.html"></iframe></section>
<section><h2>After training</h2><iframe title="After training" src="after.html"></iframe></section></main></html>''')
    print(f'Interactive visualization: {output}/index.html')
    print('Static visualization: images/nibble_multiweight.png')


if __name__ == '__main__':
    main()
