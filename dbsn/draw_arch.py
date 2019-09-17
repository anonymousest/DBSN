import sys
from graphviz import Digraph
import torch
import math

colores_list = ["yellow3", "palevioletred3", "cyan3"]
alphas_path = "../work/runadags_lr3_decayto0.5_0/alphas100.pth" #"../dbsn_seg/dbsn_bn_pw0.1_clip_3_1gpu/checkpoint-850.pt" #
dataset = "cifar10" #"camvid0" #
if dataset == "cifar10" or dataset == "cifar100":
    op_start = 1
    alphas = torch.nn.functional.softmax(torch.load(alphas_path), 1).data.cpu().numpy()
else:
    op_start = 0
    alphas = torch.nn.functional.softmax(torch.load(alphas_path)["alphas"].chunk(2)[int(dataset[-1])], 1).data.cpu().numpy()


n = int(math.sqrt(2*alphas.shape[0])) + 1

g = Digraph(
  format='pdf',
  edge_attr=dict(fontsize='20', fontname="times"),
  node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='1.5', penwidth='2', fontname="times"),
  engine='dot')
g.body.extend(['rankdir=LR'])

with g.subgraph(name='child', node_attr={'shape': 'box', 'height': '0.01', 'style': 'invisible'}) as c:
    if dataset == "cifar10" or dataset == "cifar100":
        c.edge('none', 'foo0', style="invisible", fillcolor="white", color="white", label=" ")
        c.edge('foo0', 'bar0', fillcolor="grey90", label="conv_1×1",  penwidth=str(1))
    else:
        c.edge('foo0', 'bar0', fillcolor="grey90", label="conv_3×3",  penwidth=str(1))
    c.edge('bar0', 'bar1', color=colores_list[0], label="skip_connect",  penwidth=str(1))
    c.edge('bar1', 'bar2', color=colores_list[1], label="sep_conv_3×3", penwidth=str(1))
    c.edge('bar2', 'bar3', color=colores_list[2], label="dil_conv_3×3", penwidth=str(1))

g.node("c_{k-1}", fillcolor='darkseagreen2')
for i in range(n):
    g.node(str(i), fillcolor='lightblue')
    g.edge("c_{k-1}", str(i), fillcolor="grey90", penwidth=str(0.5))

cnt = 0
for i in range(1, n):
    for j in range(i):
        edges = alphas[cnt]
        for k in range(op_start, edges.shape[0]):
            g.edge(str(j), str(i), color=colores_list[k - op_start], penwidth=str(10*edges[k]))
        cnt += 1


g.render(dataset+"_arch", view=False)
