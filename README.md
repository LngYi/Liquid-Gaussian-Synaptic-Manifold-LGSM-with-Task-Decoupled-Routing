# Liquid-Gaussian-Synaptic-Manifold-LGSM-with-Task-Decoupled-Routing
### Overcoming Catastrophic Forgetting Through Dynamic Experts

These are some fascinating results of my private research.

## Results
### Liquid-Vision
```bash
Scanning Iso: 0.0010... Acc A: 0.7176
Scanning Iso: 0.0116... Acc A: 0.6494
Scanning Iso: 0.0223... Acc A: 0.5229
Scanning Iso: 0.0329... Acc A: 0.8164
Scanning Iso: 0.0436... Acc A: 0.6873
Scanning Iso: 0.0542... Acc A: 0.6843
Scanning Iso: 0.0649... Acc A: 0.8119
Scanning Iso: 0.0755... Acc A: 0.5683
Scanning Iso: 0.0861... Acc A: 0.6829
Scanning Iso: 0.0968... Acc A: 0.6733
Scanning Iso: 0.1074... Acc A: 0.7786
Scanning Iso: 0.1181... Acc A: 0.7351
Scanning Iso: 0.1287... Acc A: 0.4583
Scanning Iso: 0.1394... Acc A: 0.5778
Scanning Iso: 0.1500... Acc A: 0.7578
Scanning Iso: 1.0000... Acc A: 0.6462

Iso      | Accuracy
---------|---------
0.0010   | ██████████████ 0.72
0.0329   | ████████████████████ 0.82
0.0649   | ███████████████████ 0.81
0.1287   | █████████ 0.46
1.0000   | █████████████ 0.65
```

### Liquid-Transformer
```bash
==================================================
DIAGNOSIS: MULTI-TASK NEURAL DISTRIBUTION
==================================================
layers.0.qkv         | Core-Zone:  77 | Transfer-Zone: 153
layers.0.attn_out    | Core-Zone:   1 | Transfer-Zone:  26
layers.0.ffn         | Core-Zone:   3 | Transfer-Zone: 128
layers.1.qkv         | Core-Zone:  72 | Transfer-Zone: 151
layers.1.attn_out    | Core-Zone:   0 | Transfer-Zone:  28
layers.1.ffn         | Core-Zone:  13 | Transfer-Zone: 140
layers.2.qkv         | Core-Zone:  49 | Transfer-Zone: 150
layers.2.attn_out    | Core-Zone:   0 | Transfer-Zone:  37
layers.2.ffn         | Core-Zone:  21 | Transfer-Zone: 111
layers.3.qkv         | Core-Zone:  45 | Transfer-Zone: 108
layers.3.attn_out    | Core-Zone:   0 | Transfer-Zone:  43
layers.3.ffn         | Core-Zone:  48 | Transfer-Zone: 111
layers.4.qkv         | Core-Zone:  30 | Transfer-Zone: 103
layers.4.attn_out    | Core-Zone:   0 | Transfer-Zone:  32
layers.4.ffn         | Core-Zone:  29 | Transfer-Zone: 108
layers.5.qkv         | Core-Zone:  95 | Transfer-Zone: 110
layers.5.attn_out    | Core-Zone:   5 | Transfer-Zone:  35
layers.5.ffn         | Core-Zone:  11 | Transfer-Zone: 122
```

- ### HEAD 0 (Story):
  
  Prompt:
  Once upon a time ...
  
  Result:
  Once upon a time, a small bird enough boat on them of different places on treasure."

  So's dad opened her and hair now boat out of places that everyone about milk together the taxi make many pictures not dad with fun together not family favorite too hard work no horn." her dad said    that green family both each shiny enough boat shiny

- ### HEAD 1 (Code) :
  Prompt:
  def find_max(list): ...

  Result:
  def find_max(list): the result is not equal to compute if examine if also then works are fine logic are used it also examine if not be fine #D==... is defined if not examine if also sorts of y are    greater than 4 or you contain are also also also also contain also fine if everything can is fine are

- ### HEAD 2 (Wiki) :
  Prompt:
  The capital city is ...
  
  Result:
  The capital city is a list of a public term written by the United States Department of the Catholic Church. It was first of the United States Marine Corps, when the same name.

  The first election was created in 1871 to 1885 and were a "In 1866", which was killed by a post-

