# Reasoner weights

Release assets of this repository. Each zip contains its own `SHA256SUMS.txt` for the files inside.

| Asset | Size (bytes) | SHA-256 |
|---|---:|---|
| `power_flow_reasoner_weights.zip` | 80,173,186 | `16ae5ff2e5f24f82992421b2223500db9a9b8c84bcd301ab599e08a158f63b8c` |
| `reactive_adjustment_reasoner_weights.zip` | 5,139,187 | `6803b7efb4d21073ad102323aa475b717571c3a2f9d9579262c91967d5a6716b` |
| `mode_generation_weights.zip` | 81,057,216 | `d42d093a5e60a73d3c063887d29b0f0a6c9b92be6310bf3b91793510269a6fd2` |
| `tsa_reasoner_weights.zip` | 72,561,229 | `eacfda267a5b865f858db54cc0e0ed17030230da5d0fe47a1d1c00b369e3bd2d` |
| `tsa_adaptation_ablation_weights.zip` | 765,147,800 | `db267ef3e2609f37c1ae000bc23dceca0277008bc9291e8ea407ab524d2fe690` |

The encoder weights are distributed in the separate repository
[gridfm-encoder-v1](https://github.com/goldenlcq-debug/gridfm-encoder-v1) (release `v1.0`,
`encoder_v1_weights_20260907.zip`, SHA-256 `885b2c2318812eb4a29cde89340441bc270975905fe96bcd718fd595c8001982`).

## Files inside the assets

### `power_flow_reasoner_weights.zip`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `power_flow_reasoner/seed0/best.pt` | 29,606,903 | `8472908037401db84bdfc1ec754e4c5ea8e024e1894b8e5f6c1779c405125e68` |
| `power_flow_reasoner/seed1/best.pt` | 29,606,903 | `f768f1b9e45d552457a96e3f49241c8408c1f0f2bccf69ca77c745386a822482` |
| `power_flow_reasoner/seed2/best.pt` | 29,606,903 | `2c272219a4f5c9a81fbb2c70b3b0fe40fd1d47653f296a160cdb6bc35fab771e` |
| `power_flow_reasoner/seed0/pf_report.json` | 3,595 | `6abe78311ca4bb94238b950847484ae8893906623f9dd6b1ce01898b868bcce6` |
| `power_flow_reasoner/seed1/pf_report.json` | 3,600 | `858dc5ec69c24c8b075800c05620d19a609161f3ccea9c58970b567888c37aca` |
| `power_flow_reasoner/seed2/pf_report.json` | 3,604 | `737bc53d943389278c262ee262d1aed91d07b0cfc28fab1b9ce412fa97377add` |

### `reactive_adjustment_reasoner_weights.zip`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `reactive_adjustment_reasoner/best.pt` | 5,394,591 | `7205ea2266a0989c8b62a9db6b235688e7df05e406dfb7e698ad12c97489d725` |
| `reactive_adjustment_reasoner/train_report.json` | 2,082 | `88fb821287bb777335fe45fb50132173318c22407a95a7bea3189ea0914cd813` |

### `mode_generation_weights.zip`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `mode_generation/denoiser/denoiser.pt` | 78,547,074 | `331399b433f534a6589c6e67252e615a0f3932cfb7f311476e2cce9a171c9375` |
| `mode_generation/denoiser/train_report.json` | 936 | `14a7997351b250fc167a2c0178386a8f7a5ae7baf8d27e27f1f9cc74e2645437` |
| `mode_generation/readout/readout.pt` | 8,941,602 | `cc505312807be31270a4bfa7f5e7f47563b3eb12a4495386698b29b39cccdafc` |
| `mode_generation/readout/train_report.json` | 246 | `a0832b718d00c077160d9e433ee97324ec9062995126f8268d4d9670f5a104d0` |
| `mode_generation/prescreener/deploy_head.json` | 3,864 | `916f50d9826d43b2aa769a4550c49cd4918beb5f8619ff2ef4caae1aa2b24947` |

### `tsa_reasoner_weights.zip`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `tsa_reasoner/seed0/best.pt` | 15,743,082 | `7e5a72347902b60b5d467c8b8ab8091791404d96fbb9b6ef83b6691108f9c471` |
| `tsa_reasoner/seed1/best.pt` | 15,745,258 | `b8b96c76c584d6170346e59a1e1708f521a57eb005bf45a6a226f6b4ed042076` |
| `tsa_reasoner/seed2/best.pt` | 15,743,274 | `d6e7c1ef7ca409ea3a1275fb31ae3020b35d7bb592af5cc28145b2e1427b7097` |
| `tsa_reasoner/seed3/best.pt` | 15,743,082 | `0a7cee8615ce0464cd0e8ccc261a77e18a08f0c7b1f745b8560ba5a2c96e6084` |
| `tsa_reasoner/seed4/best.pt` | 15,741,290 | `04b4ac5bae06d56237d0fcd559f753cb12aa087cee541f651c60a49ec94490f9` |
| `tsa_reasoner/seed0/ts_report.json` | 30,853 | `469cac5f31edd9aa60d22513169e1ea09c0e84592f02645e4cafbc9486724d94` |
| `tsa_reasoner/seed1/ts_report.json` | 33,039 | `7e51c8c17dc6fa8cefeb88945b5dacd3fe1bed0a31e9c9f9328777b0e17aa744` |
| `tsa_reasoner/seed2/ts_report.json` | 29,860 | `055fcbd0117a69229af355074de90c5b36bfddef3e34c654c71310a623e75fc6` |
| `tsa_reasoner/seed3/ts_report.json` | 29,678 | `da65e58d9a4dec148640edd00232527eaacf45211b4a74a127c9d156291c7783` |
| `tsa_reasoner/seed4/ts_report.json` | 27,728 | `0e6222575c47838e1b5fe611ace54dbddf527a4efd0afcd7f084dc78e9c46de9` |
| `tsa_reasoner/calib_dose_b.json` | 7,606 | `7769354e1b85f2512acb0d8fa9c099a05eaf67642f86e09a7cbdd713b8a227b9` |

### `tsa_adaptation_ablation_weights.zip`

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `tsa_adaptation_ablation/frozen/seed0/best.pt` | 15,739,242 | `9e839ad0d773f7c131d3d835c8b6d183e75e261354fd0e00b135f6a1b10ec9ed` |
| `tsa_adaptation_ablation/frozen/seed0/ts_report.json` | 24,426 | `9ecf863c95348120e82c0e54b5b86f58b2a3c14e6f4f1f041e5042d79996d0b8` |
| `tsa_adaptation_ablation/frozen/seed1/best.pt` | 15,740,778 | `cdbe13216055084e0225496c79fb380a98b24a6e5a06937be9ad737fc803f8b2` |
| `tsa_adaptation_ablation/frozen/seed1/ts_report.json` | 25,392 | `9572f372fcefc1c1d402c67dd6be7ee04388deda0cef15f342adaa47c6f09ddf` |
| `tsa_adaptation_ablation/frozen/seed2/best.pt` | 15,740,394 | `f3e4bc91267490c2e15cb3b1347e4665e289070ba016a0cb1dc662aaf7a3930a` |
| `tsa_adaptation_ablation/frozen/seed2/ts_report.json` | 26,305 | `17fe083d3014138f39b25d5938910ada710581f6ec0c1fddcc2df4419ea35fd9` |
| `tsa_adaptation_ablation/frozen/seed3/best.pt` | 15,741,162 | `72a2980caf3f803b799d530a986285e38c5f9f8cfd3d67ec6589e50f95c819ae` |
| `tsa_adaptation_ablation/frozen/seed3/ts_report.json` | 27,381 | `1d765f4a986d52f940c85bf9d8ae932f0af4e5241edb76ad9c7a8f09961e62d8` |
| `tsa_adaptation_ablation/frozen/seed4/best.pt` | 15,739,818 | `463a26298efd542b4d15383955d2489a8f842c5579f438f5bd41e944a0cea348` |
| `tsa_adaptation_ablation/frozen/seed4/ts_report.json` | 25,325 | `ccf08efcd439c39a36d57993d3113b5850f8d784cbf0758c436d6fb4726f4040` |
| `tsa_adaptation_ablation/full/seed0/best.pt` | 137,224,026 | `faea7a9188175974e26a488a85fded16d6dbbb506098b8da42fd5c9a16045fd8` |
| `tsa_adaptation_ablation/full/seed0/ts_report.json` | 35,095 | `6dd73c5b5200252b8bb04b9992576db73a85ae49c9c92abec9ff177093c0f1a0` |
| `tsa_adaptation_ablation/full/seed1/best.pt` | 137,224,986 | `5c640c6ee5d85aec83f34f74dc6494a2b4a411cf7c14fc103dc81dc7eb5a2702` |
| `tsa_adaptation_ablation/full/seed1/ts_report.json` | 36,675 | `54da1cbcc7cc027a58ddc7f2090b863fb8d46201ad7979bf3edf5b2c5bba4f68` |
| `tsa_adaptation_ablation/full/seed2/best.pt` | 137,228,954 | `591161a64276dcc75be181713d120fd5bafd1a770c665c923e437e75b45625f6` |
| `tsa_adaptation_ablation/full/seed2/ts_report.json` | 36,929 | `90a8b1ab526b54fb11eaefd9cb83943f7ffed5fead05ec3162d2b214a6bd28b5` |
| `tsa_adaptation_ablation/full/seed3/best.pt` | 137,223,130 | `85811e2e847c10f7dc3fa50ec2019a2d7a80bbc17caa655dcfeac1eb21c52bfa` |
| `tsa_adaptation_ablation/full/seed3/ts_report.json` | 33,635 | `25b543071317a8aaaf8c2377a0a46abeadf6fc1a7a430f96a0d05bb0e17129b9` |
| `tsa_adaptation_ablation/full/seed4/best.pt` | 137,229,850 | `3625901cb6ef3d145da69e2639f0aea04c772ade6056eeffdaecb154566c4cd9` |
| `tsa_adaptation_ablation/full/seed4/ts_report.json` | 38,121 | `307bf5c80cdc668a2c7ede5bd7b4fee4bf721310a2c1b0acae92e7a5ce8a44da` |

All checkpoints are plain PyTorch files holding state dictionaries (`reasoner_state`, `ema_state` or `decoder_state`; the fully
adapted TSA checkpoints also hold `encoder_state`), training state and configuration values; `deploy_head.json` holds the prescreener coefficients. Verify a download with `sha256sum <file>`
(Linux / macOS) or `certutil -hashfile <file> SHA256` (Windows).
