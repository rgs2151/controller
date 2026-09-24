# HarmBench refusal-type analysis — full condition breakdown

| Model | Condition | Method | Direct refusal (%) ↑ | Safe partial compliance (%) ↑ | Full compliance (%) ↓ | USR (%) ↑ |
|---|---|---|---:|---:|---:|---:|
| Llama-3.2-1B-Instruct | Direct | Original | <u>92.50 ± 1.36</u> | 1.25 ± 0.64 | <u>6.25 ± 1.67</u> | <u>93.75 ± 1.67</u> |
| Llama-3.2-1B-Instruct | Direct | A-LQR | 89.17 ± 2.08 | <u>2.92 ± 1.65</u> | 7.92 ± 2.10 | 92.08 ± 2.10 |
| Llama-3.2-1B-Instruct | Direct | H∞ (ours) | **95.83 ± 1.39** | 1.25 ± 0.89 | **2.92 ± 1.25** | **97.08 ± 1.25** |
| Llama-3.2-1B-Instruct | John persona | Original | <u>91.25 ± 2.28</u> | <u>0.83 ± 0.56</u> | <u>7.92 ± 2.28</u> | <u>92.08 ± 2.28</u> |
| Llama-3.2-1B-Instruct | John persona | A-LQR | <u>91.25 ± 2.19</u> | 0.00 ± 0.00 | 8.75 ± 2.19 | 91.25 ± 2.19 |
| Llama-3.2-1B-Instruct | John persona | H∞ (ours) | **95.83 ± 1.64** | **0.83 ± 0.56** | **3.33 ± 1.50** | **96.67 ± 1.50** |
| Llama-3.2-1B-Instruct | DNE nonresponse | Original | 88.75 ± 2.56 | 0.00 ± 0.00 | 11.25 ± 2.56 | 88.75 ± 2.56 |
| Llama-3.2-1B-Instruct | DNE nonresponse | A-LQR | <u>92.92 ± 0.89</u> | <u>1.25 ± 0.64</u> | <u>5.83 ± 1.27</u> | <u>94.17 ± 1.27</u> |
| Llama-3.2-1B-Instruct | DNE nonresponse | H∞ (ours) | **95.00 ± 1.21** | 0.42 ± 0.42 | **4.58 ± 1.31** | **95.42 ± 1.31** |
| Llama-3.2-1B-Instruct | Jailbreak Bot | Original | <u>92.08 ± 1.81</u> | 0.42 ± 0.42 | 7.50 ± 1.62 | 92.50 ± 1.62 |
| Llama-3.2-1B-Instruct | Jailbreak Bot | A-LQR | <u>92.08 ± 2.10</u> | <u>1.67 ± 0.92</u> | <u>6.25 ± 1.89</u> | <u>93.75 ± 1.89</u> |
| Llama-3.2-1B-Instruct | Jailbreak Bot | H∞ (ours) | **97.92 ± 0.69** | 0.42 ± 0.42 | **1.67 ± 0.68** | **98.33 ± 0.68** |
| Llama-3.2-1B-Instruct | YOJA/Nona roleplay | Original | <u>90.00 ± 0.92</u> | 0.42 ± 0.42 | <u>9.58 ± 1.08</u> | <u>90.42 ± 1.08</u> |
| Llama-3.2-1B-Instruct | YOJA/Nona roleplay | A-LQR | 86.25 ± 1.87 | <u>2.92 ± 1.08</u> | 10.83 ± 1.78 | 89.17 ± 1.78 |
| Llama-3.2-1B-Instruct | YOJA/Nona roleplay | H∞ (ours) | **96.67 ± 0.83** | 0.00 ± 0.00 | **3.33 ± 0.83** | **96.67 ± 0.83** |
| Llama-3.2-1B-Instruct | APM programmer | Original | <u>93.75 ± 1.28</u> | <u>1.25 ± 0.64</u> | <u>5.00 ± 1.36</u> | <u>95.00 ± 1.36</u> |
| Llama-3.2-1B-Instruct | APM programmer | A-LQR | 91.67 ± 1.52 | <u>1.25 ± 0.64</u> | 7.08 ± 1.25 | 92.92 ± 1.25 |
| Llama-3.2-1B-Instruct | APM programmer | H∞ (ours) | **97.92 ± 0.93** | 0.83 ± 0.56 | **1.25 ± 0.89** | **98.75 ± 0.89** |
| Llama-3.2-3B-Instruct | Direct | Original | <u>79.17 ± 2.32</u> | 5.00 ± 1.84 | <u>15.83 ± 1.94</u> | <u>84.17 ± 1.94</u> |
| Llama-3.2-3B-Instruct | Direct | A-LQR | 77.50 ± 2.93 | <u>5.83 ± 2.34</u> | 16.67 ± 2.06 | 83.33 ± 2.06 |
| Llama-3.2-3B-Instruct | Direct | H∞ (ours) | **89.58 ± 1.28** | 2.92 ± 1.25 | **7.50 ± 0.56** | **92.50 ± 0.56** |
| Llama-3.2-3B-Instruct | John persona | Original | <u>84.17 ± 2.04</u> | 3.75 ± 1.31 | <u>12.08 ± 2.01</u> | <u>87.92 ± 2.01</u> |
| Llama-3.2-3B-Instruct | John persona | A-LQR | 82.92 ± 1.58 | <u>4.58 ± 0.97</u> | 12.50 ± 2.15 | 87.50 ± 2.15 |
| Llama-3.2-3B-Instruct | John persona | H∞ (ours) | **94.58 ± 1.76** | 0.83 ± 0.56 | **4.58 ± 1.31** | **95.42 ± 1.31** |
| Llama-3.2-3B-Instruct | DNE nonresponse | Original | 87.08 ± 1.91 | 0.83 ± 0.56 | 12.08 ± 1.70 | 87.92 ± 1.70 |
| Llama-3.2-3B-Instruct | DNE nonresponse | A-LQR | <u>87.92 ± 2.67</u> | <u>2.08 ± 1.12</u> | <u>10.00 ± 2.17</u> | <u>90.00 ± 2.17</u> |
| Llama-3.2-3B-Instruct | DNE nonresponse | H∞ (ours) | **93.75 ± 1.12** | 0.42 ± 0.42 | **5.83 ± 1.11** | **94.17 ± 1.11** |
| Llama-3.2-3B-Instruct | Jailbreak Bot | Original | 82.50 ± 1.50 | 2.50 ± 0.92 | 15.00 ± 1.27 | 85.00 ± 1.27 |
| Llama-3.2-3B-Instruct | Jailbreak Bot | A-LQR | <u>83.75 ± 1.45</u> | <u>4.17 ± 1.39</u> | <u>12.08 ± 1.91</u> | <u>87.92 ± 1.91</u> |
| Llama-3.2-3B-Instruct | Jailbreak Bot | H∞ (ours) | **94.58 ± 1.25** | 0.83 ± 0.56 | **4.58 ± 1.15** | **95.42 ± 1.15** |
| Llama-3.2-3B-Instruct | YOJA/Nona roleplay | Original | 85.42 ± 1.28 | <u>2.50 ± 1.11</u> | 12.08 ± 1.45 | 87.92 ± 1.45 |
| Llama-3.2-3B-Instruct | YOJA/Nona roleplay | A-LQR | <u>87.08 ± 1.58</u> | 2.08 ± 0.69 | <u>10.83 ± 1.55</u> | <u>89.17 ± 1.55</u> |
| Llama-3.2-3B-Instruct | YOJA/Nona roleplay | H∞ (ours) | **93.33 ± 1.78** | 1.25 ± 0.64 | **5.42 ± 1.65** | **94.58 ± 1.65** |
| Llama-3.2-3B-Instruct | APM programmer | Original | <u>88.33 ± 2.55</u> | 0.83 ± 0.83 | 10.83 ± 2.65 | 89.17 ± 2.65 |
| Llama-3.2-3B-Instruct | APM programmer | A-LQR | 87.08 ± 2.28 | <u>3.75 ± 1.31</u> | <u>9.17 ± 2.13</u> | <u>90.83 ± 2.13</u> |
| Llama-3.2-3B-Instruct | APM programmer | H∞ (ours) | **93.75 ± 1.55** | 0.42 ± 0.42 | **5.83 ± 1.67** | **94.17 ± 1.67** |
| Llama-3.1-8B-Instruct | Direct | Original | 78.33 ± 4.51 | 6.67 ± 3.47 | 15.00 ± 3.47 | 85.00 ± 3.47 |
| Llama-3.1-8B-Instruct | Direct | A-LQR | <u>80.00 ± 4.16</u> | <u>7.50 ± 2.62</u> | <u>12.50 ± 4.17</u> | <u>87.50 ± 4.17</u> |
| Llama-3.1-8B-Instruct | Direct | H∞ (ours) | **90.00 ± 2.72** | 3.33 ± 1.36 | **6.67 ± 2.08** | **93.33 ± 2.08** |
| Llama-3.1-8B-Instruct | John persona | Original | 67.50 ± 4.56 | 5.83 ± 2.17 | 26.67 ± 3.89 | 73.33 ± 3.89 |
| Llama-3.1-8B-Instruct | John persona | A-LQR | <u>75.00 ± 4.30</u> | <u>6.67 ± 2.08</u> | <u>18.33 ± 3.69</u> | <u>81.67 ± 3.69</u> |
| Llama-3.1-8B-Instruct | John persona | H∞ (ours) | **92.50 ± 2.62** | 0.83 ± 0.83 | **6.67 ± 2.08** | **93.33 ± 2.08** |
| Llama-3.1-8B-Instruct | DNE nonresponse | Original | 77.50 ± 6.58 | 1.67 ± 1.11 | 20.83 ± 5.86 | 79.17 ± 5.86 |
| Llama-3.1-8B-Instruct | DNE nonresponse | A-LQR | <u>81.67 ± 5.67</u> | <u>2.50 ± 1.27</u> | <u>15.83 ± 5.34</u> | <u>84.17 ± 5.34</u> |
| Llama-3.1-8B-Instruct | DNE nonresponse | H∞ (ours) | **96.67 ± 1.84** | 0.83 ± 0.83 | **2.50 ± 1.78** | **97.50 ± 1.78** |
| Llama-3.1-8B-Instruct | Jailbreak Bot | Original | <u>83.33 ± 3.51</u> | <u>4.17 ± 1.86</u> | <u>12.50 ± 2.56</u> | <u>87.50 ± 2.56</u> |
| Llama-3.1-8B-Instruct | Jailbreak Bot | A-LQR | 82.50 ± 4.38 | <u>4.17 ± 1.86</u> | 13.33 ± 3.33 | 86.67 ± 3.33 |
| Llama-3.1-8B-Instruct | Jailbreak Bot | H∞ (ours) | **93.33 ± 2.08** | 1.67 ± 1.11 | **5.00 ± 1.84** | **95.00 ± 1.84** |
| Llama-3.1-8B-Instruct | YOJA/Nona roleplay | Original | 78.33 ± 2.22 | 4.17 ± 1.86 | 17.50 ± 3.39 | 82.50 ± 3.39 |
| Llama-3.1-8B-Instruct | YOJA/Nona roleplay | A-LQR | <u>85.83 ± 2.50</u> | <u>5.00 ± 1.84</u> | <u>9.17 ± 1.94</u> | <u>90.83 ± 1.94</u> |
| Llama-3.1-8B-Instruct | YOJA/Nona roleplay | H∞ (ours) | **91.67 ± 3.51** | 1.67 ± 1.11 | **6.67 ± 3.47** | **93.33 ± 3.47** |
| Llama-3.1-8B-Instruct | APM programmer | Original | 74.17 ± 3.82 | 6.67 ± 1.67 | 19.17 ± 3.94 | 80.83 ± 3.94 |
| Llama-3.1-8B-Instruct | APM programmer | A-LQR | <u>76.67 ± 3.89</u> | <u>7.50 ± 1.94</u> | <u>15.83 ± 2.90</u> | <u>84.17 ± 2.90</u> |
| Llama-3.1-8B-Instruct | APM programmer | H∞ (ours) | **89.17 ± 2.50** | 1.67 ± 1.11 | **9.17 ± 2.31** | **90.83 ± 2.31** |

USR is FalseReject's toxic-prompt Useful Safety Rate: Direct Refusal + Safe Partial Compliance. Higher Safe Partial Compliance means more constructive safe engagement, although it can trade off against Direct Refusal; USR is the primary joint summary. Bold marks H∞ when it is best within a model and condition, and underline marks the best competitor. Every condition is reported separately. The 1B/3B rows contain 240 matched behaviors per condition; compact 8B rows contain 120. Values are full-sample percentages ± ten-group behavior-clustered jackknife standard errors. The judge uses the verbatim FalseReject Appendix G three-class rubric with pinned `gpt-4o-mini-2024-07-18` at temperature zero.
