export const classes = [
  "Anger",
  "Disgust",
  "Fear",
  "Happy",
  "Neutral",
  "Sad"
];

export const models = [
  {
    "id": "bilstm-attention",
    "short": "BiLSTM + Attention",
    "name": "BiLSTM with Attention",
    "description": "Bidirectional temporal modelling with attention pooling.",
    "input": "Log-Mel + \u0394 + \u0394\u0394 (192-D)",
    "accuracy": 58.5,
    "precision": 59.7,
    "recall": 58.56,
    "f1": 58.3,
    "train": 93.98,
    "val": 63.88,
    "gap": 30.1,
    "bestEpoch": 28,
    "params": 510022,
    "status": "Evaluation results available",
    "strength": "Strong bidirectional temporal modelling"
  },
  {
    "id": "lstm-attention",
    "short": "LSTM + Attention",
    "name": "LSTM with Attention Pooling",
    "description": "Forward recurrent modelling with attention-based aggregation.",
    "input": "Log-Mel + \u0394 + \u0394\u0394 (192-D)",
    "accuracy": 57.04,
    "precision": 58.61,
    "recall": 57.17,
    "f1": 56.84,
    "train": 80.9,
    "val": 61.07,
    "gap": 19.83,
    "bestEpoch": 18,
    "params": 222534,
    "status": "Evaluation results available",
    "strength": "Compact recurrent baseline"
  },
  {
    "id": "cnn-bigru-dual",
    "short": "CNN-BiGRU + Dual Attention",
    "name": "CNN-BiGRU with Dual Attention",
    "description": "CNN local feature extraction + bidirectional GRU + dual attention.",
    "input": "Log-Mel (64 \u00d7 174)",
    "accuracy": 58.99,
    "precision": 60.51,
    "recall": 59.18,
    "f1": 58.96,
    "train": 66.02,
    "val": 62.85,
    "gap": 3.17,
    "bestEpoch": 34,
    "params": 1016103,
    "status": "Evaluation results available",
    "strength": "Strong performance/generalization balance"
  },
  {
    "id": "cnn-se-bilstm-mha",
    "short": "CNN-SE-BiLSTM + MHA",
    "name": "CNN-SE-BiLSTM with Multi-Head Attention",
    "description": "CNN + SE channel attention + BiLSTM + Multi-Head Attention.",
    "input": "Log-Mel (64 \u00d7 174)",
    "accuracy": 59.15,
    "precision": 61.97,
    "recall": 59.18,
    "f1": 59.31,
    "train": 67.53,
    "val": 61.44,
    "gap": 6.09,
    "bestEpoch": 20,
    "params": 1186054,
    "status": "Evaluation results available",
    "strength": "Combines local, channel and temporal attention"
  },
  {
    "id": "cnn-bilstm-mha-specaug",
    "short": "CNN-BiLSTM + MHA + SpecAug",
    "name": "CNN-BiLSTM with Multi-Head Attention and SpecAugment",
    "description": "CNN feature extraction + BiLSTM + MHA + training-time SpecAugment.",
    "input": "Log-Mel (64 \u00d7 174)",
    "accuracy": 59.89,
    "precision": 63.02,
    "recall": 59.77,
    "f1": 59.73,
    "train": 76.61,
    "val": 62.76,
    "gap": 13.85,
    "bestEpoch": 31,
    "params": 1180166,
    "status": "Best observed test result",
    "strength": "Best observed test accuracy and macro F1"
  },
  {
    "id": "mft-tcn-attention",
    "short": "MFT-TCN + Attention",
    "name": "MFT-TCN with Attention",
    "description": "Multi-feature temporal representation + dilated TCN + Multi-Head Attention.",
    "input": "201-D frame-level MFT representation",
    "accuracy": 54.03,
    "precision": 55.66,
    "recall": 54.03,
    "f1": 53.51,
    "train": 67.01,
    "val": 56.57,
    "gap": 10.44,
    "bestEpoch": 39,
    "params": 420000,
    "status": "Evaluation results available",
    "strength": "Multi-scale temporal convolution"
  }
];

export const bestConfusionMatrix = [
  [
    171,
    27,
    3,
    7,
    2,
    0
  ],
  [
    37,
    137,
    23,
    4,
    3,
    6
  ],
  [
    16,
    20,
    130,
    14,
    9,
    21
  ],
  [
    49,
    22,
    31,
    104,
    3,
    1
  ],
  [
    18,
    33,
    6,
    13,
    99,
    10
  ],
  [
    14,
    37,
    44,
    3,
    17,
    95
  ]
];
