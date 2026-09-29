# Speech Emotion Recognition Dashboard

A React/Vite dashboard for the SE4050 Speech Emotion Recognition project. It presents the six evaluated deep-learning models, their test performance, architecture summaries, confusion matrices, and an optional live inference workflow.

## Evaluated models

The dashboard compares the following six model architectures:

| Model | Main architecture | Primary contributor |
| --- | --- | --- |
| 1 | **BiLSTM + Attention**<br>BiLSTM -> BiLSTM -> Attention -> Dense -> Softmax | Gayanatha D.S |
| 2 | **LSTM + Attention Pooling**<br>LSTM -> LSTM -> Attention Pooling -> Dense -> Softmax | Rajapaksha L.A.T.C |
| 3 | **CNN-BiGRU + Dual Attention**<br>CNN -> BiGRU -> Attention -> Residual + LayerNorm -> GAP -> Dense | Additional Model |
| 4 | **CNN-SE-BiLSTM + Multi-Head Attention**<br>CNN -> SE -> BiLSTM -> MHA -> Dense -> Softmax | Additional Model |
| 5 | **CNN-BiLSTM + Multi-Head Attention + SpecAugment**<br>CNN -> Projection -> BiLSTM -> 8-Head MHA -> FFN -> GAP -> Dense | Sandani A.W.A |
| 6 | **MFT-TCN + Attention**<br>MFT -> 5 TCN Blocks -> MHA -> FFN -> Hybrid Pooling -> Dense | Sarangi K.P.E |

### Dashboard features

- Six evaluated model cards with architecture and input summaries
- Test accuracy comparison
- Best-model confusion matrix
- Model inspector with accuracy, macro F1, validation accuracy, and train-validation gap
- Interactive WAV upload UI
- Optional FastAPI inference backend
- Responsive interface suitable for project demonstrations and viva presentations

## Important reproducibility note

The uploaded project archive contains the evaluation results for the six models, but it does not contain the six trained `.keras` model artifacts or the raw CREMA-D audio files. Therefore the dashboard uses the verified experiment results for comparison and provides the live inference connection as an optional backend.

The current live API is designed for Model 5:
`CNN-BiLSTM with Multi-Head Attention and SpecAugment`.

To enable actual live inference, place the trained model at:

`models/cnn_bilstm_mha_specaug/best_model.keras`

and provide the same Log-Mel preprocessing/training normalization assets used during training.

Do not describe the dashboard's probability preview as a real prediction until the inference API is successfully connected.

## Run frontend

```bash
cd dashboard
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Run backend

From the project root:

```bash
pip install fastapi uvicorn python-multipart
uvicorn dashboard_api.app:app --reload --port 8000
```

Then select Model 5 in the dashboard and upload a WAV file.

## Viva demonstration flow

1. Show the overview and dataset information.
2. Show all six model cards.
3. Click each model and explain its architecture.
4. Show the accuracy comparison.
5. Show the best-model confusion matrix.
6. Explain the train-validation gap using the Model Inspector.
7. Show the prediction UI and, if the backend/model artifact is available, run a WAV prediction.
8. Explain that the test set was kept unseen during model development.

The assignment requires multiple metrics and comparison of performance, generalization, computational efficiency, complexity and practical limitations, so this dashboard is intentionally organized around those items.
