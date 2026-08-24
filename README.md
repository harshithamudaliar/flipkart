# Flipkart Catalog Intelligence System

This project integrates three components,connected system combining a return-risk model, a product-image classifier, and a LangGraph support agent.

## Project Structure
- `src/` — scripts for data generation, training, and the agent
- `data/` — generated datasets (not tracked in git; regenerate via scripts below)
- `models/` — trained model artifacts (not tracked in git; regenerate via scripts below)

## Part 1: Return-Risk Model
*(instructions)*
Run 01_preprocessing.ipynb first to generate the split/transformed data files in data/ (not tracked in git) before running any of the model notebooks.
#RESULTS
After performing a threshold sweep on the Random Forest classifier:

Best Threshold (t\_rf):* 0.42

Best F1 Score: 0.398

Best Precision: 0.265

Best Recall: 0.799

This threshold balances recall (catching more true returns) with precision (reducing false alarms). The relatively high recall indicates the model is effective at identifying risky orders, though precision remains modest — reflecting the challenge of minimizing false positives in return-risk prediction.

## Part 2: Product-Image Classifier
*(instructions)*
-Run part2-image-classifier's train_image_classifier.py to auto download FashionMNIST/ and load data
#Dataset
-  Fashion-MNIST (Zalando Research)-48,000 train / 12,000 validation (stratified 20% split of the 60,000 training images) / 10,000 test 
-# Model
- **Backbone:** EfficientNet-B0 (`IMAGENET1K_V1` weights), all layers frozen
- **Head:** `Linear(1280→128) → ReLU → Dropout(0.3) → Linear(128→10)`
- **Training:** backbone features cached once, head trained on cached
  features for 10 epochs, batch size 64, Adam optimizer, lr=1e-3
# Results
- Validation accuracy : 92.63%
- Feature extraction alone exceeded the 80% threshold, so fine-tuning of   the backbone's late layers was not required.
- Final test accuracy: 91.91%

# Confusion Matrix & Analysis
Full 10×10 confusion matrix and per-class precision/recall are in
`notebooks/05_classify_product_image`.

## Part 3: LangGraph Support Agent
*(instructions )*
Run 06_flipkart_support agent for 

#Results
- Precision@3 is around 0.33 when only one document exists for a query, since most query have one relevant document the precision score is around 0.33 for most of them.
- Recall@3 is around 0.90 meaning retrieval finds relevant documnets within top 3, making it a more meaningful metric

#TRANSCRIPTS
Transcripts are saved in the transcripts/ folder. Each file contains plain text logs of user–agent exchanges.

## Example Transcript
*Turn 1 User: What's the return risk for order ORD1042?
Turn 1 Agent: Predicted return risk: None (probability None).

Turn 2 User: Remind me, what did you say about that order?
Turn 2 Agent: Predicted return risk: None (probability None).last_order_id=ORD1042
*
