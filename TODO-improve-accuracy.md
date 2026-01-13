# TODO: Improve Emotion Recognition Accuracy

## Current Issues
- Final accuracy is 0.20 (below random 0.166 for 6 classes)
- Tuning accuracy was 0.3525 but dropped significantly
- Warnings about undefined metrics indicate some classes have no predictions

## Proposed Improvements
- [ ] Increase CNN learning rate from 1e-4 to 1e-3 for faster convergence
- [ ] Increase max augmented samples from 5500 to 20000 to leverage more data
- [ ] Add more diverse augmentations (volume changes, speed perturbations)
- [ ] Enhance CNN architecture with more convolutional blocks and filters
- [ ] Implement k-fold cross-validation instead of single train/test split
- [ ] Add additional features (chroma, spectral contrast) alongside MFCC
- [ ] Experiment with removing SVM and using CNN directly for classification
- [ ] Add validation monitoring to prevent overfitting (plot training vs validation loss)

## Implementation Plan
1. Modify hyperparameters (learning rate, max samples)
2. Enhance augmentation functions
3. Update CNN architecture
4. Implement k-fold CV
5. Add feature extraction functions
6. Test CNN-only vs CNN+SVM
7. Add training visualization
