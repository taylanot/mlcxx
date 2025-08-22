# examples 

Here lies the some examples for the usage of this project.

---
## openml-report.cpp

This example demonstrates how to use the tool for generating reports on datasets from the OpenML database. Running the executable with the command `./openml-report openml-id` will automatically download the specified dataset and display a report in the terminal.

---

## simple.cpp

This example shows how to generate learning curves. The results are saved for both a tuned and an untuned version of the model. Using your preferred plotting tool, you can obtain figures such as:

![Learning curve for a fixed Decision Tree model on the Banana Dataset with training set sizes in the range $[10,350]$.][./figures/nottuned_plot.png]

![Learning curve for a hyper-parameter tuned Decision Tree model on the Banana Dataset with training set sizes in the range $[10,350]$. Tuning is done with a simple train-validation split, and the only hyper-parameter varied is the minimum leaf size.][./figures/tuned_plot.png]

Since both figures are based on experiments repeated 1000 times, additional statistical measures of generalization performance are also computed with high confidence.

To illustrate the scale of the experiments even for this simple problem, see the figure below:

![Each scatter point represents a classification where 50 Decision Tree classifiers with different minimum leaf sizes are trained on a subset of the training data. The best model is selected using a validation set, then retrained on the combined training and validation data, and finally tested on a separate test set.][./figures/nottuned.png]

---

## ann_reg.cpp / ann_class.cpp
These examples demonstrate how to use the wrapper around the mlpack::ann module. You can define and train your networks as shown in the examples, which are designed to be self-explanatory.

If you want to save trained models, remember to add the flag `define MLPACK_ENABLE_ANN_SERIALIZATION`. Note that enabling serialization may slightly slow down compilation.
