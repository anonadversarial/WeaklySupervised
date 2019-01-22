# Weakly Supervised

This repo contains code corresponding to our ICML 2019 submission
# Training a Model
To train a model run `python train_model.py --cifar_batch_size "cifar batch size" --gaussian_batch_size "gaussian batch size" --convex_batch_size "convex batch size" --mnist_batch_size "mnist batch size" --fashion_batch_size "fashion batch size"  --cifar_directory "cifar directory" --mnist_directory "mnist directory" --fashion_directory "fashion directory" --num_epochs "number of epochs" --output_directory "output`

# Evaluating Effort
To evaluate the effort of a model run `python test_model.py --model_names "which model you are testing" --num_iters "number of iterations to limit computation to" --mnist_directory "mnist directory"`

# Generating Adversarial Images
To generate adversarial images run `python generate_adversarial_images.py --model_name "name of model" --mnist_directory "mnist directory"--save_examples "whether to save the examples" --output_directory "where to save the examples if we choose to save them"`
