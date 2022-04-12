# two_link_control
Various nonlinear adaptive controllers leading up to doing deep adaptive control using a two-link arm for simulation. Different branches have various strategies leading up to a deep adaptive control strategy. Most will also have a pdf explaining the math behind the code.

I chose a two-link because it is relatively easy to picture and the desired trajecteory is basically picking up a drink.

Hope you enjoy and can adapt this to your own problems!

main: has a nonlinear gradient adaptive controller
concurrent_learning: adds concurrent learning to the gradient controller
integral_concurrent_learning: adds integral concurrent learning to the gradient controller
single_layer_nn: adds a single layer network to the gradient controller to help estimate the dynamic disturbance 
two_link_two_layer: adds a two layer network to the gradient controller to help estimate the dynamic disturbance 
multi_layer_torch: adds a deep network to the gradient controller to help estimate the dynamic disturbance 
