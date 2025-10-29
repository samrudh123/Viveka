Idea from paper - [LLMs know more than they show](https://arxiv.org/pdf/2410.02707)

Experimental Setup: 

1. Check repository for reproducible code, and figure out the details (choice of hyperparameters, optimizers etc.) of the techniques/functions that they used. 
2. Implement the code in the Viveka repository, and make sure it is working like it is supposed to (label everything, keep the code clean) --> it should be extracting activations at desired layer at desired token. 
3. Load a model and a dataset (start with small and then scale up)
4. Train probes on these extracted activations: 
	- [ ] Try different methods of training linear probes other than Logistic Regression (neural network, LDA etc.)
	- [ ] Primary focus : activation patterns
5. Once we have established some sort of results, we can try to scale it across tasks and datasets, and improve on existing methods. 
