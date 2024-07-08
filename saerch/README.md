# SAErch

Main idea is to use SAE features as controllable sliders in our paper search engine.

Steps we need to do:

1.  Embed each `astro-ph` abstract with an appropriate embedding model.

    i.  How do we chunk the abstracts? By sentence? Just the whole abstract? My initial thought is that the whole abstract would be more useful, as you are (after all) searching for a whole paper, and it would also reduce the number of embeddings we need to train an SAE over.

2.  Train an SAE model (probably a top-k SAE) on the embeddings.

    i.  What type of SAE architecture do we use? Vanilla SAE with L1 penalty? Or top-k?

    ii. What projection up do we use - 8, 16, 32, 64? My thought is to use a lower projection up so we have more general features, which seems most useful for our search UI.

3.  Use the trained SAE to store the feature activations from all abstracts. Since the SAE is trained to be sparse, we should be able to store this as a sparse matrix. (I don't think we need this, but I'm just putting it here for now.)

4.  For a bunch of features, we use automated interpretability to label that feature, along with a confidence score in our interpretation.

    i.  How many features do we label?

    ii. How do we sort which features are most important to label? Probably by the bimodality of their activation distribution (i.e. activate strongly or not at all) and also how many times they activate across our dataset (we can get all of this from the sparse SAE activations matrix).

5.  A user will send a query, and we want to construct the SAE hidden representation (i.e. the active features) and the decoded SAE reconstruction (from this hidden representation). That is, our reconstruction is just our query embedding. Then, as the user changes a slider, we boost/reduce that feature in the hidden representation, and use the decoder to reconstruct the embedding. Then, we just redo the vector-matrix multiply that represents the semantic search, which will reorder the top-k.