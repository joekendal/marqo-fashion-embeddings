# Marqo FashionSigLIP Model

## Overview

Replicate URL: [https://replicate.com/joekendal/marqo-fashion-embeddings](https://replicate.com/joekendal/marqo-fashion-embeddings)

The Marqo FashionSigLIP model is designed to enhance fashion-related search and recommendation systems.

It outperforms OpenAI's CLIP and FashionCLIP across relevant benchmarks for open fashion datasets. For more info visit their [announcement](https://www.marqo.ai/blog/search-model-for-fashion), [the model card](https://huggingface.co/Marqo/marqo-fashionSigLIP) and/or GitHub [repo](https://github.com/marqo-ai/marqo-FashionCLIP)

This replicate API takes an input of an image + text for the fashion product and returns the embeddings of each + a concatenated embedding of both which can be used for multi-modal search instead.

For example, you may have a product JSON in your database with multiple fields. In this case you can vectorize the values by sorting the keys in alphabetical order, concatenating the string and string[] fields with commas (excluding irrelevant info such as the id) and then lowercase the combined string.

You would use this with the image embedding to perform semantic queries in your vector store.

They have licensed this model under Apache 2.0 so you are free to use it commercially according to the terms of the license.