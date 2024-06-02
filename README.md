# Sequential FT-Transformer

This is an adaptation of the FT-Transformer model architecture for sequential data. The model has been reworked from Antons Ruberts Tensorflow FT-Transformer implementation. In addition to changing the model to support sequential data, the model has undergone numerous changes to enable support for Keras 3. Some design changes have also been made to allow for more flexibility. For example the old implementation required passing numpy arrays through the model to create the bins for the PLE numerical embedding layer. This has been moved out of the model, since it creates a bottlneck when performing distributed training using Horovod for example.

## Installation

```bash
$ pip install sequential_ft_transformer
```

## Usage

The notebooks folder shows multiple examples of the FT-Transformer being used in different situations. 

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`sequential_ft_transformer` was created by Christian Orr. It is licensed under the terms of the MIT license.

## References

- [FT-Transformer] - [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959)
- [Numerical Embeddings] - [OnEmbeddings for Numerical Features in Tabular Deep Learning](https://arxiv.org/abs/2203.05556)
- [Antons Ruberts Tensorflow FT-Transformer] - [TabTransformerTF](https://github.com/aruberts/TabTransformerTF)
