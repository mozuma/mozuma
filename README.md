# MoZuMa

MoZuMa is a model zoo for multimedia search application. It provides an easy to use interface to run models for:

- **Text to image retrieval**: Rank images by their similarity to a text query.
- **Image similarity search**: Rank images by their similarity to query image.
- **Image classification**: Add labels to images.
- **Face detection**: Detect and retrieve images with similar faces.
- **Object detection**: Detect and retrieve images with similar objects.
- **Video keyframes extraction**: Retrieve the important frames of a video.
  Key-frames are used to apply all the other queries on videos.
- **Multilingual text search**: Rank similar sentences from a text query in multiple languages.

## Quick links

- [Documentation](https://mozuma.github.io/mozuma/)
- [Models](https://mozuma.github.io/mozuma/models/)
- [For developers](https://mozuma.github.io/mozuma/contributing/0-setup.md)

## Example gallery

See `docs/examples/` for a collection of ready to use notebooks.

## Citation

Please cite as:

```bibtex
@inproceedings{mozuma,
  author = {Massonnet, St\'{e}phane and Romanelli, Marco and Lebret, R\'{e}mi and Poulsen, Niels and Aberer, Karl},
  title = {MoZuMa: A Model Zoo for Multimedia Applications},
  year = {2022},
  isbn = {9781450392037},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3503161.3548542},
  doi = {10.1145/3503161.3548542},
  abstract = {Lots of machine learning models with applications in Multimedia Search are released as Open Source Software. However, integrating these models into an application is not always an easy task due to the lack of a consistent interface to run, train or distribute models. With MoZuMa, we aim at reducing this effort by providing a model zoo for image similarity, text-to-image retrieval, face recognition, object similarity search, video key-frames detection and multilingual text search implemented in a generic interface with a modular architecture. The code is released as Open Source Software at https://github.com/mozuma/mozuma.},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {7335–7338},
  numpages = {4},
  keywords = {multimedia search, vision and language, open source software},
  location = {Lisboa, Portugal},
  series = {MM '22}
}
```
