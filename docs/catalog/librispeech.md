<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>
  <meta itemprop="name" content="librispeech" />
  <meta itemprop="description" content="LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,&#10;prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read&#10;audiobooks from the LibriVox project, and has been carefully segmented and aligned.&#10;&#10;It&#x27;s recommended to use lazy audio decoding for faster reading and smaller dataset size:&#10;- install `tensorflow_io` library: `pip install tensorflow-io`&#10;- enable lazy decoding: `tfds.load(&#x27;librispeech&#x27;, builder_kwargs={&#x27;config&#x27;: &#x27;lazy_decode&#x27;})`&#10;&#10;To use this dataset:&#10;&#10;```python&#10;import tensorflow_datasets as tfds&#10;&#10;ds = tfds.load(&#x27;librispeech&#x27;, split=&#x27;train&#x27;)&#10;for ex in ds.take(4):&#10;  print(ex)&#10;```&#10;&#10;See [the guide](https://www.tensorflow.org/datasets/overview) for more&#10;informations on [tensorflow_datasets](https://www.tensorflow.org/datasets).&#10;&#10;" />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/librispeech" />
  <meta itemprop="sameAs" content="http://www.openslr.org/12" />
  <meta itemprop="citation" content="@inproceedings{panayotov2015librispeech,&#10;  title={Librispeech: an ASR corpus based on public domain audio books},&#10;  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},&#10;  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},&#10;  pages={5206--5210},&#10;  year={2015},&#10;  organization={IEEE}&#10;}" />
</div>

# `librispeech`


Note: This dataset has been updated since the last stable release. The new
versions and config marked with
<span class="material-icons" title="Available only in the tfds-nightly package">nights_stay</span>
are only available in the `tfds-nightly` package.

*   **Description**:

LibriSpeech is a corpus of approximately 1000 hours of read English speech with
sampling rate of 16 kHz, prepared by Vassil Panayotov with the assistance of
Daniel Povey. The data is derived from read audiobooks from the LibriVox
project, and has been carefully segmented and aligned.

It's recommended to use lazy audio decoding for faster reading and smaller
dataset size: - install `tensorflow_io` library: `pip install tensorflow-io` -
enable lazy decoding: `tfds.load('librispeech', builder_kwargs={'config':
'lazy_decode'})`

*   **Additional Documentation**:
    <a class="button button-with-icon" href="https://paperswithcode.com/dataset/librispeech">
    Explore on Papers With Code
    <span class="material-icons icon-after" aria-hidden="true"> north_east
    </span> </a>

*   **Homepage**: [http://www.openslr.org/12](http://www.openslr.org/12)

*   **Source code**:
    [`tfds.audio.Librispeech`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/audio/librispeech.py)

*   **Download size**: `Unknown size`

*   **Dataset size**: `Unknown size`

*   **Auto-cached**
    ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)):
    Unknown

*   **Splits**:

Split | Examples
:---- | -------:

*   **Feature structure**:

```python
FeaturesDict({
    'chapter_id': int64,
    'id': string,
    'speaker_id': int64,
    'speech': Audio(shape=(None,), dtype=int16),
    'text': Text(shape=(), dtype=string),
})
```

*   **Feature documentation**:

Feature    | Class        | Shape   | Dtype  | Description
:--------- | :----------- | :------ | :----- | :----------
           | FeaturesDict |         |        |
chapter_id | Tensor       |         | int64  |
id         | Tensor       |         | string |
speaker_id | Tensor       |         | int64  |
speech     | Audio        | (None,) | int16  |
text       | Text         |         | string |

*   **Supervised keys** (See
    [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)):
    `('speech', 'text')`

*   **Figure**
    ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)):
    Not supported.

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):
    Missing.

*   **Citation**:

```
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
```


## librispeech/default (default config) <span class="material-icons" title="Available only in the tfds-nightly package">nights_stay</span>

*   **Config description**: Default dataset.

*   **Versions**:

    *   **`2.1.1`** (default): Fix speech data type with dtype=tf.int16.
    *   `2.1.2`: Add 'lazy_decode' config.

## librispeech/lazy_decode <span class="material-icons" title="Available only in the tfds-nightly package">nights_stay</span>

*   **Config description**: Raw audio dataset.

*   **Versions**:

    *   `2.1.1`: Fix speech data type with dtype=tf.int16.
    *   **`2.1.2`** (default): Add 'lazy_decode' config.
