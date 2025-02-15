<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>
  <meta itemprop="name" content="resisc45" />
  <meta itemprop="description" content="RESISC45 dataset is a publicly available benchmark for Remote Sensing Image&#10;Scene Classification (RESISC), created by Northwestern Polytechnical University&#10;(NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700&#10;images in each class.&#10;&#10;To use this dataset:&#10;&#10;```python&#10;import tensorflow_datasets as tfds&#10;&#10;ds = tfds.load(&#x27;resisc45&#x27;, split=&#x27;train&#x27;)&#10;for ex in ds.take(4):&#10;  print(ex)&#10;```&#10;&#10;See [the guide](https://www.tensorflow.org/datasets/overview) for more&#10;informations on [tensorflow_datasets](https://www.tensorflow.org/datasets).&#10;&#10;&lt;img src=&quot;https://storage.googleapis.com/tfds-data/visualization/fig/resisc45-3.0.0.png&quot; alt=&quot;Visualization&quot; width=&quot;500px&quot;&gt;&#10;&#10;" />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/resisc45" />
  <meta itemprop="sameAs" content="http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html" />
  <meta itemprop="citation" content="@article{Cheng_2017,&#10;   title={Remote Sensing Image Scene Classification: Benchmark and State of the Art},&#10;   volume={105},&#10;   ISSN={1558-2256},&#10;   url={http://dx.doi.org/10.1109/JPROC.2017.2675998},&#10;   DOI={10.1109/jproc.2017.2675998},&#10;   number={10},&#10;   journal={Proceedings of the IEEE},&#10;   publisher={Institute of Electrical and Electronics Engineers (IEEE)},&#10;   author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},&#10;   year={2017},&#10;   month={Oct},&#10;   pages={1865-1883}&#10;}" />
</div>

# `resisc45`


Warning: Manual download required. See instructions below.

*   **Description**:

RESISC45 dataset is a publicly available benchmark for Remote Sensing Image
Scene Classification (RESISC), created by Northwestern Polytechnical University
(NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700
images in each class.

*   **Additional Documentation**:
    <a class="button button-with-icon" href="https://paperswithcode.com/dataset/resisc45">
    Explore on Papers With Code
    <span class="material-icons icon-after" aria-hidden="true"> north_east
    </span> </a>

*   **Homepage**:
    [http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)

*   **Source code**:
    [`tfds.image_classification.Resisc45`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/image_classification/resisc45.py)

*   **Versions**:

    *   **`3.0.0`** (default): No release notes.

*   **Download size**: `Unknown size`

*   **Dataset size**: `407.97 MiB`

*   **Manual download instructions**: This dataset requires you to
    download the source data manually into `download_config.manual_dir`
    (defaults to `~/tensorflow_datasets/downloads/manual/`):<br/>
    Dataset can be downloaded from OneDrive:
    https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
    After downloading the rar file, please extract it to the manual_dir.

*   **Auto-cached**
    ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)):
    No

*   **Splits**:

Split     | Examples
:-------- | -------:
`'train'` | 31,500

*   **Feature structure**:

```python
FeaturesDict({
    'filename': Text(shape=(), dtype=string),
    'image': Image(shape=(256, 256, 3), dtype=uint8),
    'label': ClassLabel(shape=(), dtype=int64, num_classes=45),
})
```

*   **Feature documentation**:

Feature  | Class        | Shape         | Dtype  | Description
:------- | :----------- | :------------ | :----- | :----------
         | FeaturesDict |               |        |
filename | Text         |               | string |
image    | Image        | (256, 256, 3) | uint8  |
label    | ClassLabel   |               | int64  |

*   **Supervised keys** (See
    [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)):
    `('image', 'label')`

*   **Figure**
    ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)):

<img src="https://storage.googleapis.com/tfds-data/visualization/fig/resisc45-3.0.0.png" alt="Visualization" width="500px">

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):

<!-- mdformat off(HTML should not be auto-formatted) -->

{% framebox %}

<button id="displaydataframe">Display examples...</button>
<div id="dataframecontent" style="overflow-x:auto"></div>
<script>
const url = "https://storage.googleapis.com/tfds-data/visualization/dataframe/resisc45-3.0.0.html";
const dataButton = document.getElementById('displaydataframe');
dataButton.addEventListener('click', async () => {
  // Disable the button after clicking (dataframe loaded only once).
  dataButton.disabled = true;

  const contentPane = document.getElementById('dataframecontent');
  try {
    const response = await fetch(url);
    // Error response codes don't throw an error, so force an error to show
    // the error message.
    if (!response.ok) throw Error(response.statusText);

    const data = await response.text();
    contentPane.innerHTML = data;
  } catch (e) {
    contentPane.innerHTML =
        'Error loading examples. If the error persist, please open '
        + 'a new issue.';
  }
});
</script>

{% endframebox %}

<!-- mdformat on -->

*   **Citation**:

```
@article{Cheng_2017,
   title={Remote Sensing Image Scene Classification: Benchmark and State of the Art},
   volume={105},
   ISSN={1558-2256},
   url={http://dx.doi.org/10.1109/JPROC.2017.2675998},
   DOI={10.1109/jproc.2017.2675998},
   number={10},
   journal={Proceedings of the IEEE},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
   year={2017},
   month={Oct},
   pages={1865-1883}
}
```

