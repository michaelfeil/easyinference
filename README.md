# simpleinference
A stable and easy-to-use inference library with a focus on a sync-to-async API

```bash
pip install simpleinference
```

```python
from simpleinference import SimpleInference

# Run any model
register = SimpleInference(model_id=[
  # sentence-embeddings
  "michaelfeil/bge-small-en-v1.5",
  # sentence-embeddings and image-embeddings
  "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
  # classification models
  "philschmid/tiny-bert-sst2-distilled",
  # rerankers
  "mixedbread-ai/mxbai-rerank-xsmall-v1"
],
engine="torch"
)

sentences = ["Paris is in France.", "Berlin is in Germany.", "I love SF"]
images = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
question = "Where is Paris?"

register.embed(sentences=sentences, model_id="michaelfeil/bge-small-en-v1.5")
register.rerank(query=question, docs=sentences, model_id="mixedbread-ai/mxbai-rerank-xsmall-v1")
register.classify(model_id="philschmid/tiny-bert-sst2-distilled", sentences=sentences)
register.image_embed(model_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", images=images)
```

All functions return
```python
>>> embedding_fut = register.embed(sentences=sentences, model_id="michaelfeil/bge-small-en-v1.5")
>>> print(embedding_fut)
<Future at 0x7fa0e97e8a60 state=pending>
>>> time.sleep(1) and print(embedding_fut)
<Future at 0x7fa0e97e9c30 state=finished returned tuple>
>>> embedding_fut.result()
([array([-3.35943862e-03, ..., -3.22808176e-02], dtype=float32)], 19)
```
