# Evaluation Data

This folder contains the result that was put on for evaluation in our paper.

- `input` contains all the model response and the source conversation from *ALL OTHER MODELS*.
- `mrag` contains all the model response and the source conversation from *MEDRAG SYSTEM*. The ones with `no_rag` were ran with RAG option disabled.
- `pkl` contains all the model response and the meta data (e.g. retrieval sequences) from *OUR MODEL*.

We have already put an example file under each folder in `evaluation`. If you'd like to run all the tests, please

- put all the `json` files from `input` to `evaluation/input`
- put all the `json` files from `mrag` to `evaluation/mrag`
- put all the `json` files from `pkl` to `evaluation/pkl`

Then run the evaluation script based on the readme in the root directory.

Please note that due to certain randomness in the LLM, and we are using LLM as the judge in some metrics, the results may vary slightly. However, the overall trend is consistent.
