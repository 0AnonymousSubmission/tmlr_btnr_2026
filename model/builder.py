import random
from typing import Any

import quimb.tensor as qt


class Inputs:
    """
    Flexible Input loader.
    
    input_labels can be:
    1. Tuple/String: ("p", "x") -> Auto-maps to input[i] (or input[0] if single input).
    2. List (Explicit): [source_idx, ("p", "x")] -> Uses inputs[source_idx].
    """
    def __init__(self, inputs: list[Any], 
                 outputs: list[Any], 
                 outputs_labels: list[str],
                 input_labels: list[str | tuple[str, ...] | list[Any]], 
                 batch_dim: str = "s",
                 batch_size = None):

        # 1. Configuration
        self.inputs_data = inputs
        self.outputs_data = outputs
        self.outputs_labels = outputs_labels
        self.input_labels = input_labels
        self.batch_dim = batch_dim
        
        self.batch_size = inputs[0].shape[0] if batch_size is None else batch_size
        self.samples = outputs[0].shape[0]
        
        # 2. Pre-compute batches
        self.batches = self._create_batches()

    def _create_batches(self) -> list[tuple[list[qt.Tensor], qt.Tensor]]:
        batches = []
        
        # Generator for raw data slices
        # This yields a dict {0: batch_A, 1: batch_B} containing ALL inputs
        raw_splits = self.batch_splits(
            self.inputs_data, 
            self.outputs_data[0], 
            self.batch_size
        )

        for input_dict, y_tensor in raw_splits:
            # Construct the TN inputs based on definitions
            mu = self._prepare_batch(input_dict)
            batches.append((mu, y_tensor))
            
        return batches

    # --- Properties ---

    @property
    def data_mu(self):
        for mu, _ in self.batches:
            yield mu

    @property
    def data_y(self):
        for _, y in self.batches:
            yield y

    @property
    def data_mu_y(self):
        for mu, y in self.batches:
            yield mu, y

    @property
    def data_sigma(self):
        for mu, _ in self.batches:
            prime_tensors = []
            for t in mu:
                prime_inds = tuple(
                    f"{ind}_prime" if ind != self.batch_dim else ind 
                    for ind in t.inds
                )
                prime_tags = {f"{tag}_prime" for tag in t.tags}
                prime_tensor = qt.Tensor(data=t.data, inds=prime_inds, tags=prime_tags)
                prime_tensors.append(prime_tensor)
            yield mu + prime_tensors

    # --- Core Logic ---

    def batch_splits(self, xs, y, B):
        """Generates slices for ALL input tensors provided."""
        s = y.shape[0]
        for i in range(0, s, B):
            # Target Tensor
            tensor = qt.Tensor(
                data=y[i:i+B],
                inds=(self.batch_dim, *self.outputs_labels),
                tags={'output'}
            )
            
            # Input Dict: Map index -> Data Slice
            # We slice ALL inputs, regardless of how many labels consume them
            batch = {idx: x[i:i+B] for idx, x in enumerate(xs)}
            
            yield batch, tensor

    def _prepare_batch(self, input_data: dict[int, Any]) -> list[qt.Tensor]:
        """
        Constructs the list of QT tensors for the network based on input_labels definitions.
        """
        tensors = []
        
        for i, definition in enumerate(self.input_labels):
            
            # 1. Parse Definition
            if isinstance(definition, list):
                # Explicit: [source_idx, (inds...)]
                source_idx = definition[0]
                inds_def = definition[1]
            else:
                # Implicit: (inds...) or "ind"
                # If only 1 input exists, use 0. Else assume 1-to-1 mapping (i -> i)
                source_idx = 0 if len(self.inputs_data) == 1 else i
                inds_def = definition
            
            # 2. Parse Indices
            if isinstance(inds_def, str):
                inds = (self.batch_dim, inds_def)
                tag_suffix = inds_def
            else:
                # Tuple
                inds = (self.batch_dim, *inds_def)
                tag_suffix = "_".join(inds_def)

            # 3. Fetch Data
            if source_idx not in input_data:
                raise ValueError(f"Label definition {i} requests input index {source_idx}, but only {len(input_data)} inputs provided.")
            
            data = input_data[source_idx]
            
            # 4. Create Tensor
            # Tag convention: input_{indices} plus I{i} for site location
            tags = {f'input_{tag_suffix}', f'I{i}'}
            tensor = qt.Tensor(data=data, inds=inds, tags=tags)
            tensors.append(tensor)
        
        return tensors

    def shuffle(self):
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def __str__(self):
        if not self.batches:
            return ">>> InputLoader (Empty)"

        mu, y = self.batches[0]
        mu_inds = [list(t.inds) for t in mu]
        
        header = (
            f"\n>>> InputLoader Summary (Batch Size: {self.batch_size}, "
            f"Samples: {self.samples}, Batches: {len(self.batches)})\n"
            f"{'TYPE':<8} | {'SHAPE':<15} | {'INDICES'}\n"
            f"{'-'*60}\n"
        )
        
        row_y = f"{'Target':<8} | {y.shape!s:<15} | {y.inds}\n"
        row_mu = f"{'Mu':<8} | {mu[0].shape!s:<15} | {mu_inds}\n"
        
        return header + row_y + row_mu
