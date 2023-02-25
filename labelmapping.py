from numbers import Number

class LabelMapper:
    def __init__(self, label_mapping=None, update=True):
        if label_mapping is None:
            self._label_mapping = {}
            self._inverse_label_mapping = {}
            self._highest_label = -1
        else:
            self._label_mapping = label_mapping
            self._inverse_label_mapping = {
                k: v for v, k in label_mapping.items()
            }
            self._highest_label = max([v for v in label_mapping.values()])
        self._should_update = update

    def _update(self, unique_labels):
        for label in unique_labels:
            if not label in self._label_mapping:
                if self._should_update:
                    self._label_mapping[label] = self._highest_label + 1
                    self._inverse_label_mapping[self._highest_label + 1] = label
                    self._highest_label += 1
                else:
                    self._label_mapping[label] = None

    def __call__(self, labels):
        if isinstance(labels, Number):
            self._update([labels])
            return self._label_mapping[labels]
        
        unique_labels = torch.unique(labels)
        unique_labels = [int(x) for x in unique_labels]
        self._update(unique_labels)

        res = labels[:]
        for k, v in self._label_mapping.items():
            res[labels == k] = v
        return res

    def inverse_map(self, labels):
        if isinstance(labels, Number):
            return self._inverse_label_mapping[labels]

        res = labels[:]
        for k, v in self._inverse_label_mapping.items():
            res[labels == k] = v
        return res

    def map_range(self, end):
        mapped_range = []
        placeholder_index = self._highest_label + 1
        for i in range(end):
            if i in self._label_mapping:
                mapped_range.append(self._label_mapping[i])
            else:
                mapped_range.append(placeholder_index)
                placeholder_index += 1
        return mapped_range
