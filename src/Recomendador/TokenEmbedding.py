import torch
class TokenEmbedding:
    def __init__(self, file_name, n):
      self.idx_to_token, self.idx_to_vec, self.dim = self._load_embedding(
        file_name, n)
      self.unknown_idx = 0
      self.token_to_idx = {token: idx for idx, token in
                          enumerate(self.idx_to_token)}
  

    def _load_embedding(self, file_name, n):
      idx_to_token, idx_to_vec = ['<unk>'], []
      with open( file_name, 'r') as f:
        first_read = True
        i=0
        for line in f:
          if n<i: break
          else: i+=1
          if first_read:
            first_read = False
            continue
          elems = line.rstrip().split(' ')
          token, elems = elems[0], [float(elem) for elem in elems[1:]]
          # Skip header information, such as the top row in fastText
          if len(elems) > 1:
              idx_to_token.append(token)
              idx_to_vec.append(elems)
      idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
      return idx_to_token, torch.tensor(idx_to_vec), len(idx_to_vec[0])
    def __getitem__(self, tokens):
      indices = [self.token_to_idx.get(token, self.unknown_idx)
                for token in tokens]
      vecs = self.idx_to_vec[torch.tensor(indices)]
      return vecs

    def __len__(self):
      return len(self.idx_to_token)