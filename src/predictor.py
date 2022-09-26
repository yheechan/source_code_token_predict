import torch
import torch.nn.functional as F

def predict(prefix, postfix, model):

    prefix = torch.tensor(prefix).unsqueeze(dim=0)
    postfix = torch.tensor(postfix).unsqueeze(dim=0)

    model.cpu()

    logits = model.forward(prefix, postfix)
    probs = F.softmax(logits, dim=1).squeeze(dim=0)

    result = probs*100

    sorted, indices = torch.sort(result, descending=True)

    for i in range(10):
        choice = '#' + str(i+1) + ') ' + str(indices[i].item()) + ': ' + str(sorted[i].item()) + '%'
        print(choice)