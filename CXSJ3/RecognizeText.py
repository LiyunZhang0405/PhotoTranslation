import torch
from torch.autograd import Variable
from PIL import Image
import recognization.dataset
from recognization import crnn, dataset, utils

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

def load_model(path):
    model = crnn.CRNN(32, 1, 37, 256)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

def recognize(model, image_path, num):
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))

    # data = dataset.lmdbDataset(root=image_path, transform=transformer)
    # # data_loader = torch.utils.data.DataLoader(
    # #     data, shuffle=True, batch_size=num, num_workers=int(2))
    # preds = model(data)
    # ans = ''
    # for pred in preds:
    #     _, pred = preds.max(2)
    #     pred = pred.transpose(1, 0).contiguous().view(-1)
    #     preds_size = Variable(torch.IntTensor([pred.size(0)]))
    #     sim_pred = converter.decode(pred.data, preds_size.data, raw=False)
    #     ans += ' ' + sim_pred
    # return ans

    image = Image.open(image_path).convert('L')
    image = transformer(image)
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred
