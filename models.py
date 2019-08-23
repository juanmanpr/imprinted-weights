import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from googlenet_master import create_model as create_gn
        
class Net(nn.Module):
    def __init__(self, num_classes=100, norm=True, scale=True, extractor_type='default'):
        super(Net,self).__init__()
        
        if extractor_type == 'googlenet':
            self.extractor = GN_Extractor()    
        else:
            self.extractor = Extractor()
            
        self.embedding = Embedding(self.extractor.feat_dim)
        self.classifier = Classifier(num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm:
            x = self.l2_norm(x)
        if self.scale:
            x = self.s * x
        x = self.classifier(x)
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor,self).__init__()
        basenet = models.resnet50(pretrained=True)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])
        self.feat_dim = 2048
        
    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class GN_Extractor(nn.Module):
    def __init__(self):
        super(GN_Extractor,self).__init__()
        self.extractor = create_gn()
        self.feat_dim = 1024

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    def __init__(self, feat_dim=2048):
        super(Embedding,self).__init__()
        self.fc = nn.Linear(feat_dim, 256)

    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x
        
##########################################################################################
##########################################################################################
class LinearDiag(nn.Module):
    def __init__(self, num_features=256, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1) # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out
     
class FeatExemplarAvgBlock(nn.Module):
    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        labels_train_transposed = labels_train.transpose(1,2)
        weight_novel = torch.bmm(labels_train_transposed, features_train)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel
                
class WeightGenerator(nn.Module):
    def __init__(self, num_features=256, use_attention=False):
        super(WeightGenerator,self).__init__()
        
        self.avg_generator = LinearDiag(num_features, bias=False)
        
    def forward(self, imprinted_weights):
        x = self.avg_generator(imprinted_weights)
        return x
        
        
class DynamicNet(nn.Module):
    def __init__(self, num_classes=100, norm=True, scale=True, extractor_type='default'):
        super(DynamicNet,self).__init__()
        
        if extractor_type == 'googlenet':
            self.extractor = GN_Extractor()    
        else:
            self.extractor = Extractor()
            
        self.embedding = Embedding(self.extractor.feat_dim)
        self.classifier = Classifier(num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x, base_class_indexes = None, novel_class_classifiers = None, detach_feature=False):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm:
            x = self.l2_norm(x)
        if self.scale:
            x = self.s * x
        
        if detach_feature:
            x = x.detach()
        
        if base_class_indexes is not None:
            #get classification weigths
            weight_base = self.classifier.fc.weight[base_class_indexes]
            weight = torch.cat((weight_base, novel_class_classifiers))
            
            # apply classifier
            x = F.linear(x, weight)
        else:
            x = self.classifier(x)
                
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
    


