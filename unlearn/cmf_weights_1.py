import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.vit import vit_b_16, vit_b_32,  vit_l_16, vit_l_32, vit_h_14
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as pl


def hardmax_loss(W, H, y):
    # H [B,d], W [d, K]
    HW = H @ W
    yt = y.unsqueeze(1)
    new_HW = HW - HW.gather(1, yt)
    rows = torch.arange(H.shape[0]) # range(B)
    new_HW[rows, y] = -torch.inf
    loss = 100 * torch.max(new_HW)
    return loss

class CMFWeights(nn.Module):
    def __init__(
        self,
        num_classes  = 10,
        feature_dim = 512
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # weights size is (K,d)
        self.register_buffer("weight", torch.zeros(num_classes, feature_dim))
        self.register_buffer("mu",     torch.zeros(feature_dim))     

    @torch.no_grad()
    def update(self, features, labels, momentum) -> torch.Tensor:
        """
        Returns l2-normalized class weights by averaging the features from the same class
        """
        # add each feature to weights accroding to its labels
        #print("[CMF.update] momentum used =", momentum)
        self.weight = momentum * self.weight
        self.weight.index_add_(0, labels, (1 - momentum) * features)
        # normlize to unit norm
        self.weight = F.normalize(self.weight)
        return self.weight
    
    @torch.no_grad()
    def update_class_means(self, features, labels, momentum: float):
        """firstbyclassaveragevalue，againdo EMA；notinthisin normalize。"""
        K, D = self.num_classes, self.feature_dim
        sums   = torch.zeros(K, D, device=features.device, dtype=features.dtype)
        counts = torch.zeros(K,   device=features.device, dtype=features.dtype)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
        mask = counts > 0
        means = torch.zeros_like(sums)
        means[mask] = sums[mask] / counts[mask].unsqueeze(1)
        self.weight[mask] = momentum * self.weight[mask] + (1.0 - momentum) * means[mask]

    def normalized(self, do_norm: bool):
        """getoutputusein logits/to/foralign's/ofweight；needtimeagaindo L2 normoneify。"""
        return F.normalize(self.weight) if do_norm else self.weight
    

class ModelModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.history_log = {}
        args.encoder = args.arch.lower()
        if not hasattr(args, "no_normalization"): args.no_normalization = False
        if not hasattr(args, "CMF_momentum"): args.CMF_momentum = 0.9
        if not hasattr(args, "temperature"): args.temperature = 1.0
        if not hasattr(args, "loss"): args.loss = "CE"
        if not hasattr(args, "optimizer"): args.optimizer = "SGD"
        if not hasattr(args, "learning_rate"): args.learning_rate = 1e-5
        if not hasattr(args, "CMFClassifier"): args.CMFClassifier = True
        if not hasattr(args, "remove_FC"): args.remove_FC = False

        if args.remove_FC and args.CMFClassifier==False:
            raise ValueError("Cannot remove FC when not using CMFClassifier")
        
        if args.remove_FC:
            print("Removing FC layer from the encoder and using CMF classifier.")
        print("CMF_momentum:", args.CMF_momentum)
        
        # ---------- encoder ----------
        if args.encoder == "resnet18":
            full_model = ResNet18(num_classes=args.num_classes, dataset=args.dataset)
        elif args.encoder == "resnet34":
            full_model = ResNet34(num_classes=args.num_classes, dataset=args.dataset)
        else:
            raise ValueError("Unsupported encoder")
        
        if args.remove_FC:
            full_model.fc = nn.Identity()
            self.encoder = full_model  # keeporiginalconstructandlayer name
            self.flatten = nn.Flatten()
            self.feature_dim = 512  # from ResNet config
        else:
            # useuseoriginal encoder
            self.encoder = full_model
            self.feature_dim = args.num_classes  # from ResNet config
        

        #use CMF classifier or linear classifier
        if args.CMFClassifier:
            self.CMFweights = CMFWeights(num_classes=args.num_classes, feature_dim=self.feature_dim)
        else:
            self.linear = nn.Linear(args.feature_dim, args.num_classes, bias=False)

        self.criterion = torch.nn.CrossEntropyLoss()
        # define acc
        self.accuracy = MulticlassAccuracy(num_classes=args.num_classes)

    def extract_features(self, x):
        x = self.encoder(x)
        if x.ndim == 4:
            x = torch.flatten(x, start_dim=1)  # ensures (B, D)
        return x


    @torch.no_grad()
    def recompute_cmf(self, loader, device):
        """
        useone loader（recommended retain_loader）reset CMF 's/of W and μ：
          1) sample-level L2：z = normalize(f)
          2) classaveragevalue m_c
          3) global mean μ（bysamplethisnumberadd）
          4) W_c = normalize(m_c - μ)
        """
        self.eval()
        K, D = self.CMFweights.num_classes, self.CMFweights.feature_dim

        sums   = torch.zeros(K, D, device=device)
        counts = torch.zeros(K,   device=device)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            f = self.extract_features(xb).float()
            z = F.normalize(f, dim=1)              # sample-level L2
            sums.index_add_(0, yb, z)
            counts.index_add_(0, yb, torch.ones_like(yb, dtype=counts.dtype))
            #sums.index_add_(0, yb, f)
            #counts.index_add_(0, yb, torch.ones_like(yb, dtype=counts.dtype))

        means = torch.zeros(K, D, device=device, dtype=torch.float32)
        mask  = counts > 0
        means[mask] = sums[mask] / counts[mask].unsqueeze(1)

        if mask.any():
            mu = (sums[mask].sum(dim=0) / counts[mask].sum()).detach()
        else:
            mu = torch.zeros(D, device=device)

        
        if mask.any():
            means[mask] = means[mask] - mu
            means[mask] = F.normalize(means[mask], dim=1)

        # write
        self.CMFweights.weight.copy_(means)
        self.CMFweights.mu.copy_(mu)

        return None, None, None, None, None

        # —— optionaldegree —— 
        W = self.CMFweights.weight                 # [K, D]
        H = means  

        Wn = F.normalize(W, dim=1)
        Hn = F.normalize(H, dim=1)

        Wn_safe = Wn.clone(); Wn_safe[~mask] = 0.0
        Hn_safe = Hn.clone(); Hn_safe[~mask] = 0.0
        G_WW = Wn_safe @ Wn_safe.t()
        G_HH = Hn_safe @ Hn_safe.t()
        G_WH = Wn_safe @ Hn_safe.t()   

        print("mean G_WW:", G_WW.mean().item())
        print("mean G_HH:", G_HH.mean().item())
        print("mean G_WH:", G_WH.mean().item())

        return Wn.detach().cpu().numpy(), Hn.detach().cpu().numpy(), G_WW.detach().cpu().numpy(), G_HH.detach().cpu().numpy(), G_WH.detach().cpu().numpy()

        
            

            
    def _preprocess_feats_for_cmf(self, f):
        """
        ẑ = normalize( normalize(f) - μ )
        and recompute_cmf generateform's/of W 's/ofkeepsupportconsistent
        """
        z = F.normalize(f, dim=1)
        mu = getattr(self.CMFweights, "mu", None)
        if mu is None or mu.numel() == 0:
            return z
        zc = z - mu.unsqueeze(0)
        zc = F.normalize(zc, dim=1)
        return zc

        #mu = getattr(self.CMFweights, "mu", None)
        #if mu is None or mu.numel() == 0:
        #    return f
        #return f - mu.unsqueseze(0)

    def forward_a(self, batch, stage):
        x,y = batch
        features = self.extract_features(x)

        ##if not self.args.no_normalization:
            #features = F.normalize(features)

        if self.args.CMFClassifier:
            #if stage == "train":
            #    self.CMFweights.update(features=features, labels=y, momentum=self.args.CMF_momentum)
            #weights = self.CMFweights.weight
            z = self._preprocess_feats_for_cmf(features)
            W = self.CMFweights.weight
            logits = (z @ W.t()) * self.args.temperature
            loss = self.criterion(logits, y)
        else:
            if not self.args.no_normalization:
                features = F.normalize(features)
            weights = F.normalize(self.linear.weight) if not self.args.no_normalization else self.linear.weight
            if self.args.loss == "CE":
                logits = (features @ weights.t()) * self.args.temperature
                loss = self.criterion(logits, y)
            elif self.args.loss == "Hardmax":
                loss = hardmax_loss(W=weights.t(), H=features, y=y)
                logits = (features @ weights.t()) * 1000
        
        #if self.args.loss == "CE":
        #    logits = (features @ weights.t()) * self.args.temperature
        #    #logits = self.linear(features)
        #    loss = self.criterion(logits, y)
        #elif self.args.loss == "Hardmax":
        #    loss = hardmax_loss(W = weights.t(), H = features, y = y)
        #    logits = (features @ weights.t()) * 1000
        accuracy = self.accuracy(logits, y)
        
        return loss, accuracy
    
    def forward(self, x):
        features = self.extract_features(x)
        if self.args.CMFClassifier:
            z = self._preprocess_feats_for_cmf(features)   # ★ logicconsistent
            W = self.CMFweights.weight
            logits = (z @ W.t()) * self.args.temperature
            return logits
        else:
            if not self.args.no_normalization:
                features = F.normalize(features)
            W = self.linear.weight
            if not self.args.no_normalization:
                W = F.normalize(W)
            logits = (features @ W.t()) * self.args.temperature
            return logits



    def training_step(self, batch, batch_idx):
        loss, acc = self.forward_a(batch, "train")
        self.log("loss/train", loss, prog_bar=True)
        self.log("acc/train", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward_a(batch, "val")
        self.log("loss/val", loss, prog_bar=True)
        self.log("acc/val", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward_a(batch, "test")
        self.log("loss/test", loss, prog_bar=True)
        self.log("acc/test", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.args.learning_rate,
                                        momentum=0.9,
                                        weight_decay=self.args.weight_decay,
                                        nesterov=True)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs)
        return [optimizer], [scheduler]






