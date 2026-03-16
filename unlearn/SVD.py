import numpy as np
import torch
from collections import OrderedDict

from collections import OrderedDict
import torch

def get_projection_matrix(device, Mr, Mf, freeze_except_last: bool=False):
    """
    Mr, Mf: dict of projection components for each layer key
    freeze_except_last: if True, only compute real projections for the final head.
    """
    update_dict = OrderedDict()
    for act in Mr.keys():
        # identity projection
        I = torch.eye(Mf[act].shape[0], device=device)
        if freeze_except_last:
            # if freeze_except_last is/be True，rule/thenonlykeepafteronelayer's/ofmatrix
            #  act isnotis“afteronelayer”keycharacter：ResNet/VGG 's/of fc、classifier，ViT 's/of heads
            # thisin act styleincontain 'fc' or 'classifier' or 'heads'
            if not (act.startswith("fc") or act.startswith("classifier") or act.startswith("heads")):
                #update_dict[act] = I
                continue
        #  SVD unlearn 
        mr = Mr[act]
        mf = Mf[act]
        # P = I - (Mf - Mf @ Mr)
        update_dict[act] = I - (mf - mf.matmul(mr))
    if freeze_except_last:
        print("Only updating the last layer's projection matrix.")
        # if freeze_except_last is/be True，rule/thenonlykeepafteronelayer's/ofmatrix
        # thisinafteronelayer's/ofkeycharactercontain 'fc' or 'classifier' or 'heads'
        # for example ResNet 's/of fc，VGG 's/of classifier，ViT 's/of heads    
    return update_dict



def SVD_unlearn(args, model, device, retain_loader, forget_loader, train_loader, test_loader, train_dataset, val_index=None, **kwargs):
    from unlearn.tools import maybe_eval_and_save
    from utils import test
    model.eval()
    index_list = []
    targets = np.array(train_dataset.targets)
    for i in range(args.num_classes):
        if i != args.unlearn_class[0]:
            class_i_index = np.intersect1d(np.where(i == targets)[0], val_index)
            index_list.extend(class_i_index[:int(args.SVD_samples // (args.num_classes - 1))])
    small_retain_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, index_list), batch_size=args.SVD_samples, shuffle=True
    )
    small_forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=args.SVD_samples, shuffle=True
    )
    with torch.no_grad():
        for data, _ in small_retain_loader:
            data = data.to(device)
            Mr = model.get_scaled_projections(data, args.SVD_alpha_r, args.SVD_max_patches)
            break
        for data, _ in small_forget_loader:
            data = data.to(device)
            Mf = model.get_scaled_projections(data, args.SVD_alpha_f, args.SVD_max_patches)
            break
    model.project_weights(get_projection_matrix(device, Mr, Mf, freeze_except_last=args.freeze_except_last))

    retain_acc, forget_acc, metric = test(
            model, device, test_loader,
            args.unlearn_class, args.class_label_names, args.num_classes,
            job_name=args.unlearn_method, set_name=f"Test Set"
        )
    model.history_log = {
         "retain_acc": [retain_acc],
        "forget_acc": [forget_acc],
    }
    return model


