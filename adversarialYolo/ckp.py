import torch

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, scheduler):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    scheduler: scheduler we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize scheduler from checkpoint to scheduler
    scheduler.load_state_dict(checkpoint['scheduler'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # loss
    ep_det_loss = checkpoint['ep_det_loss']
    ep_nps_loss = checkpoint['ep_nps_loss']
    ep_tv_loss = checkpoint['ep_tv_loss']
    ep_loss = checkpoint['ep_loss']
    # return model, scheduler, epoch value, min validation loss 
    return model, scheduler, checkpoint['epoch'], valid_loss_min.item(), ep_det_loss.item(), ep_nps_loss.item(), ep_tv_loss.item(), ep_loss.item()