import torch
from torch import nn
import sys
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *

from src.adapters import Adaptor, ContextGateAdaptor, ContextConcatAdaptor, GatedAttention, ContextGateEarlyAdaptor
####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################


def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    fine_tune = False
    freeze = True
    #import pdb; pdb.set_trace()


    if fine_tune:
        model = load_model(hyp_params, name=hyp_params.name)
        # Freeze all parameters
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        

        # finetune the output layer 
        feature_dim = 180
        model.proj1 = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)    # hard code here (to be fixed !)
        model.proj2 = nn.Linear(in_features=feature_dim, out_features=feature_dim, bias=True)
        model.out_layer = nn.Linear(in_features=feature_dim, out_features=1, bias=True)

        # update the conv1d with "LoRA"

        #prev_weight = model.proj_a.weight
        #import pdb; pdb.set_trace()
        # adaptor_v = Adaptor(ndim = hyp_params.orig_d_v)
        # adaptor_a = Adaptor(ndim = hyp_params.orig_d_a)
        
        # adaptor_l = Adaptor(ndim = hyp_params.orig_d_l)
        # #adaptor_v = Adaptor(ndim = hyp_params.orig_d_v)
        # #adaptor_v = ContextGateAdaptor(ndim = hyp_params.orig_d_v, context=True, threshold = 0.4)
        # adaptor_a = Adaptor(ndim = hyp_params.orig_d_a)
        # adaptor_v = Adaptor(ndim = hyp_params.orig_d_v)
        #adaptor_l = Adaptor(ndim = hyp_params.orig_d_l)
        #adaptor_l = ContextGateAdaptor(ndim = hyp_params.orig_d_l, context=True, threshold = 0.4)
        #import pdb; pdb.set_trace()
        #adaptor_v = ContextGateEarlyAdaptor(ndim = hyp_params.orig_d_v, context=True, threshold = 0.9)       # import pdb; pdb.set_trace()
        #adaptor_l = ContextGateEarlyAdaptor(ndim = hyp_params.orig_d_l, context=True, threshold = 0.5)
        #adaptor_l.adapter = adaptor_l_init
        adaptor_v = Adaptor(ndim = hyp_params.orig_d_v)
        adaptor_a = Adaptor(ndim = hyp_params.orig_d_a) 
        adaptor_l = Adaptor(ndim = hyp_params.orig_d_l)    

        print('pass')

           
        
        
        #adaptor_v = ContextGateAdaptor(ndim = hyp_params.orig_d_v, context=True, threshold = 0.65)
        #adaptor_a = Adaptor(ndim = hyp_params.orig_d_a)
        

        model.proj_v = adaptor_v
        model.proj_a = adaptor_a
        model.proj_l = adaptor_l

        import pdb; pdb.set_trace()
       # model.trans_l_mem = GatedAttention(ndim = model.d_l*2, threshold = 0.1, orig_self_attn = model.trans_l_mem)#replace with gated attn
       # model.trans_v_mem = GatedAttention(ndim = model.d_v*2, threshold = 0.65, orig_self_attn = model.trans_v_mem)#replace with gated attn
    
    

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    if hyp_params.aligned or hyp_params.model=='MULT':
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    #torch.cuda.empty_cache()
    #import gc
    #gc.collect()
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']
    
    scheduler = settings['scheduler']

   

    def train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        #prev_weights = None
        for i_batch, (batch_X, batch_Y, batch_META, batch_context) in enumerate(train_loader):
            #import pdb; pdb.set_trace()
            sample_ind, text, audio, vision = batch_X
            context_t, context_v = batch_context
            context_v = context_v.to(torch.float32)
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if ctc_criterion is not None:
                ctc_a2l_module.zero_grad()
                ctc_v2l_module.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr, context_t, context_v = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), context_t.cuda(), context_v.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
            
            ######## CTC STARTS ######## Do not worry about this if not working on CTC
            if ctc_criterion is not None:
                ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module

                audio, a2l_position = ctc_a2l_net(audio) # audio now is the aligned to text
                vision, v2l_position = ctc_v2l_net(vision)
                
                ## Compute the ctc loss
                l_len, a_len, v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
                # Output Labels
                l_position = torch.tensor([i+1 for i in range(l_len)]*batch_size).int().cpu()
                # Specifying each output length
                l_length = torch.tensor([l_len]*batch_size).int().cpu()
                # Specifying each input length
                a_length = torch.tensor([a_len]*batch_size).int().cpu()
                v_length = torch.tensor([v_len]*batch_size).int().cpu()
                
                ctc_a2l_loss = ctc_criterion(a2l_position.transpose(0,1).cpu(), l_position, a_length, l_length)
                ctc_v2l_loss = ctc_criterion(v2l_position.transpose(0,1).cpu(), l_position, v_length, l_length)
                ctc_loss = ctc_a2l_loss + ctc_v2l_loss
                ctc_loss = ctc_loss.cuda() if hyp_params.use_cuda else ctc_loss
            else:
                ctc_loss = 0
            ######## CTC ENDS ########
                
            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    context_t = None
                    context_v = None
                    #preds_i, hiddens_i = net(text_i, audio_i, vision_i, context_t, context_v)
                    preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                    
                    if hyp_params.dataset == 'iemocap':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                ctc_loss.backward()
                combined_loss = raw_loss + ctc_loss
            else:
                #context_t = None
                #import pdb; pdb.set_trace()
                context_t = None
                context_v = None
                #preds, hiddens = net(text, audio, vision, context_t, context_v)
                preds, hiddens = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss + ctc_loss
                combined_loss.backward()
            
            if ctc_criterion is not None:
                torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params.clip)
                torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params.clip)
                ctc_a2l_optimizer.step()
                ctc_v2l_optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False, last_test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []
        return_index = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META, batch_context) in enumerate(loader):
                
                sample_ind, text, audio, vision = batch_X
                context_t, context_v = batch_context
                context_v = context_v.to(torch.float32)
                #import pdb; pdb.set_trace()
                if last_test:
                    return_index.append(batch_META[0])
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr, context_t, context_v = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), context_t.cuda(), context_v.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                    ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                    ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module
                    audio, _ = ctc_a2l_net(audio)     # audio aligned to text
                    vision, _ = ctc_v2l_net(vision)   # vision aligned to text
                
                net = nn.DataParallel(model) if batch_size > 10 else model
                context_t = None
                context_v = None
                #import pdb; pdb.set_trace()
                #preds, _ = net(text, audio, vision, context_t, context_v)
                preds, _ = net(text, audio, vision)
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
        #('preds', preds)
        #print('truths', truths)
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)

        return avg_loss, results, truths, return_index
    #model = load_model(hyp_params, name=hyp_params.name)
    #_, results, truths, video_index = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True, last_test=True)
    #import pdb; pdb.set_trace()
    best_valid = 1e8
    prev_weights = model.proj1.weight
    for epoch in tqdm(range(1, hyp_params.num_epochs+1)):
        #import pdb; pdb.set_trace()
        #prev_weights = model.proj_v.down_proj.weight
        start = time.time()
        train_loss = train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion)
        
        val_loss, _, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=False)
        test_loss, _, _, _ = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)
        end = time.time()
        duration = end-start
        scheduler.step(test_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f} | Train Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss, train_loss))
        print("-"*50)
       # import pdb; pdb.set_trace()
        
        #if val_loss < best_valid:
        if test_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = test_loss
    #import pdb; pdb.set_trace()
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths, video_index = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True, last_test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi':
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        eval_iemocap(results, truths)
    elif hyp_params.dataset == 'mustard_ind':
        eval_mustard(results, truths, video_index)
    elif hyp_params.dataset == 'mustard':
        eval_mustard(results, truths, video_index)
    elif hyp_params.dataset == 'mustard_w_context':
        eval_mustard(results, truths, video_index)
    elif hyp_params.dataset == 'mustard_dep_t_context':
        eval_mustard(results, truths, video_index)
    elif hyp_params.dataset == 'mustard_indep_t_context':
        eval_mustard(results, truths, video_index)
    else:
        eval_mustard(results, truths, video_index)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')
