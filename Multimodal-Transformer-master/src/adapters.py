import torch 
import torch.nn as nn 
import math 

  # define adapator 
class Multiheadattention(nn.Module):
  def __init__(self, num_attention_heads = 4, hidden_dim = 32):
    super().__init__()

    self.num_attention_heads = num_attention_heads
    self.attention_head_size = int(hidden_dim / num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(hidden_dim, self.all_head_size)
    self.key = nn.Linear(hidden_dim, self.all_head_size)
    self.value = nn.Linear(hidden_dim, self.all_head_size)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, k, q, v, attention_mask):
    bs, num_head, seq_len, att_size = q.shape
    score = torch.einsum("bnqd,bnkd->bnqk",q, k)
    # normalize the scores

    if attention_mask != None:
      score = score + attention_mask
    score = score / math.sqrt(att_size)
    score = score.softmax(dim = -1)
    
    # multiply the attention scores to the value and get back V'
    vv = torch.einsum("bnlk,bnql->bqnk",v, score)
    return vv.reshape(bs, seq_len, num_head * att_size)


  def forward(self, Q,K,V, attention_mask = None):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # dimensions of all *_layers are [bs, num_attention_heads, seq_len, attention_head_size]
    # print('SHAPE: ',K.shape)
    key_layer = self.transform(K, self.key)
    value_layer = self.transform(V, self.value)
    query_layer = self.transform(Q, self.query)
    # calculate the multi-head attention
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class Adaptor(nn.Module):
    def __init__(self, ndim, rank=32):
        
        super().__init__()
        self.conv = nn.Conv1d(in_channels = ndim, out_channels = 30, kernel_size=1, padding=0, bias=False)
        self.rank = rank
        self.down_proj = nn.Linear(ndim, rank)
        self.dropout = nn.Dropout(p=0.1)
        self.up_proj = nn.Linear(rank, ndim)
        nn.init.normal_(self.down_proj.weight, std = 1/rank)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, input):
        # import pdb; pdb.set_trace()
        input =  input.transpose(1, 2)
        down_projection = self.down_proj(input)
        #down_projection = self.dropout(down_projection)
        up_projection = self.up_proj(down_projection)
        up_projection = self.dropout(up_projection)
        sum_projection = up_projection + input 

        sum_projection = sum_projection.transpose(1, 2)
        
        return self.conv(sum_projection)


class ContextConcatAdaptor(nn.Module):
    '''
    Late fusion adaptor to replace model.proj2
    Concat the context with the last_hs as the input of the classification layer
    '''
    def __init__(self, proj, context_dim):
        super().__init__()
        self.proj = proj 
        self.linear = nn.Linear(context_dim, proj.in_features)
    
    def forward(self, last_hs, context):
        # context is one single vector, can be CLS token
        context = self.linear(context)
        # simple additive fusion 
        return self.proj(last_hs) + context

# class ContextGateAdaptor(nn.Module):
#     '''
#     Adapter to fuse context with input embedding with gate 
#     Gate is the ordinary complementary gates (OCG)
#     https://arxiv.org/pdf/2102.10407.pdf
#     '''
#     def __init__(self, ndim, rank=32):
        
#         super().__init__()
#         self.conv = nn.Conv1d(in_channels = ndim, out_channels = 30, kernel_size=1, padding=0, bias=False)
#         self.rank = rank
#         self.adapter_main = nn.Sequential(
#             nn.Linear(ndim, rank),
#             nn.Linear(rank, ndim),
#             nn.Dropout(p=0.1)
#         )
#         nn.init.normal_(self.adapter_main[0].weight, std = 1/rank)
#         nn.init.zeros_(self.adapter_main[1].weight)

#         # [T;T_c] shape (T+T_c, ndim)
#         # gate.shape = (T+T_c,T+T_c)
#         # W shape(ndim,T_c)
#         self.attention = Multiheadattention(num_attention_heads=4, hidden_dim=ndim)
#         self.layer_norm = nn.LayerNorm(ndim)
#         self.threshold = 0.1 # the threshold of the gate


#     def forward(self, input, context):
#         # import pdb; pdb.set_trace()
#         input =  input.transpose(1, 2)
#         att = self.attention(Q=input, K=context, V=context)
        
#         input_sig = torch.sigmoid(input)
#         att_sig = 1 - input_sig
#         G_input = torch.where(input_sig > self.threshold, att_sig, 0)
#         G_att = torch.where(att_sig > self.threshold, att_sig, 0)

#         input = G_input * input + G_att * att
#         input = self.layer_norm(input)

#         up_projection = self.adapter_main(input)
#         input_main = up_projection + input # input of main modality after lora

#         sum_projection = input_main.transpose(1, 2)
        
#         return self.conv(sum_projection)




class ContextGateAdaptor(nn.Module):
    '''
    Adapter to fuse context with input embedding with gate 
    Gate is the ordinary complementary gates (OCG)
    https://arxiv.org/pdf/2102.10407.pdf
    '''
    def __init__(self, ndim, rank=32):
        
        super().__init__()
        # self.conv = nn.Conv1d(in_channels = ndim, out_channels = 30, kernel_size=1, padding=0, bias=False)
        # self.rank = rank
        # self.adapter_main = nn.Sequential(
        #     nn.Linear(ndim, rank),
        #     nn.Linear(rank, ndim),
        #     nn.Dropout(p=0.1)
        # )
        # nn.init.normal_(self.adapter_main[0].weight, std = 1/rank)
        # nn.init.zeros_(self.adapter_main[1].weight)

        # [T;T_c] shape (T+T_c, ndim)
        # gate.shape = (T+T_c,T+T_c)
        # W shape(ndim,T_c)
        self.attention = Multiheadattention(num_attention_heads=4, hidden_dim=ndim)
        self.layer_norm = nn.LayerNorm(ndim)
        self.threshold = 0.1 # the threshold of the gate

        self.adapter = Adaptor(ndim, rank = rank)



    def forward(self, input, context):
        #import pdb; pdb.set_trace()
        
        if context != None:
          #print('pass')
          input =  input.transpose(1, 2)
          # import pdb; pdb.set_trace()
          att = self.attention(Q=input, K=context, V=context)
          
          input_sig = torch.sigmoid(input)
          att_sig = 1 - input_sig
          G_input = torch.where(input_sig > self.threshold, att_sig, 0)
          G_att = torch.where(att_sig > self.threshold, att_sig, 0)

          input = G_input * input + G_att * att
          input = self.layer_norm(input)

          # up_projection = self.adapter_main(input)
          # input_main = up_projection + input # input of main modality after lora

          #sum_projection = input_main.transpose(1, 2)
        
        #return self.conv(sum_projection)
        return self.adapter(input)
    

class GatedAttention(nn.Module):
  '''
  Adapter to fuse context with input embedding with gate 
  Gate is the ordinary complementary gates (OCG)
  https://arxiv.org/pdf/2102.10407.pdf
  '''
  def __init__(self, ndim, orig_self_attn, rank=32):
      
      super().__init__()

      self.gated_attention = Multiheadattention(num_attention_heads=4, hidden_dim = ndim)
      self.layer_norm = nn.LayerNorm(ndim)
      self.threshold = 0.1 # the threshold of the gate

      self.self_attn = orig_self_attn



  def forward(self, input, context):
      #import pdb; pdb.set_trace()
      
      if context != None:
        #print('pass')

        #pass both context and utterance into original pretrained self-attention
        input_self_attn = self.self_attn(input)
        if type(input_self_attn) == tuple:
            input_self_attn = input_self_attn[0]
        # input_self_attn = input_self_attn[-1]   #take last

        context_self_attn = self.self_attn(context)
        if type(context_self_attn) == tuple:
            context_self_attn = context_self_attn[0]
        # context_self_attn = context_self_attn[-1]   



        input_self_attn =  input_self_attn.transpose(0,1)
        context_self_attn =  context_self_attn.transpose(0,1)

        # import pdb; pdb.set_trace()
        att = self.gated_attention(Q=input_self_attn, K=context_self_attn, V=context_self_attn)
        
        input_sig = torch.sigmoid(input_self_attn)
        att_sig = 1 - input_sig
        G_input = torch.where(input_sig > self.threshold, att_sig, 0)
        G_att = torch.where(att_sig > self.threshold, att_sig, 0)

        input_self_attn = G_input * input_self_attn + G_att * att
        input_self_attn = self.layer_norm(input_self_attn)

      return input_self_attn


