from helper import *
import dltools
from model.message_passing import MessagePassing


class RAGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p = params
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_rels = num_rels
		self.act = act
		self.device = None

		self.w_loop = get_param((in_channels, out_channels))
		self.w_in = get_param((in_channels, out_channels))
		self.w_out = get_param((in_channels, out_channels))
		self.w_rel = get_param((in_channels, out_channels))
		self.loop_rel = get_param((1, in_channels))

		self.drop = torch.nn.Dropout(self.p.dropout)
		self.bn = torch.nn.BatchNorm1d(out_channels)

		self.attention_in = dltools.AdditiveAttention(out_channels, out_channels, out_channels, self.p.dropout)
		self.attention_out = dltools.AdditiveAttention(out_channels, out_channels, out_channels, self.p.dropout)
		self.entity_weight_in = get_param((in_channels, in_channels))
		self.entity_weight_out = get_param((in_channels, in_channels))

		if self.p.bias:
			self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	# x.shape = [14541, 200] -> [entity_number, embedding_dimension];
	# edge_index.shape = [2, 544230] -> [sub and obj, sample number];
	# edge_type.shape = [544230] -> [sample_number] rel_embedding = [474, 200]
	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)	# [474, 200] -> [475, 200]
		num_edges = edge_index.size(1) // 2		# edges的数量，因为加入了逆关系
		num_ent = x.size(0)		# 实体数量

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]	# 输出和输入点 in_index -> [2, 272115] out_index -> [2, 272115]
		self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]		# 输出和输入边 in_type-> [272115] out_type -> [272115]

		self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)  # [2, 14541]
		self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)  # [14541]

		self.in_norm = self.compute_norm(self.in_index,  num_ent)  # [272115]
		self.out_norm = self.compute_norm(self.out_index, num_ent)  # [272115]

		in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, edge_norm=self.in_norm, mode='in')
		loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, mode='loop')
		out_res = self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')

		out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		if self.p.bias:
			out = out + self.bias
		out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		if self.p.opn == 'corr':
			trans_embed = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub':
			trans_embed = ent_embed - rel_embed
		elif self.p.opn == 'mult':
			trans_embed = ent_embed * rel_embed
		else:
			raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		if mode == 'loop':
			weight = getattr(self, 'w_{}'.format(mode))
			rel_emb = torch.index_select(rel_embed, 0, edge_type)
			xj_rel = self.rel_transform(x_j, rel_emb)
			out = torch.mm(xj_rel, weight)
		else:
			# entity weight
			if self.p.graph_entity_weight:
				entity_weight = getattr(self, 'entity_weight_{}'.format(mode))
				x_j = torch.mm(x_j, entity_weight)

			# relation weight
			weight = getattr(self, 'w_{}'.format(mode))
			rel_emb = torch.index_select(rel_embed, 0, edge_type)
			xj_rel = self.rel_transform(x_j, rel_emb)
			if self.p.graph_relation_weight:
				out = torch.mm(xj_rel, weight)
			else:
				# out = torch.cat([xj_rel, xj_rel], dim=-1)
				out = xj_rel

			# attention
			if self.p.graph_attention:
				attention = getattr(self, 'attention_{}'.format(mode))
				out = torch.unsqueeze(out, 1)
				attention_encoder = torch.cat([out] * 3, dim=1)
				out = attention(out, attention_encoder, attention_encoder, None)
				out = torch.squeeze(out)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	# edge_index -> [2, 272115]  num_ent -> 14541
	def compute_norm(self, edge_index, num_ent):
		row, col = edge_index	# row->[272115]  col->[272115]
		edge_weight = torch.ones_like(row).float()  # [272115]
		deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # [14541]	# Summing number of weights of the edges
		deg_inv = deg.pow(-0.5)  # [14541]  # D^{-0.5}
		deg_inv[deg_inv == float('inf')] = 0
		norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
