from collections import defaultdict
import torch.nn

from helper import *
from model.ragcn_conv import RAGCNConv


class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p = params
		self.act = torch.tanh
		self.bceloss = torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)


class RAGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, entity_matrix, relation_matrix, params=None):
		super(RAGCNBase, self).__init__(params)

		self.edge_index = edge_index
		self.edge_type = edge_type
		self.entity_matrix = entity_matrix
		self.relation_matrix = relation_matrix

		self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
		self.device = self.edge_index.device

		self.init_rel = get_param((num_rel, self.p.init_dim))
		# if self.p.score_func == 'transe':
		# 	self.init_rel = get_param((num_rel,   self.p.init_dim))
		# else:
		# 	self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.occurrence_feature:
			self.reconstruct_entity_fc = torch.nn.Linear(self.p.init_dim * 3, self.p.init_dim)
			re_init_embed = self.reconstruct_occurrence_entity_embedding()
			self.reconstruct_relation_fc = torch.nn.Linear(self.p.init_dim * 2, self.p.init_dim)
			re_init_rel = self.reconstruct_occurrence_relation_embedding()

			self.init_embed = Parameter(re_init_embed)
			if self.p.score_func == 'transe':
				self.init_rel = Parameter(re_init_rel)
			else:
				self.init_rel = Parameter(torch.cat([re_init_rel, re_init_rel], dim=0))
		else:
			if self.p.score_func == 'transe':
				self.init_rel = get_param((num_rel,   self.p.init_dim))
			else:
				self.init_rel = get_param((num_rel*2, self.p.init_dim))

		self.conv1 = RAGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
		self.conv2 = RAGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):

		r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)	# TransE -> [474, 200]; Others [237, 200]
		x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x = drop1(x)
		x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) if self.p.gcn_layer == 2 else (x, r)
		x = drop2(x) if self.p.gcn_layer == 2 else x

		sub_emb = torch.index_select(x, 0, sub)
		rel_emb = torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x, r

	def reconstruct_occurrence_entity_embedding(self):
		re_init_embed = []
		for i, entity_emb in enumerate(self.init_embed):
			if i >= len(self.entity_matrix):
				re_init_embed.append(torch.cat([entity_emb] * 3).view(1, -1))
				continue
			co_entities = list(self.entity_matrix[i][0])
			co_relations = list(self.entity_matrix[i][1])

			co_entities_embedding = self.init_embed[co_entities]
			co_relations_embedding = self.init_rel[co_relations]

			co_entities_embedding = torch.mean(co_entities_embedding, 0)
			co_relations_embedding = torch.mean(co_relations_embedding, 0)
			if self.p.occurrence_feature_method == 'linear':
				re_init_embed.append(torch.cat([entity_emb, co_entities_embedding, co_relations_embedding]).unsqueeze(0))
			else:
				re_init_embed.append(1/3 * entity_emb + 1/3 * co_entities_embedding + 1/3 * co_relations_embedding)

		re_init_embed = torch.cat(re_init_embed, dim=0)
		if self.p.occurrence_feature_method == 'linear':
			re_init_embed = self.reconstruct_entity_fc(re_init_embed)
		else:
			re_init_embed = re_init_embed.view([-1, self.p.init_dim])
		return re_init_embed

	def reconstruct_occurrence_relation_embedding(self):
		re_init_rel = []
		for i, rel_emb in enumerate(self.init_rel):
			co_entities = list(self.relation_matrix[i])
			co_entities_embedding = self.init_embed[co_entities]
			re_init_rel.append(torch.cat([rel_emb, torch.mean(co_entities_embedding, 0)]).unsqueeze(0))
		re_init_embed = torch.cat(re_init_rel, dim=0)
		return self.reconstruct_relation_fc(re_init_embed)


class RAGCN_TransE(RAGCNBase):
	def __init__(self, edge_index, edge_type, entity_matrix, relation_matrix, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, entity_matrix, relation_matrix, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):
		"""
		Parameters
		----------
		sub: [128, ]
		rel: [128, ]
		Returns
		-------
		"""
		sub_emb, rel_emb, all_ent, all_rel = self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb = sub_emb + rel_emb

		x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		score = torch.sigmoid(x)

		return score


class RAGCN_DistMult(RAGCNBase):
	def __init__(self, edge_index, edge_type, entity_matrix, relation_matrix, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, entity_matrix, relation_matrix, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent, all_rel = self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb = sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score


class RAGCN_ConvE(RAGCNBase):
	def __init__(self, edge_index, edge_type, entity_matrix, relation_matrix, rel2id, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, entity_matrix, relation_matrix, params)

		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h = int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w = self.p.k_h - self.p.ker_sz + 1
		self.flat_sz = flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

		if self.p.use_entailment:
			self.rel_fc = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim - self.p.entailment_dim)
			self.entailment_fc = torch.nn.Linear(self.p.embed_dim, self.p.entailment_dim)
			self.rel2id = rel2id
			self.relation_reverse_len = len(relation_matrix) * 2
			self.entailment = Parameter(torch.Tensor(self.relation_reverse_len, self.p.embed_dim), requires_grad=False)
			xavier_normal_(self.entailment.data)
			self.bn3 = torch.nn.BatchNorm1d(self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed = e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp = torch.cat([e1_embed, rel_embed], 1)
		stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent, all_rel = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		if self.p.use_entailment:
			self.get_entailment_relation(all_rel)
			entailment = self.entailment[rel]
			entailment = self.entailment_fc(entailment)
			rel_emb = self.rel_fc(rel_emb)
			rel_emb = torch.cat([rel_emb, entailment], dim=1)
			rel_emb = self.bn3(rel_emb)

		stk_inp = self.concat(sub_emb, rel_emb)
		x = self.bn0(stk_inp)
		x = self.m_conv1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.feature_drop(x)
		x = x.view(-1, self.flat_sz)
		x = self.fc(x)
		x = self.hidden_drop2(x)
		x = self.bn2(x)
		x = F.relu(x)

		x = torch.mm(x, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

	def get_entailment_relation(self, rel_emb):

		entailment_sta = defaultdict(int)
		# with open('./data/' + self.p.dataset + '/' + 'entailment.txt', 'r', encoding='UTF-8') as f:
		with open('./data/' + self.p.dataset + '/' + self.p.entailment_name, 'r', encoding='UTF-8') as f:
			for line in f:
				line = line.strip().split()
				rule_head = self.rel2id.get(line[1])
				with torch.no_grad():
					self.entailment[rule_head] += float(line[2]) * rel_emb[rule_head]
				entailment_sta[rule_head] += 1

		for rel_id, num in entailment_sta.items():
			self.entailment[rel_id] /= num


class RAGCN_Entailment(RAGCNBase):
	def __init__(self, edge_index, edge_type, entity_matrix, relation_matrix, rel2id, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, entity_matrix, relation_matrix, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

		self.relation_reverse_len = len(relation_matrix) * 2
		self.rel2id = rel2id
		self.entailment_matrix = get_param((self.relation_reverse_len, self.p.init_dim))
		self.entailment = Parameter(
			torch.nn.init.zeros_(torch.empty(self.relation_reverse_len, self.p.init_dim)),
			requires_grad=False
		)

	def forward(self, sub, rel):
		sub_emb, rel_emb, all_ent, all_rel = self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb = sub_emb + rel_emb - self.entailment[rel]

		x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		score = torch.sigmoid(x)

		return score

	def get_entailment(self):
		entailment_sta = defaultdict(int)
		with open('./data/' + self.p.dataset + '/' + 'entailment.txt', 'r', encoding='UTF-8') as f:
			for line in f:
				line = line.strip().split()
				rule_head = self.rel2id.get(line[1])
				self.entailment[rule_head] += float(line[2]) * self.entailment_matrix[rule_head]
				entailment_sta[rule_head] += 1

		for rel_id, num in entailment_sta.items():
			self.entailment[rel_id] /= num
		return self.entailment

