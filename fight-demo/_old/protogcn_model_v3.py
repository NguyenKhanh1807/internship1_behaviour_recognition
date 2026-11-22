"""ProtoGCN (Liu et al., CVPR 2025) - ban port thuan PyTorch, khong can mmcv/mmaction.

Chep 1-1 tu repo goc github.com/firework8/ProtoGCN (commit main, 2025):
  protogcn/models/gcns/protogcn.py      -> GCN_Block, Prototype_Reconstruction_Network, ProtoGCN
  protogcn/models/gcns/utils/gcn.py     -> unit_gcn  (Motion Topology Enhancement)
  protogcn/models/gcns/utils/tcn.py     -> unit_tcn, mstcn
  protogcn/models/heads/simple_head.py  -> SimpleHead
  protogcn/models/losses/Class_Specific_Contrastive_Loss.py -> ClassSpecificContrastiveLoss
  protogcn/utils/graph.py (layout coco_new, mode random)
  protogcn/datasets/pipelines/pose_related.py: Kinetics_Transform (COCO-17 -> 20 khop)
Ten tham so giu nguyen nhu ban goc de nap duoc checkpoint chinh thuc (backbone.*, cls_head.*).
"""
import copy as cp

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-4

# ----------------------------- do thi coco_new (20 khop) -----------------------------
# 0-16: COCO-17 ; 17 = giua 2 hong ; 18 = giua cot song ; 19 = giua 2 vai (tam do thi)
COCO_NEW_INWARD = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 17), (12, 17), (9, 7), (7, 5), (10, 8), (8, 6),
                   (5, 19), (6, 19), (1, 0), (3, 1), (2, 0), (4, 2), (0, 19), (17, 18), (18, 19)]
FLIP20 = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19]


def kinetics_transform(x):
    """(..., 17, C) -> (..., 20, C): them 3 khop ao (giong Kinetics_Transform cua ProtoGCN, ca kenh score)."""
    hip = (x[..., 11, :] + x[..., 12, :]) / 2
    sho = (x[..., 5, :] + x[..., 6, :]) / 2
    mid = (hip + sho) / 2
    return np.concatenate([x, hip[..., None, :], mid[..., None, :], sho[..., None, :]], -2).astype(np.float32)


# pose_related.py JointToBone(dataset='coco_new'): bone[v1] = joint[v1] - joint[v2]; score = trung binh 2 dau xuong
BONE_PAIRS_COCO_NEW = ((0, 19), (1, 0), (2, 0), (3, 1), (4, 2), (5, 19), (6, 19), (7, 5), (8, 6), (9, 7), (10, 8),
                       (11, 17), (12, 17), (13, 11), (14, 12), (15, 13), (16, 14), (17, 18), (18, 19), (19, 19))


def gen_feat(x, modality="j"):
    """GenSkeFeat(dataset='coco_new', feats=[modality]) cho x (..., T, 20, 3), T = truc -3.
    j: toa do khop | b: vector xuong | jm: van toc khop (hieu 2 frame) | bm: van toc xuong.
    Theo code goc: voi coco_new, ToMotion lay hieu ca kenh score (khong trung binh)."""
    if modality in ("b", "bm"):
        b = np.zeros_like(x)
        for v1, v2 in BONE_PAIRS_COCO_NEW:
            b[..., v1, :] = x[..., v1, :] - x[..., v2, :]
            b[..., v1, 2] = (x[..., v1, 2] + x[..., v2, 2]) / 2
        x = b
    if modality in ("jm", "bm"):
        m = np.zeros_like(x)
        m[..., :-1, :, :] = np.diff(x, axis=-3)
        x = m
    return np.ascontiguousarray(x, dtype=np.float32)


def random_graph(num_node=20, num_filter=8, init_std=0.02, init_off=0.04):
    """graph.py: mode='random' - A khoi tao ngau nhien (K=8 to-po con), hoc hoan toan tu du lieu."""
    return np.random.randn(num_filter, num_node, num_node) * init_std + init_off


# ----------------------------------- khoi co ban -----------------------------------
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels) if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class mstcn(nn.Module):
    """TCN da ti le: 4 nhanh gian no (d=1..4) + max-pool + 1x1, cong them 1 'khop toan cuc' (trung binh V)."""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_joints=25, dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'], stride=1):
        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.act = nn.ReLU()
        self.num_joints = num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))
        mid_channels = out_channels // num_branches
        rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            if cfg[0] == 'max':
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None)))
        self.branches = nn.ModuleList(branches)
        tin = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(nn.BatchNorm2d(tin), self.act, nn.Conv2d(tin, out_channels, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)
        out = torch.cat([b(x) for b in self.branches], dim=1)
        local_feat = out[..., :V]
        global_feat = torch.einsum('nct,v->nctv', out[..., V], self.add_coeff[:V])
        return self.drop(self.bn(self.transform(local_feat + global_feat)))


class unit_gcn(nn.Module):
    """GCN voi Motion Topology Enhancement: A_dong = A_hoc + alpha*tanh(x1_i - x2_j) + beta*softmax(x1_i . x2_j)."""

    def __init__(self, in_channels, out_channels, A, ratio=0.125, intra_act='softmax', inter_act='tanh'):
        super().__init__()
        self.num_subsets = A.size(0)
        self.mid_channels = int(ratio * out_channels)
        mid, K = self.mid_channels, self.num_subsets
        self.act = nn.ReLU()
        self.intra_act, self.inter_act = intra_act, inter_act
        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(nn.Conv2d(in_channels, mid * K, 1), nn.BatchNorm2d(mid * K), self.act)
        self.post = nn.Conv2d(mid * K, out_channels, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)
        self.alpha = nn.Parameter(torch.zeros(K))
        self.beta = nn.Parameter(torch.zeros(K))
        self.conv1 = nn.Conv2d(in_channels, mid * K, 1)
        self.conv2 = nn.Conv2d(in_channels, mid * K, 1)
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        res = self.down(x)
        A = self.A[None, :, None, None]                                         # 1 K 1 1 V V
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1 = self.conv1(x).reshape(n, self.num_subsets, self.mid_channels, -1, v).mean(-2, keepdim=True)
        x2 = self.conv2(x).reshape(n, self.num_subsets, self.mid_channels, -1, v).mean(-2, keepdim=True)
        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)                              # N K C 1 V V
        inter_graph = getattr(self, self.inter_act)(diff) * self.alpha[0]
        A = inter_graph + A
        intra_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        intra_graph = getattr(self, self.intra_act)(intra_graph) * self.beta[0]
        A = (intra_graph + A).squeeze(3)                                        # N K C V V
        x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous().reshape(n, -1, t, v)
        x = self.post(x)
        gcl_graph = (inter_graph + intra_graph).squeeze(3).reshape(n, -1, v, v)  # do thi chuyen dong cho PRN
        return self.act(self.bn(x) + res), gcl_graph


class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        x, gcl_graph = self.gcn(x, A)
        return self.relu(self.tcn(x) + res), gcl_graph


class Prototype_Reconstruction_Network(nn.Module):
    """PRN: moi canh (i,j) cua do thi chuyen dong -> softmax tren 400 prototype -> tai tao = to hop prototype."""

    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias=False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.memory_matrix(torch.softmax(self.query_matrix(x), dim=-1)))


class ProtoGCN(nn.Module):
    def __init__(self, A, in_channels=3, base_channels=96, ch_ratio=2, num_stages=10, inflate_stages=[5, 8],
                 down_stages=[5, 8], num_prototype=400, **kwargs):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))                  # data_bn_type 'VC'
        lw_kwargs = [cp.deepcopy(kwargs) for _ in range(num_stages)]
        modules = [GCN_Block(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            cin = base_channels
            inflate_times += i in inflate_stages
            cout = int(96 * ch_ratio ** inflate_times + EPS)
            base_channels = cout
            modules.append(GCN_Block(cin, cout, A.clone(), stride, **lw_kwargs[i - 1]))
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.post = nn.Conv2d(base_channels, base_channels, 1)
        self.bn = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU()
        self.prn = Prototype_Reconstruction_Network(384, num_prototype)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        graphs = []
        for blk in self.gcn:
            x, g = blk(x)
            graphs.append(g)
        x = x.reshape((N, M) + x.shape[1:])
        c_graph = x.size(2)
        graph = graphs[-1].view(N, M, c_graph, V, V).mean(1).view(N, c_graph, V * V)
        re = torch.stack([self.prn(graph[i].permute(1, 0)).permute(1, 0).view(c_graph, V, V) for i in range(N)])
        re = self.relu(self.bn(self.post(re)))
        return x, re.mean(1).view(N, -1)                                         # (N,M,C,T,V), (N, V*V)


class ClassSpecificContrastiveLoss(nn.Module):
    """CSC: chieu do thi tai tao (V*V) -> 256 chieu, keo ve 'tam lop' (EMA cua mau dung), day xa tam lop khac."""

    def __init__(self, n_class, n_channel=400, h_channel=256, tmp=0.125, mom=0.9, pred_threshold=0.0):
        super().__init__()
        self.n_class, self.tmp, self.mom, self.pred_threshold = n_class, tmp, mom, pred_threshold
        self.register_buffer('avg_f', torch.randn(h_channel, n_class), persistent=False)
        self.cl_fc = nn.Linear(n_channel, h_channel)
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feature, lbl, logit):
        feature = self.cl_fc(feature)
        eye = torch.eye(self.n_class, device=lbl.device)
        pred_one, lbl_one = eye[logit.max(1)[1]], eye[lbl]
        prob = torch.softmax(logit, 1)
        mask = lbl_one * pred_one * (prob > self.pred_threshold).float()          # chi mau du doan DUNG
        f = feature.permute(1, 0)
        mask_sum = mask.sum(0, keepdim=True)
        f_mask = (f @ mask) / (mask_sum + 1e-12)
        has = (mask_sum > 1e-8).float()
        has = torch.where(has > 0.1, torch.full_like(has, self.mom), torch.ones_like(has))
        f_mem = self.avg_f.detach() * has + (1 - has) * f_mask
        with torch.no_grad():
            self.avg_f = f_mem.detach()
        fn = feature / (feature.norm(dim=1, keepdim=True) + 1e-12)
        fm = f_mem.permute(1, 0)
        fm = fm / (fm.norm(dim=-1, keepdim=True) + 1e-12)
        score = (fm @ fn.permute(1, 0) / self.tmp).permute(1, 0).contiguous()   # B x n_class
        return self.loss(score, lbl).mean()


class SimpleHead(nn.Module):
    def __init__(self, num_classes, in_channels=384, weight=0.2, n_channel=400):
        super().__init__()
        self.weight = weight
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.csc_loss = ClassSpecificContrastiveLoss(num_classes, n_channel)
        self.ce = nn.CrossEntropyLoss()
        nn.init.normal_(self.fc_cls.weight, 0, 0.01), nn.init.zeros_(self.fc_cls.bias)

    def forward(self, x):
        N, M, C, T, V = x.shape
        return self.fc_cls(x.mean((3, 4)).mean(1))                              # GAP (T,V) roi trung binh M nguoi

    def loss(self, cls_score, graph, label):
        """L = CE + 0.2 * CSC (base.py: loss_cls_1 + weight * loss_cls_2)."""
        return self.ce(cls_score, label) + self.weight * self.csc_loss(graph, label.detach(), cls_score.detach())


class Recognizer(nn.Module):
    def __init__(self, num_classes, num_prototype=400):
        super().__init__()
        self.backbone = ProtoGCN(random_graph(), num_prototype=num_prototype,
                                 tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'])
        self.cls_head = SimpleHead(num_classes)

    def forward(self, x):
        feat, graph = self.backbone(x)
        return self.cls_head(feat), graph


def load_pretrained(model, path):
    """Nap checkpoint chinh thuc (FineGYM/K400/NTU). Bo fc_cls (so lop khac) va reset thong ke data_bn
    (checkpoint goc dung toa do pixel, du lieu cua minh da PreNormalize2D ve [-1, 1])."""
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    own = model.state_dict()
    keep, skip = {}, []
    for k, v in sd.items():
        k2 = k[7:] if k.startswith('module.') else k
        if k2 in own and own[k2].shape == v.shape and not k2.startswith('cls_head.fc_cls'):
            keep[k2] = v
        else:
            skip.append(k2)
    miss = [k for k in own if k not in keep]
    model.load_state_dict(keep, strict=False)
    bn = model.backbone.data_bn
    bn.reset_running_stats()
    return len(keep), skip, miss
