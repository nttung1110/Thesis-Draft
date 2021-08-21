import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pdb



def apply_sigmoid(value):
    return 1/(1+np.exp(-value))

class PosePairAttenNet(nn.Module):
    def __init__(self, input_dim_fuse, input_dim_key_atten, num_key):
        super(PosePairAttenNet, self).__init__()

        # self.input_dim = input_dim
        self.input_dim_fuse = input_dim_fuse
        self.input_dim_key_atten = input_dim_key_atten
        self.num_key = num_key

        self.attention_1 = nn.Linear(self.input_dim_key_atten*2, 512)
        self.attention_2 = nn.Linear(512, 64)
        self.attention_3 = nn.Linear(64, 1)

        self.fc1 = nn.Linear(self.input_dim_fuse + 2052, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)


    def forward(self, fuse_info, query_vector, key_vector, value_vector):
        
        kq_concat = torch.cat((key_vector, query_vector
                                            .expand(key_vector.shape[0], key_vector.shape[1], key_vector.shape[2])), dim=2)

        affinity_kq_1 = F.relu(self.attention_1(kq_concat))
        affinity_kq_2 = F.relu(self.attention_2(affinity_kq_1))
        affinity_kq_3 = self.attention_3(affinity_kq_2)

        atten_weights = F.softmax(affinity_kq_3, dim=1)
        aggregate_feature_pose = torch.bmm(value_vector.transpose(1, 2),
                                 atten_weights)
        
        aggregate_feature_pose = aggregate_feature_pose.squeeze(2)
        fuse_info = fuse_info.squeeze(1)

        x = torch.cat((aggregate_feature_pose, fuse_info), 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x

    def forward_return_attentional_map(self, fuse_info, query_vector, key_vector, value_vector):
        kq_concat = torch.cat((key_vector, query_vector
                                            .expand(key_vector.shape[0], key_vector.shape[1], key_vector.shape[2])), dim=2)

        affinity_kq_1 = F.relu(self.attention_1(kq_concat))
        affinity_kq_2 = F.relu(self.attention_2(affinity_kq_1))
        affinity_kq_3 = self.attention_3(affinity_kq_2)

        atten_weights = F.softmax(affinity_kq_3, dim=1)
        aggregate_feature_pose = torch.bmm(value_vector.transpose(1, 2),
                                 atten_weights)
        
        aggregate_feature_pose = aggregate_feature_pose.squeeze(2)
        fuse_info = fuse_info.squeeze(1)

        x = torch.cat((aggregate_feature_pose, fuse_info), 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x, atten_weights