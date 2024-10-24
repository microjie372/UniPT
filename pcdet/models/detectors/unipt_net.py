from .detector3d_template_unipt import Detector3DTemplate_UniPT, Detector3DTemplate_UniPT_DB_3
from pcdet.utils import common_utils

class UniPT(Detector3DTemplate_UniPT):
    def __init__(self, model_cfg, num_class, num_class_s2, dataset, dataset_s2, source_one_name):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, dataset=dataset,
                         dataset_s2=dataset_s2, source_one_name=source_one_name)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
            loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()

            ret_dict = {
                'loss': loss_1 + loss_2
            }
            return ret_dict, tb_dict_1, disp_dict_1
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def get_training_loss_s1(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss_s1()
        loss_cls, tb_dict = self.dense_head_s1.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_cls': loss_cls.item(),
            **tb_dict
        }

        loss = loss_rpn + loss_cls
        return loss, tb_dict, disp_dict

    def get_training_loss_s2(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss_s2()
        loss_cls, tb_dict = self.dense_head_s2.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_cls': loss_cls.item(),
            **tb_dict
        }

        loss = loss_rpn + loss_cls
        return loss, tb_dict, disp_dict
    

class UniPT_DB_3(Detector3DTemplate_UniPT_DB_3):
    def __init__(self, model_cfg, num_class, num_class_s2, num_class_s3, dataset, dataset_s2, dataset_s3, source_one_name, source_1):
        super().__init__(model_cfg=model_cfg, num_class=num_class, num_class_s2=num_class_s2, num_class_s3=num_class_s3, 
                         dataset=dataset, dataset_s2=dataset_s2, dataset_s3=dataset_s3, source_one_name=source_one_name, source_1=source_1)
        self.module_list = self.build_networks()
        self.source_one_name = source_one_name
        self.source_1 = source_1

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss_1, tb_dict_1, disp_dict_1 = self.get_training_loss_s1()
            loss_2, tb_dict_2, disp_dict_2 = self.get_training_loss_s2()
            loss_3, tb_dict_3, disp_dict_3 = self.get_training_loss_s3()

            ret_dict = {
                'loss': loss_1 + loss_2 + loss_3
            }
            return ret_dict, tb_dict_1, disp_dict_1
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def get_training_loss_s1(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss_s1()
        loss_cls, tb_dict = self.dense_head_s1.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_cls': loss_cls.item(),
            **tb_dict
        }

        loss = loss_rpn + loss_cls
        return loss, tb_dict, disp_dict

    def get_training_loss_s2(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss_s2()
        loss_cls, tb_dict = self.dense_head_s2.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_cls': loss_cls.item(),
            **tb_dict
        }

        loss = loss_rpn + loss_cls
        return loss, tb_dict, disp_dict
    
    def get_training_loss_s3(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss_s3()
        loss_cls, tb_dict = self.dense_head_s3.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_cls': loss_cls.item(),
            **tb_dict
        }

        loss = loss_rpn + loss_cls
        return loss, tb_dict, disp_dict

