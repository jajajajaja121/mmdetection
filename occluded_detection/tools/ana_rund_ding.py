from analyse import Analyse
from analyse import Transform
if __name__ == "__main__":
    anno_path = [
                '/Users/urwlee/personal/Mystudy/学校相关/比赛会议等/科赛水下检测/dataset/layouts/train.json'
                 ]
    pic_path = [
                '/Users/urwlee/personal/Mystudy/学校相关/比赛会议等/科赛水下检测/dataset/train/image'
                ]
    analyse = Analyse(anno_path,pic_path,flag_test = False)
    flag_test = False
    flag = 1
    if flag == 0: #vis result
        models = ['/home/dingmingxu/work/work/code/mmdetection_ding/bottle_flaw/round1/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_small_2/epoch_12.pth']
        configs = ['/home/dingmingxu/work/work/code/mmdetection_ding/bottle_flaw/round1/configs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_small_2.py']
        pic_dir = "/home/dingmingxu/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_train1_20191223/images"
        flag_traindata = True
        analyse.vis_result(models, configs, pic_dir, flag_traindata = flag_traindata)
    elif flag == 1: # vis image and label
        label =[]
        image_name = []
        anno_path = [
            '/Users/urwlee/personal/Mystudy/学校相关/比赛会议等/科赛水下检测/dataset/layouts/train.json'
        ]
        pic_path = [
            '/Users/urwlee/personal/Mystudy/学校相关/比赛会议等/科赛水下检测/dataset/train/image'
                    ]
        flag_compare = False
        analyse_ = Analyse(anno_path,pic_path,flag_test = flag_test)
        analyse_.vis_label(analyse=analyse,label = label,image_name = image_name,flag_compare=flag_compare)
    elif flag == 2: # generate result
        pic_path = "/home/zhangming/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_testA_20191223/images/"
        json_out_path = ""
        model2make_json = ""
        config2make_json = ""
        analyse.result_from_dir(config2make_json, model2make_json, json_out_path, pic_path)
    elif flag == 2.5:
        small_config= ""
        big_config=""
        small_model=""
        big_model=""
        json_out_path=""
        pic_path=""
        analyse.result_from_dir_multi(small_config,big_config,small_model,big_model,json_out_path,pic_path)
    elif flag == 3: # remove bg
        DATASET_PATH = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213"
        analyse.wash(DATASET_PATH)
    elif flag==4:
        analyse.draw_cls_colum()
        # analyse.draw_cls_colum_part()
        # analyse.draw_cls_colum_area()
    elif flag==5: # draw good analyse pic
        analyse.ana_boxes()
    elif flag==6:
        aug_save_path = ""
        json_save_path = ""
        analyse.add_aug_data(add_num=2000, aug_save_path=None, json_file_path=None)
    elif flag==7:
        #json_path = "/home/dingmingxu/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_train1_20191223/annotations_without_bg_anno_and_pic_coco.json"
        json_path = "/home/wdh/kaggle/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed.json"
        s_out_path = "/home/wdh/kaggle/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed_small.json"
        b_out_path = "/home/wdh/kaggle/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed_big.json"
        analyse.split_json(json_path, s_out_path, b_out_path)
    elif flag==8:
        json_path = "/home/dingmingxu/work/work/code/mmdetection_ding/bottle_flaw/round1/results/result_1.json"
        analyse.check_json_with_error_msg(json_path)
    elif flag==9:
        json_path = ['/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed.json']
        image_path = ['/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/images']
        json_file_path = '/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed_balance_cls_6_7_8_10_to_600.json'
        image_save_path= '/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/images_balance_cls_6_7_8_10_to_600'
        label = [7]
        analyse = Analyse(json_path, image_path)
        trans = Transform(Analyse=analyse, flag_det=True, flag_seg=False, flag_debug=True)
        analyse.add_aug_data(trans, add_num=600, aug_save_path = image_save_path,json_file_path=json_file_path, label=label)
    elif flag==10:
        json_path = ""
        analyse.check_json_with_error_msg(json_path)
    elif flag==11:
        json_path = '/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed.json'
        image_dir = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/images"
        save_json = '/home/wdh/kaggle/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed_crop.json'
        analyse.crop_img(json_path,image_dir,save_json)
    elif flag==12:
        json_file_list=[
            "/Users/urwlee/Documents/result_33_testb.json",
            "/Users/urwlee/Documents/result_43_testb.json",
            "/Users/urwlee/Documents/result_9_testb.json",
            "/Users/urwlee/Documents/result_hrnet_testb.json",
        ]
        result_file = "/Users/urwlee/Documents/result_33_43_9_hrnet_testb.json"
        analyse.mergecoco2coco(json_file_list,result_file)
    elif flag==13:
        round1_json = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed.json"
        round2_json = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/annotations_washed.json"
        save_json1 = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed_remove_678.json"
        save_json2 = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/annotations_washed_remove_678.json"
        analyse.mk_round2_json(round1_json,round2_json,save_json1,save_json2)
    elif flag==14:
        json = [
                '/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed_remove_678.json',
                ]
        pic_path = ["/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/images",
                    ]
        flag_water = False
        analyse.split_train_val(json,pic_path,flag_water=flag_water,ratio = 0.2)
    elif flag==15:
        image_dir=""
        analyse.gamma(image_dir)
    elif flag==16:
        image_dir = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/images"
        analyse.analyse_group(image_dir)
    elif flag==17:
        image_dir = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/images"
        analyse.compute_mean_std(image_dir)
    elif flag==18:
        json_path = '/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/annotations_washed_remove_678_1_2_3_4_5_9_10.json'
        bad_image_path = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/image_clean3"
        remove_label = [30]
        stay_label = [1,2,3,4,5,9,10]
        analyse.modify_json(json_path,remove_label,stay_label,bad_image_path=bad_image_path)
    elif flag==19:
        json_path = '/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/annotations.json'
        fg_path = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/images"
        bg_path = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/images_without_pinggai"
        save_path = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/images_11_duck"
        json_out_path = "/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round2_train_20200213/annotations_duck11to1213.json"
        # analyse.cut_pinggai(bg_path)
        analyse.duck_label11(json_path,fg_path,bg_path,save_path,json_out_path)
    elif flag==20:
        json_path = '/Users/urwlee/personal/Mystudy/GitHub/bottle_flaw/utils_round2/results/testab.json'
        analyse.choose(json_path)