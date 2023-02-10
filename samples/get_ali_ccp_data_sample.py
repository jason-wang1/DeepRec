

def read_ali_ccp_data():
    feat_name = ['101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124',
                 '125', '126', '127', '128', '129', '205', '206', '207', '210', '216',
                 '508', '509', '702', '853', '301']
    raw_feat = {'150_14', '509', '702', '110_14', '853', '109_14', '508', '127_14'}
    index_feats_dict = {}  # {index: {feat_field_id: [feat_id:value]}}
    with open("C:\\data\\Ali-CCP\\common_features_train.csv", 'rt', encoding='ascii') as f:
        i = 0
        for item in f:
            i += 1
            if i % 100000 == 0:
                print(f"read {i / 100000} * 10w")
            if item.endswith("\n"):
                item = item[:-1]
            item = item.split(",")
            feature_list = item[-1].split(chr(1))
            if len(feature_list) != int(item[1]):
                raise ValueError(f"error feature index: {item[0]}")
            if item[0] not in index_feats_dict:
                index_feats_dict[item[0]] = {}
            for feat in feature_list:
                tmp = feat.split(chr(2))
                feat_field = tmp[0]
                feat_id_value = tmp[1].split(chr(3))
                feat_id = feat_id_value[0]
                feat_value = feat_id_value[1]
                if feat_field not in index_feats_dict[item[0]]:
                    index_feats_dict[item[0]][feat_field] = []
                if feat_field in raw_feat:
                    index_feats_dict[item[0]][feat_field].append(feat_id+":"+feat_value)
                else:
                    index_feats_dict[item[0]][feat_field].append(feat_id)
            for feat_field, feat_id_value in index_feats_dict[item[0]].items():
                index_feats_dict[item[0]][feat_field] = "|".join(feat_id_value)

    with open("C:\\data\\Ali-CCP\\sample_skeleton_train.csv", 'rt', encoding='ascii') as fr:
        with open("C:\\data\\Ali-CCP\\sample_train_1w.csv", 'w') as fw:
            i = 0
            for item in fr:
                i += 1
                if i == 1000000:
                    break
                if item.endswith("\n"):
                    item = item[:-1]
                feats_dict = {}  # {feat_field_id: [feat_id:value]}
                item = item.split(",")
                feature_list = item[-1].split(chr(1))
                index = item[3]
                for feat in feature_list:
                    tmp = feat.split(chr(2))
                    feat_field = tmp[0]
                    feat_id_value = tmp[1].split(chr(3))
                    feat_id = feat_id_value[0]
                    feat_value = feat_id_value[1]
                    if feat_field not in feats_dict:
                        feats_dict[feat_field] = []
                    if feat_field in raw_feat:
                        feats_dict[feat_field].append(feat_id + ":" + feat_value)
                    else:
                        feats_dict[feat_field].append(feat_id)
                for feat_field, feat_id_value in feats_dict.items():
                    feats_dict[feat_field] = "|".join(feat_id_value[:20])
                feats_dict = {**feats_dict, **index_feats_dict.get(index, {})}
                feature = [feats_dict.get(e, '') for e in feat_name]
                feature = ",".join(feature)
                sample = f"{item[0]},{item[1]},{item[2]},{feature}\n"
                fw.write(sample)


if __name__ == '__main__':
    read_ali_ccp_data()
