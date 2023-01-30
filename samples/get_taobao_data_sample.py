base_dir = "C:\\data\\taobao_data"


def filter_behavior_log():
    i = 0
    with open(base_dir+"\\behavior_log.csv", 'r') as fr:
        with open(base_dir+"\\cvr_behavior_log.csv", 'w') as fw:
            for item in fr:
                i += 1
                if i % 10000000 == 0:
                    print(f"read {i/10000000} kw items")
                item_list = item.split(',')
                if "pv" not in item_list[2]:
                    fw.write(item)


def get_behavior_list():
    cvr_dict = {}  # {user_id: [[(time_stamp, cate, brand)], [(time_stamp, cate, brand)], [(time_stamp, cate, brand)]]}
    with open(base_dir+"\\cvr_behavior_log_100.csv", 'r') as fr:
        i = 0
        for item in fr:
            item = item.split(',')
            i += 1
            if i != 1 and len(item) == 5:
                if item[0] not in cvr_dict:
                    cvr_dict[item[0]] = [[], [], []]
                if item[2] == "cart":
                    cvr_dict[item[0]][0].append((item[1], item[3], item[4]))
                elif item[2] == "fav":
                    cvr_dict[item[0]][1].append((item[1], item[3], item[4]))
                elif item[2] == "buy":
                    cvr_dict[item[0]][2].append((item[1], item[3], item[4]))
    print("read data completed!")
    with open(base_dir+"\\cvr_behavior_feat_log.csv", 'w') as fw:
        for user, beh_list in cvr_dict.items():
            cart_list = sorted(beh_list[0], key=lambda x: x[0])
            fav_list = sorted(beh_list[1], key=lambda x: x[0])
            buy_list = sorted(beh_list[2], key=lambda x: x[0])

    pass


if __name__ == '__main__':
    # filter_behavior_log()
    pass