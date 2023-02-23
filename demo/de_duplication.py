import json
import os

# 标注去重，去掉没有标注的图片并重新标注，且删除文件夹中的图片文件
def clear_coco(coco_json, img_src, json_dst):
    # coco_json = json.load(open("___.json"))
    anno_imgid = list(set([i['image_id'] for i in coco_json['annotations']]))
    print(f"一共有{len(anno_imgid)}张有标注的图片。")
    data_images = []
    mark_id = 0
    sum_remove = 0
    for img in coco_json['images']:
        if img['id'] in anno_imgid:
            # 遍历annotation中所有id为img['id']的都更新为mark_id
            for anno in coco_json['annotations']:
                if anno['image_id'] == img['id']:
                    anno['image_id'] = mark_id

            img['id'] = mark_id
            data_images.append(img)
            mark_id += 1
        else:
            # 删除图片文件
            img_file = os.path.join(img_src, img['file_name'])
            if os.path.exists(img_file):
                os.unlink(img_file)
                sum_remove += 1
            continue
    print(f"一共移除了{sum_remove}张图片。")
    coco_json['images'] = data_images
    
    # save
    with open(json_dst, "w", encoding='utf-8') as sf:
        sf.write(json.dumps(coco_json, indent=2))
    return coco_json

if __name__ == '__main__':
    main_dir = './custum_datasets'
    # CLEAR
    with open(os.path.join(main_dir, 'annotations/instances_default.json')) as f:
        data = json.load(f)
    # print(data.keys())
    data = clear_coco(data,img_src= os.path.join(main_dir, 'images'), json_dst=os.path.join(main_dir, './annotations/instances_default_cleared.json'))