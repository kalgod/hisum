import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import time
import h5py
from tqdm import tqdm
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pickle

def get_video_title(video_id):
    # 构造 YouTube 视频的 URL
    url = f'https://www.youtube.com/watch?v={video_id}'
    
    # 发送请求
    response = requests.get(url)
    print(url)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 解析网页内容
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找标题
        title = soup.find('title').text
        
        # 清理标题字符串
        title = title.replace(' - YouTube', '').strip()
        
        return title
    else:
        print(f"Error fetching the video: {response.status_code}")
        return None

def get_meta():
    meta_data = "dataset/metadata.csv"
    df = pd.read_csv(meta_data)
    names={}
    rows=list(df.itertuples())
    i=0
    for row in tqdm(rows):
        # print(row)
        video_id=row.video_id
        youtube_id=row.youtube_id
        name=get_video_title(youtube_id)
        # youtube_name=row.youtube_name
        names[video_id]=name
        with open('tmp.pkl', 'wb') as pickle_file:
            pickle.dump(names, pickle_file)
        i+=1
    return names

last_time = None

def generate_time_vector(n):
    global last_time
    
    # 如果是第一次调用，则使用当前时间
    if last_time is None:
        current_time = datetime.now()
    else:
        current_time = last_time

    # 创建一个空的时间列表
    time_vector = []

    for _ in range(n):
        # 将当前时间添加到时间列表，并格式化为字符串
        time_vector.append(current_time.strftime('%Y/%m/%d %H:%M:%S'))
        # 每次增加一秒
        current_time += timedelta(seconds=1)

    # 更新 last_time 变量为最后生成的时间
    last_time = current_time

    return np.array(time_vector)

def contains_nan_or_none(row):
    # 尝试将第二个元素转换为 float，如果不能转换则返回 True
    try:
        float(row[1])  # 检查第二列是否可以转换为浮点数
        return False
    except (ValueError, TypeError):
        return True

def main(args):
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output_100000.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./dataset/output.csv")
    # all_data,tmp_col=clean("./dataset/all_bw.csv","./Time-Series-Library/dataset/bandwidth/bandwidth.csv")
    video_data = h5py.File("./dataset/mr_hisum.h5", 'r')
    # names=get_meta()
    all_id=list(video_data.keys())
    res=[]
    for i in range (len(all_id)):
        if (i>=100000/250): break
        video_id=all_id[i]
        name="None"
        gtscore=np.array(video_data[video_id]['gtscore'])
        name_expanded = np.full(gtscore.shape, name)  # 创建与 gtscore 同形状的数组
        id_expanded = np.full(gtscore.shape, video_id)  # 创建与 gtscore 同形状的数组
        date=generate_time_vector(gtscore.shape[0])
        if (np.isnan(gtscore).any()): continue
        # 拼接 gtscore 和扩展后的 video_id
        # print(gtscore.shape,video_id_expanded.shape)
        cur=np.vstack((date,gtscore,id_expanded,name_expanded)).T
        # print(cur.shape)
        res.append(cur)  # 将拼接结果添加到列表中
    res=np.vstack(res)
    # res = np.array([row for row in res if not contains_nan_or_none(row)])
    print(res.shape)
    df = pd.DataFrame(res, columns=['date', 'OT', 'video_id', 'video_name'])
    # 将 DataFrame 导出为 CSV 文件
    csv_file_path = './dataset/gtscore_10w.csv'  # 你想要保存的 CSV 文件路径
    df.to_csv(csv_file_path, index=False)
    print(f"CSV 文件已保存到 {csv_file_path}")
    # all_data,train_idx,test_idx=split(args,"./dataset/output_100000.csv")
    # train_loader=CustomDataset(all_data,train_idx,args)
    # test_loader=CustomDataset(all_data,test_idx,args)
    # train_loader=DataLoader(train_loader,batch_size=args.batch,shuffle=True)
    # test_loader=DataLoader(test_loader,batch_size=args.batch,shuffle=False)

    # if (args.fea_len==1): model=DNNModel(args.in_len,1,args.out_len).to(args.device)
    # else: model=DNNModel(args.in_len,all_data.shape[1],args.out_len).to(args.device)
    # model.eval()
    # model=train(args,train_loader,test_loader,model)

    # model.load_state_dict(torch.load("./checkpoints/model_DNN_inlen_{}_outlen_{}_fealen_{}_epoch_{}.pth".format(args.in_len,args.out_len,args.fea_len,args.epochs-1)))
    # model=model.to(args.device)
    # test_loss,test_rmse,test_mae = test(args,test_loader,model)
    # print("Test Loss: {0:.7f} Test RMSE: {1:.7f} Test MAE: {2:.7f}".format(test_loss,test_rmse,test_mae))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple train function with args')
    parser.add_argument('-in_len', type=int, default=5, help='in len')
    parser.add_argument('-out_len', type=int, default=5, help='in len')
    parser.add_argument('-fea_len', type=int, default=1, help='in len')
    parser.add_argument('-batch', type=int, default=32, help='in len')
    parser.add_argument('-epochs', type=int, default=10, help='in len')
    parser.add_argument('-lr', type=float, default=5e-4, help='in len')
    parser.add_argument('-device', type=str, default="cuda", help='in len')

    args = parser.parse_args()
    main(args)
