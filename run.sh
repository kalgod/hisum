python3 main.py --model DETR --batch_size 128 > log/detr.log
python3 main.py --model MLP --batch_size 128 > log/mlp.log
python3 main.py --model PGL_SUM --batch_size 128 > log/pgl_sum.log
python3 main.py --model VASNet --batch_size 128 > log/vasnet.log
python3 main.py --model SL_module --batch_size 128 > log/sl_module.log