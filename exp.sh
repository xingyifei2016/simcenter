export CUDA_VISIBLE_DEVICES=2,3,4
python3 train.py --checkpoint-dir 'checkpoints/resnet18_0.001' --model-name resnet18 --lr 0.001
python3 train.py --checkpoint-dir 'checkpoints/resnet18_0.0001' --model-name resnet18 --lr 0.0001
python3 train.py --checkpoint-dir 'checkpoints/resnet18_0.01' --model-name resnet18 --lr 0.01
python3 train.py --checkpoint-dir 'checkpoints/resnet18_0.05' --model-name resnet18 --lr 0.05
python3 train.py --checkpoint-dir 'checkpoints/resnet18_0.0005' --model-name resnet18 --lr 0.0005
python3 train.py --checkpoint-dir 'checkpoints/resnet18_0.005' --model-name resnet18 --lr 0.005
python3 train.py --checkpoint-dir 'checkpoints/resnet34_0.001' --model-name resnet34 --lr 0.001 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet34_0.0001' --model-name resnet34 --lr 0.0001 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet34_0.01' --model-name resnet34 --lr 0.01 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet34_0.05' --model-name resnet34 --lr 0.05 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet34_0.0005' --model-name resnet34 --lr 0.0005 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet34_0.005' --model-name resnet34 --lr 0.005 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet50_0.001' --model-name resnet50 --lr 0.001 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet50_0.0001' --model-name resnet50 --lr 0.0001 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet50_0.01' --model-name resnet50 --lr 0.01 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet50_0.05' --model-name resnet50 --lr 0.05 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet50_0.0005' --model-name resnet50 --lr 0.0005 --multi-gpu 1
python3 train.py --checkpoint-dir 'checkpoints/resnet50_0.005' --model-name resnet50 --lr 0.005 --multi-gpu 1
