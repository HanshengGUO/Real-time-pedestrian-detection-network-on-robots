class DefaultConfig():
    #backbone
    backbone = "darknet19"
    #backbone = "slimdarknet19"
    mean = [0.483, 0.452, 0.401] 
    std = [0.226, 0.221, 0.221]
    #backbone = "resnet50"
    #mean = [0.485, 0.456, 0.406]  
    #std = [0.229, 0.224, 0.225]   
    pretrained=True
    freeze_stage_1=False # finetune剪枝模型的时候要关掉
    freeze_bn=False # finetune剪枝模型的时候要

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=20
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.5
    max_detection_boxes_num=1000
