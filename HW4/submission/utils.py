Config={}
Config['root_path']='polyvore_outfits'
Config['meta_file']='polyvore_item_metadata.json'
Config['checkpoint_path']='runs'
Config['checkpoint_file']='/home/adityan/PycharmProjects/EE599_Fall2020/HW4/mobilenet_classifier/runs/model.pth'
Config['resume_epoch']=0

Config['use_cuda']=False
Config['debug']=False
Config['num_epochs']=5
Config['batch_size']=4

Config['learning_rate']=0.001
Config['num_workers']=5
Config['use_custom_model']=True
Config['tensorboard_log']=True
Config['finetune']=True
