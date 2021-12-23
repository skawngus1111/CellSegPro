import warnings
warnings.filterwarnings('ignore')

from utils import *
from Trainer import databowl2018_trainer

def main(args) :
    print("hello world")
    device = get_deivce()
    dataset_rootdir = os.path.join('.', args.data_path)

    try:
        dataset_dir = os.path.join(dataset_rootdir, args.data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    if 'bowl' in args.data_type :
        databowl2018_trainer(args, device, dataset_dir)
    else :
        print("Wrong Data type : [DataBowl]")
        sys.exit()

if __name__=='__main__' :
    args = argparsing()
    # for model_name in ['unet'] :
    #     args.model_name = model_name
    #     args.pretrained_encoder = True if model_name == 'fcn8s' else False
    main(args)