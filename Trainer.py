from sklearn.model_selection import train_test_split, KFold

from utils import *
from Experiment import *

def databowl2018_trainer(args, device, dataset_dir) :
    data = pd.read_csv(os.path.join(dataset_dir, 'stage1_train_labels.csv'))
    data_imageId = data.ImageId.drop_duplicates()

    train, test = train_test_split(data_imageId, test_size=0.15, shuffle=False, random_state=4321)
    test_frame = data.iloc[test.index[0]:]

    splits = KFold(n_splits=args.k_fold, shuffle=False).split(train)
    print("Apply {}-fold cross validation".format(args.k_fold))
    for fold, (train_idx, val_idx) in enumerate(splits):
        print("+" * 50, "Fold = {}({})".format(fold + 1, args.model_name), "+" * 50)
        print("#Train Fold = {}".format(len(train_idx)))
        print("#Val Fold = {}".format(len(val_idx)))
        print("#Test = {}".format(len(test)))

        train_fold_frame = data.iloc[train.iloc[train_idx].index[0]:train.iloc[train_idx].index[-1]]
        val_fold_frame = data.iloc[train.iloc[val_idx].index[0]:train.iloc[val_idx].index[-1]]

        experiment = DataBowl2018Experiment(device, args.data_type, dataset_dir, train_fold_frame, val_fold_frame, test_frame,
                                            args.image_size,
                                            args.batch_size,
                                            args.num_workers,
                                            args.model_name,
                                            args.epochs,
                                            args.optimizer,
                                            args.criterion,
                                            args.lr,
                                            args.momentum,
                                            args.weight_decay,
                                            args.pretrained_encoder,
                                            args.step,
                                            args.train,
                                            fold, args.save_path)

        if args.train :
            model, optimizer, history = experiment.fit()

            save_result(model, optimizer, history,
                        data_type=args.data_type,
                        image_size=args.image_size,
                        batch_size=args.batch_size,
                        model_name=args.model_name,
                        lr=args.lr, epochs=args.epochs, fold=fold,
                        save_path=args.save_path)
            del model
        else :
            model_dirs, save_model_path = get_save_path(data_type=args.data_type,
                                                        image_size=args.image_size,
                                                        batch_size=args.batch_size,
                                                        model_name=args.model_name,
                                                        lr=args.lr, epochs=args.epochs, fold=fold,
                                                        save_path=args.save_path)
            test_result = experiment.fit()

            save_metrics(test_result, model_dirs, save_model_path)

    # 5개의 폴드에 대한 성능의 평균 및 표준편차 계산
    save_total_metrics(data_type=args.data_type,
                       image_size=args.image_size,
                       batch_size=args.batch_size,
                       model_name=args.model_name,
                       lr=args.lr,
                       epochs=args.epochs,
                       k_fold=args.k_fold, save_path=args.save_path)