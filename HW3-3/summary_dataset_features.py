import sys

from lib.dataset import Generate_dataset_list

def argv_help():
    print('''dataset_feature = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
            \n'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            \n'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            \n'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
            \n'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            \n'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            \n'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            \n'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']''')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 summary_dataset_features.py [feature_selection]')
        print('if want to check features, please type python3 summary_dataset_features.py help')
        exit()

    if len(sys.argv) == 2 and sys.argv[1] == 'help':
        argv_help()
        exit()
    elif len(sys.argv) == 2 and sys.argv[1] != 'help':
        worker = Generate_dataset_list()
        worker.build_dataset(sys.argv[1])
    else:
        print('Usage: python3 summary_dataset_features.py [feature_selection]')
        print('if want to check features, please type python3 summary_dataset_features.py help')
        exit()


