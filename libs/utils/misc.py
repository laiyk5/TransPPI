def split_by_label(data, label):
    '''
    Split list of data by different labels.
        Parameters:
            data (list)
            label (list): 0 and 1 is accepcted.
        Returns:
            pos_data, neg_data
    '''
    pos_data = []
    neg_data = []
    for i in range(len(data)):
        if (label[i] == 1):
            pos_data.append(data[i])
        elif (label[i] == 0):
            neg_data.append(data[i])
        else:
            raise ValueError("Undefined label.")

    return pos_data, neg_data
